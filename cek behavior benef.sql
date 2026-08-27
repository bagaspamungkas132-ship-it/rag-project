-- =====================================================================
-- FLAG (HiveQL): Lawan transaksi yang sebelumnya tidak aktif, sekarang
-- jadi sering/rajin bertransaksi dengan nasabah (Screening Q#2)
--
-- Logic sama seperti versi BigQuery:
--   - Current window  = 3 bulan & 6 bulan terakhir (dihitung sekaligus,
--     dibedakan kolom window_months, pakai LATERAL VIEW explode)
--   - Baseline window = SELALU 3 bulan tepat sebelum current window mulai
--       * window_months = 3  -> baseline = bulan ke(-6) s/d ke(-3)
--       * window_months = 6  -> baseline = bulan ke(-9) s/d ke(-6)
--   - Counterpart dianggap "baru & jadi rajin" jika:
--       a) tidak muncul sama sekali di baseline window, DAN
--       b) di current window jumlah transaksi >= freq_threshold, DAN
--       c) counterpart_bank IN ('OCBC','OVERBOOKING') -> hanya di sini
--          kita punya visibilitas riwayat lengkap.
--   - counterpart_bank lain -> ditandai UNKNOWN_OTHER_BANK, tidak dihitung
--     sebagai "baru & jadi rajin" karena data historisnya tidak kita punya.
--
-- PERBEDAAN UTAMA vs BigQuery:
--   - Tidak ada DECLARE / UNNEST -> pakai SET hivevar (opsional) dan
--     LATERAL VIEW explode(array(...)).
--   - DATE_SUB(date, INTERVAL n MONTH)  -> add_months(date, -n)
--   - DATE(col)                          -> to_date(col)
--   - COUNTIF(cond)                      -> SUM(CASE WHEN cond THEN 1 ELSE 0 END)
--   - ARRAY_AGG(x IGNORE NULLS)          -> collect_list(CASE WHEN ... THEN x END)
--   - ANY_VALUE(col)                     -> MAX(col) (asumsi nilainya konsisten
--                                            per accountid+counterpart)
--   - Range join antar tabel kecil (params) dihindari; window_months
--     langsung di-explode ke tabel transaksi via LATERAL VIEW supaya
--     tidak kena batasan cartesian-product join di Hive strict mode.
--
-- ASUMSI / YANG PERLU DISESUAIKAN:
--   1. Nama kolom `counterpart_bank` -> ganti sesuai nama kolom asli.
--   2. Query menghitung transaksi IN & OUT sekaligus. Untuk fokus ke
--      transaksi masuk saja, tambahkan filter di CTE `exploded`:
--        AND t.direction = 'IN'
--   3. freq_threshold = 3 -> ambang "sering/rajin", sesuaikan sesuai
--      kebijakan/model kamu (bisa lewat hivevar di bawah).
-- =====================================================================

-- (opsional) parameter yang bisa diubah tanpa edit query, jalankan
-- lewat Hive CLI / Beeline sebelum query:
-- SET hivevar:freq_threshold=3;

WITH exploded AS (
  SELECT
    t.accountid,
    t.counterpart_accountid,
    t.counterpart_bank,                          -- << sesuaikan nama kolom kalau beda
    t.transaction_amount_idr,
    to_date(t.transaction_datetime) AS trx_date,
    window_months
  FROM t2_omd_pv_conf.trx_model_backtesting_daily_v2 t
  LATERAL VIEW explode(array(3, 6)) w AS window_months
  WHERE t.transaction_status = 'SUCCESS'
    AND t.counterpart_accountid IS NOT NULL
    AND to_date(t.transaction_datetime) >= add_months(current_date(), -9)  -- cover kasus terjauh (6bln window + 3bln baseline)
    AND to_date(t.transaction_datetime) <  current_date()
),

current_period AS (
  SELECT
    window_months,
    accountid,
    counterpart_accountid,
    MAX(counterpart_bank)         AS counterpart_bank,
    COUNT(*)                      AS txn_count_current,
    SUM(transaction_amount_idr)   AS amount_current
  FROM exploded
  WHERE trx_date >= add_months(current_date(), -window_months)
    AND trx_date <  current_date()
  GROUP BY window_months, accountid, counterpart_accountid
),

baseline_period AS (
  SELECT
    window_months,
    accountid,
    counterpart_accountid,
    COUNT(*) AS txn_count_baseline
  FROM exploded
  WHERE trx_date >= add_months(current_date(), -(window_months + 3))
    AND trx_date <  add_months(current_date(), -window_months)
  GROUP BY window_months, accountid, counterpart_accountid
),

flagged AS (
  SELECT
    c.window_months,
    c.accountid,
    c.counterpart_accountid,
    c.counterpart_bank,
    c.txn_count_current,
    c.amount_current,
    COALESCE(bl.txn_count_baseline, 0) AS txn_count_baseline,
    CASE
      WHEN c.counterpart_bank NOT IN ('OCBC', 'OVERBOOKING') THEN 'UNKNOWN_OTHER_BANK'
      WHEN COALESCE(bl.txn_count_baseline, 0) = 0
           AND c.txn_count_current >= 3                       -- << ganti 3 dengan ${hivevar:freq_threshold} kalau pakai parameter
           THEN 'NEW_NOW_FREQUENT'
      ELSE 'NOT_FLAGGED'
    END AS counterpart_flag
  FROM current_period c
  LEFT JOIN baseline_period bl
    ON  c.window_months = bl.window_months
    AND c.accountid = bl.accountid
    AND c.counterpart_accountid = bl.counterpart_accountid
)

SELECT
  accountid,
  window_months,
  COUNT(DISTINCT counterpart_accountid) AS total_distinct_counterparty,
  SUM(CASE WHEN counterpart_flag = 'NEW_NOW_FREQUENT'   THEN 1 ELSE 0 END) AS new_now_frequent_counterparty_count,
  SUM(CASE WHEN counterpart_flag = 'UNKNOWN_OTHER_BANK' THEN 1 ELSE 0 END) AS other_bank_counterparty_count,
  collect_list(CASE WHEN counterpart_flag = 'NEW_NOW_FREQUENT' THEN counterpart_accountid END) AS new_now_frequent_counterparty_list,
  CASE
    WHEN SUM(CASE WHEN counterpart_flag = 'NEW_NOW_FREQUENT' THEN 1 ELSE 0 END) >= 1 THEN true
    ELSE false
  END AS flag_suspicious_counterparty_pattern
FROM flagged
GROUP BY accountid, window_months
ORDER BY accountid, window_months;