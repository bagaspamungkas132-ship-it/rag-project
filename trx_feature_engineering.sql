-- ============================================================================
-- FRAUD FEATURE ENGINEERING dari trx_model_backtesting_daily_v2
-- Jalankan di DBeaver (koneksi Hive). SESUAIKAN nama tabel yang ditandai <GANTI>
-- ============================================================================
-- CATATAN PENTING SOAL LEAKAGE:
--   Semua query di bawah filter "activity_date < alert_month" -- HANYA pakai
--   transaksi SEBELUM tanggal alert. Jangan pernah hilangkan filter ini,
--   walau cuma buat "coba-coba" -- sekali kebiasaan itu masuk, gampang lolos
--   ke versi final tanpa sadar.
--
-- CATATAN PERFORMANCE: tabel transaksi mentah biasanya BESAR. Kalau ada kolom
-- partition (terlihat ada `partition_date` di tabelmu), tambahkan filter
-- partition_date di WHERE clause supaya query tidak full-scan semua histori.
-- ============================================================================


-- ----------------------------------------------------------------------------
-- STEP 0: Daftar unik (key1, accountid, alert_month) dari alert yang mau kita
--         kasih fitur baru. GANTI dengan tabel/sumber FDS kamu yang sebenarnya.
-- ----------------------------------------------------------------------------
-- CREATE TABLE t2_omd_pv_conf.tmp_alerts AS
-- SELECT DISTINCT key1, accountid, alert_month
-- FROM <GANTI: tabel_sumber_fds>;


-- ----------------------------------------------------------------------------
-- STEP 1: Ringkasan transaksi mentah per akun per alert (point-in-time, 3 bulan
--         ke belakang). Ini versi "amount" dan "channel" yang belum pernah kita
--         punya sebelumnya -- selama ini cuma pakai count/pct, belum pakai
--         nominal transaksi asli.
-- ----------------------------------------------------------------------------
CREATE TABLE t2_omd_pv_conf.trx_features_pointintime AS
WITH trx_window AS (
    SELECT
        a.key1,
        a.accountid,
        a.alert_month,
        t.transaction_amount_idr,
        t.direction,
        t.type_mapping,
        t.counterpart_bank,
        t.counterpart_accountid,
        t.event_type,
        t.transaction_status,
        t.transaction_datetime
    FROM t2_omd_pv_conf.tmp_alerts a
    JOIN t2_omd_pv_conf.trx_model_backtesting_daily_v2 t
      ON t.accountid = a.accountid
     AND t.activity_date >= add_months(a.alert_month, -3)   -- window 3 bulan, samakan dgn l3m yg sudah ada
     AND t.activity_date <  a.alert_month                   -- WAJIB: hanya transaksi SEBELUM alert
)
SELECT
    key1,
    accountid,
    alert_month,
    COUNT(*)                                                                   AS n_trx_l3m,
    SUM(transaction_amount_idr)                                                AS total_amount_l3m,
    AVG(transaction_amount_idr)                                                AS avg_amount_l3m,
    MAX(transaction_amount_idr)                                                AS max_amount_l3m,
    SUM(CASE WHEN direction = 'OUT' THEN transaction_amount_idr ELSE 0 END)    AS total_amount_out_l3m,
    SUM(CASE WHEN direction = 'IN'  THEN transaction_amount_idr ELSE 0 END)    AS total_amount_in_l3m,
    COUNT(DISTINCT counterpart_accountid)                                      AS n_unique_counterparty_l3m,
    COUNT(DISTINCT counterpart_bank)                                          AS n_unique_bank_l3m,
    COUNT(DISTINCT event_type)                                                 AS n_channel_l3m,
    SUM(CASE WHEN transaction_status <> 'SUCCESS' THEN 1 ELSE 0 END)          AS n_failed_trx_l3m,
    SUM(CASE WHEN HOUR(transaction_datetime) < 6
              OR HOUR(transaction_datetime) >= 22 THEN 1 ELSE 0 END)          AS n_trx_odd_hour_l3m
FROM trx_window
GROUP BY key1, accountid, alert_month;

-- Insight yang bisa diambil dari tabel ini:
--   - total_amount_out_l3m vs total_amount_in_l3m -> rasio arus kas keluar-masuk
--   - n_unique_counterparty_l3m tinggi = uang disebar ke banyak tujuan (mirip layering)
--   - n_unique_bank_l3m tinggi = transfer lintas bank banyak (juga pola layering)
--   - n_channel_l3m > 1 = pakai banyak kanal (internet banking + mobile + ATM dst)
--   - n_trx_odd_hour_l3m = transaksi jam ganjil (tengah malam), sering dipakai
--     buat menghindari perhatian


-- ----------------------------------------------------------------------------
-- STEP 2: FITUR JARINGAN -- popularitas counterparty (sinyal mule account)
--         "Apakah akun ini pernah kirim uang ke rekening yang JUGA banyak
--          dipakai akun-akun lain?" -- itu pola klasik mule/collector account.
-- ----------------------------------------------------------------------------

-- 2a. Hitung berapa banyak accountid BERBEDA yang kirim ke tiap counterpart_accountid,
--     per bulan snapshot (biar bisa di-join point-in-time nanti)
CREATE TABLE t2_omd_pv_conf.counterparty_popularity_monthly AS
SELECT
    counterpart_accountid,
    date_format(activity_date, 'yyyy-MM') AS bulan_snapshot,
    COUNT(DISTINCT accountid)             AS n_unique_sender_bulan_ini
FROM t2_omd_pv_conf.trx_model_backtesting_daily_v2
WHERE direction = 'OUT'
  AND counterpart_accountid IS NOT NULL
GROUP BY counterpart_accountid, date_format(activity_date, 'yyyy-MM');

-- 2b. Join ke tiap alert: dari semua counterparty yang dipakai akun ini di window
--     l3m, ambil yang PALING populer (paling banyak dipakai akun lain).
--     Catatan: perbandingan bulan snapshot di sini pakai bulan SEBELUM alert_month,
--     supaya tetap point-in-time.
CREATE TABLE t2_omd_pv_conf.trx_features_network AS
WITH trx_window AS (
    SELECT
        a.key1,
        a.accountid,
        a.alert_month,
        t.counterpart_accountid,
        date_format(t.activity_date, 'yyyy-MM') AS bulan_trx
    FROM t2_omd_pv_conf.tmp_alerts a
    JOIN t2_omd_pv_conf.trx_model_backtesting_daily_v2 t
      ON t.accountid = a.accountid
     AND t.activity_date >= add_months(a.alert_month, -3)
     AND t.activity_date <  a.alert_month
     AND t.direction = 'OUT'
     AND t.counterpart_accountid IS NOT NULL
)
SELECT
    tw.key1,
    tw.accountid,
    tw.alert_month,
    MAX(cp.n_unique_sender_bulan_ini) AS max_counterparty_popularity_l3m,
    AVG(cp.n_unique_sender_bulan_ini) AS avg_counterparty_popularity_l3m
FROM trx_window tw
JOIN t2_omd_pv_conf.counterparty_popularity_monthly cp
  ON cp.counterpart_accountid = tw.counterpart_accountid
 AND cp.bulan_snapshot        = tw.bulan_trx
GROUP BY tw.key1, tw.accountid, tw.alert_month;

-- Insight: max_counterparty_popularity_l3m tinggi = akun ini pernah kirim ke
-- rekening yang JUGA jadi tujuan banyak akun lain di bulan yang sama --
-- indikasi rekening tujuan itu kemungkinan "collector"/mule account.


-- ----------------------------------------------------------------------------
-- STEP 3: Gabungkan semua ke satu tabel final, siap di-export/dibaca dari VSCode
-- ----------------------------------------------------------------------------
CREATE TABLE t2_omd_pv_conf.trx_features_final AS
SELECT
    p.key1, p.accountid, p.alert_month,
    p.n_trx_l3m, p.total_amount_l3m, p.avg_amount_l3m, p.max_amount_l3m,
    p.total_amount_out_l3m, p.total_amount_in_l3m,
    p.n_unique_counterparty_l3m, p.n_unique_bank_l3m, p.n_channel_l3m,
    p.n_failed_trx_l3m, p.n_trx_odd_hour_l3m,
    n.max_counterparty_popularity_l3m, n.avg_counterparty_popularity_l3m
FROM t2_omd_pv_conf.trx_features_pointintime p
LEFT JOIN t2_omd_pv_conf.trx_features_network n
  ON  n.key1       = p.key1
  AND n.accountid  = p.accountid
  AND n.alert_month = p.alert_month;

-- Cek hasilnya dulu sebelum lanjut ke VSCode:
-- SELECT * FROM t2_omd_pv_conf.trx_features_final LIMIT 20;
-- SELECT COUNT(*) FROM t2_omd_pv_conf.trx_features_final;  -- harus sama dgn jumlah baris FDS
