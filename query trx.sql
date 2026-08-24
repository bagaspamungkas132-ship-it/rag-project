WITH trx_in AS (
    SELECT 
        accountid,
        transaction_datetime AS in_datetime,
        transaction_amount_idr AS in_amount,
        ROW_NUMBER() OVER (PARTITION BY accountid ORDER BY transaction_datetime) AS in_seq
    FROM t2_omd_pv_conf.tmp_fnsh_trx_all_fds_mdl
    WHERE direction = 'IN'
      AND transaction_status = 'SUCCESS'
),
trx_out AS (
    SELECT 
        accountid,
        transaction_datetime AS out_datetime,
        transaction_amount_idr AS out_amount,
        counterpart_accountid
    FROM t2_omd_pv_conf.tmp_fnsh_trx_all_fds_mdl
    WHERE direction = 'OUT'
      AND transaction_status = 'SUCCESS'
),
-- semua OUT yang jatuh dalam window 3 jam setelah tiap IN, di hari yang sama
in_out_window AS (
    SELECT 
        i.accountid,
        i.in_datetime,
        i.in_amount,
        o.out_datetime,
        o.out_amount,
        o.counterpart_accountid
    FROM trx_in i
    JOIN trx_out o
        ON i.accountid = o.accountid
       AND o.out_datetime > i.in_datetime
       AND o.out_datetime <= i.in_datetime + INTERVAL '3 hours'
       AND DATE(o.out_datetime) = DATE(i.in_datetime)
),
-- agregasi: total OUT & jumlah counterparty per event IN
accumulation_check AS (
    SELECT 
        accountid,
        in_datetime,
        in_amount,
        SUM(out_amount) AS total_out_amount,
        COUNT(*) AS out_txn_count,
        COUNT(DISTINCT counterpart_accountid) AS distinct_out_counterparties,
        ROUND(
            LEAST(in_amount, SUM(out_amount))::numeric 
            / NULLIF(GREATEST(in_amount, SUM(out_amount)), 0) * 100, 2
        ) AS total_similarity_pct
    FROM in_out_window
    GROUP BY accountid, in_datetime, in_amount
),
accumulation_flagged AS (
    SELECT *,
        CASE WHEN total_similarity_pct >= 90 THEN 1 ELSE 0 END AS is_accumulation_match
    FROM accumulation_check
),
rapid_count AS (
    SELECT 
        accountid,
        COUNT(*) AS in_event_count,                        -- jumlah event IN yang punya OUT susulan
        SUM(is_accumulation_match) AS accumulation_match_count,
        SUM(out_txn_count) AS total_out_txn,
        MAX(distinct_out_counterparties) AS max_distinct_counterparties,
        MIN(in_datetime) AS first_event,
        MAX(in_datetime) AS last_event,
        (MAX(in_datetime) - MIN(in_datetime)) AS period_span
    FROM accumulation_flagged
    GROUP BY accountid
)
SELECT 
    accountid,
    in_event_count,
    accumulation_match_count,
    total_out_txn,
    max_distinct_counterparties,
    first_event,
    last_event,
    period_span,
    CASE 
        WHEN in_event_count BETWEEN 5 AND 10
             AND period_span BETWEEN INTERVAL '3 months' AND INTERVAL '6 months'
             AND accumulation_match_count >= 5
        THEN 'FLAG_SAME_DAY_PASS'
        ELSE NULL
    END AS flag_same_day_pass
FROM rapid_count
WHERE in_event_count >= 5
ORDER BY accumulation_match_count DESC, in_event_count DESC;