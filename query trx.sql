WITH trx_in AS (
    SELECT 
        accountid,
        transaction_datetime AS in_datetime,
        transaction_amount_idr AS in_amount
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
rapid_pairs AS (
    SELECT 
        i.accountid,
        i.in_datetime,
        o.out_datetime,
        i.in_amount,
        o.out_amount,
        o.counterpart_accountid,
        EXTRACT(EPOCH FROM (o.out_datetime - i.in_datetime)) / 3600 AS hours_diff
    FROM trx_in i
    JOIN trx_out o
        ON i.accountid = o.accountid
       AND o.out_datetime > i.in_datetime
       AND o.out_datetime <= i.in_datetime + INTERVAL '3 hours'
       AND DATE(o.out_datetime) = DATE(i.in_datetime)   -- hari yang sama
),
rapid_count AS (
    SELECT 
        accountid,
        COUNT(*) AS rapid_move_count,
        COUNT(DISTINCT counterpart_accountid) AS distinct_counterparties,
        MIN(in_datetime) AS first_event,
        MAX(out_datetime) AS last_event,
        (MAX(out_datetime) - MIN(in_datetime)) AS period_span
    FROM rapid_pairs
    GROUP BY accountid
)
SELECT 
    accountid,
    rapid_move_count,
    distinct_counterparties,
    first_event,
    last_event,
    period_span,
    CASE 
        WHEN rapid_move_count BETWEEN 5 AND 10 
             AND period_span BETWEEN INTERVAL '3 months' AND INTERVAL '6 months'
        THEN 'FLAG_SAME_DAY_PASS'
        ELSE NULL
    END AS flag_same_day_pass
FROM rapid_count
WHERE rapid_move_count >= 5
ORDER BY rapid_move_count DESC;