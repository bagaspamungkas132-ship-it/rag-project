drop table if exists t2_omd_pv_conf.tmp_income_anomaly purge;
create temporary table t2_omd_pv_conf.tmp_income_anomaly as
with base as (
  select
    t1.customerid,
    -- job kosong/null dikelompokkan jadi satu bucket biar tetap kehitung, bukan hilang
    coalesce(nullif(trim(t1.customerjob), ''), 'UNKNOWN_JOB') as customerjob,
    t2.Monthly_Income
  from t2_omd_pv_conf.tmp_mi_all_income_clust t1
  left join t2_omd_pv_conf.tmp_sus_income t2
    on t1.customerid = t2.cif
),

-- stats hanya dihitung dari row yang income-nya valid (>0, not null)
job_stats as (
  select
    customerjob,
    percentile_approx(Monthly_Income, 0.25) as q1,
    percentile_approx(Monthly_Income, 0.50) as median_income,
    percentile_approx(Monthly_Income, 0.75) as q3,
    count(1) as n_customer
  from base
  where Monthly_Income is not null
    and Monthly_Income > 0
  group by customerjob
),

job_stats_iqr as (
  select
    *,
    (q3 - q1) as iqr,
    (q1 - 1.5 * (q3 - q1)) as lower_bound,
    (q3 + 1.5 * (q3 - q1)) as upper_bound
  from job_stats
)

select
  b.customerid,
  b.customerjob,
  b.Monthly_Income,
  s.median_income,
  s.q1,
  s.q3,
  s.lower_bound,
  s.upper_bound,
  s.n_customer,
  case
    -- income null / tidak valid -> flag terpisah, bukan NORMAL/ANOMALY
    when b.Monthly_Income is null then 'NO_INCOME_DATA'
    when b.Monthly_Income <= 0 then 'INVALID_INCOME'
    -- job unknown tetap bisa dinilai kalau statsnya ada, tapi ditandai low-confidence
    when b.customerjob = 'UNKNOWN_JOB' and s.n_customer >= 5 then
      case when b.Monthly_Income < s.lower_bound or b.Monthly_Income > s.upper_bound
           then 'ANOMALY_LOW_CONFIDENCE' else 'NORMAL_LOW_CONFIDENCE' end
    when s.n_customer is null or s.n_customer < 5 then 'INSUFFICIENT_DATA'
    when b.Monthly_Income < s.lower_bound or b.Monthly_Income > s.upper_bound then 'ANOMALY'
    else 'NORMAL'
  end as income_flag
from base b
left join job_stats_iqr s
  on b.customerjob = s.customerjob
;