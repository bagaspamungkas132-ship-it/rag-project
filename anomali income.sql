-- =========================================================
-- 1. CLEAN DATA: filter income valid & cap outlier ekstrem (P99.9)
-- =========================================================
drop table if exists t2_omd_pv_conf.tmp_income_clean purge;
create temporary table t2_omd_pv_conf.tmp_income_clean as
with base as (
  select
    t1.customerid,
    coalesce(nullif(trim(t1.customerjob), ''), 'UNKNOWN_JOB') as customerjob,
    t1.customerjoblevel,
    t1.employment_start_date,
    t1.monthly_income,
    -- tenure dalam tahun, dari employment_start_date sampai hari ini
    datediff(current_date(), to_date(t1.employment_start_date)) / 365.25 as tenure_years
  from t2_omd_pv_conf.tmp_mi_all_income_clust t1
  where t1.monthly_income is not null
    and t1.monthly_income > 0
),
cap as (
  select percentile_approx(monthly_income, 0.999) as cap_value
  from base
)
select
  b.*,
  ln(b.monthly_income) as log_income,
  greatest(b.tenure_years, 0) as tenure_years_clean
from base b
cross join cap c
where b.monthly_income <= c.cap_value;   -- buang outlier ekstrem seperti kasus 1.999.999.999

-- =========================================================
-- 2. HITUNG REGRESI LINEAR MANUAL PER SEGMEN (job + joblevel)
--    slope = cov(x,y) / var(x)   ->   pakai regr_slope/regr_intercept Hive
-- =========================================================
drop table if exists t2_omd_pv_conf.tmp_income_regression purge;
create temporary table t2_omd_pv_conf.tmp_income_regression as
select
  t.*,
  count(1) over (partition by customerjob, customerjoblevel) as n_segment,
  -- Hive punya fungsi regr_slope & regr_intercept bawaan (Hive 2.x+)
  regr_slope(log_income, tenure_years_clean) over (partition by customerjob, customerjoblevel) as slope,
  regr_intercept(log_income, tenure_years_clean) over (partition by customerjob, customerjoblevel) as intercept,
  -- fallback: rata-rata log_income per segmen (dipakai kalau segmen terlalu kecil utk regresi)
  avg(log_income) over (partition by customerjob, customerjoblevel) as avg_log_income_segment
from t2_omd_pv_conf.tmp_income_clean t;

-- =========================================================
-- 3. HITUNG PREDICTED INCOME & RESIDUAL
--    Kalau n_segment < 15, fallback ke rata-rata segmen (tanpa tenure)
-- =========================================================
drop table if exists t2_omd_pv_conf.tmp_income_residual purge;
create temporary table t2_omd_pv_conf.tmp_income_residual as
select
  customerid,
  customerjob,
  customerjoblevel,
  monthly_income,
  tenure_years_clean,
  n_segment,
  case
    when n_segment >= 15 and slope is not null
      then (intercept + slope * tenure_years_clean)
    else avg_log_income_segment
  end as predicted_log_income,
  log_income - (
    case
      when n_segment >= 15 and slope is not null
        then (intercept + slope * tenure_years_clean)
      else avg_log_income_segment
    end
  ) as residual
from t2_omd_pv_conf.tmp_income_regression;

-- =========================================================
-- 4. HITUNG MEDIAN & MAD DARI RESIDUAL PER SEGMEN
--    (median_residual & MAD dihitung via percentile_approx, robust thd outlier)
-- =========================================================
drop table if exists t2_omd_pv_conf.tmp_income_mad purge;
create temporary table t2_omd_pv_conf.tmp_income_mad as
with resid_median as (
  select
    customerjob,
    customerjoblevel,
    percentile_approx(residual, 0.5) as median_residual
  from t2_omd_pv_conf.tmp_income_residual
  group by customerjob, customerjoblevel
),
resid_with_median as (
  select
    r.*,
    m.median_residual,
    abs(r.residual - m.median_residual) as abs_dev
  from t2_omd_pv_conf.tmp_income_residual r
  join resid_median m
    on r.customerjob = m.customerjob
   and r.customerjoblevel = m.customerjoblevel
),
mad_calc as (
  select
    customerjob,
    customerjoblevel,
    percentile_approx(abs_dev, 0.5) as mad
  from resid_with_median
  group by customerjob, customerjoblevel
)
select
  rwm.customerid,
  rwm.customerjob,
  rwm.customerjoblevel,
  rwm.monthly_income,
  rwm.tenure_years_clean,
  rwm.residual,
  rwm.median_residual,
  mc.mad,
  -- Modified Z-score, hindari div by zero kalau MAD = 0
  0.6745 * (rwm.residual - rwm.median_residual) / nullif(mc.mad, 0) as modified_z_income
from resid_with_median rwm
join mad_calc mc
  on rwm.customerjob = mc.customerjob
 and rwm.customerjoblevel = mc.customerjoblevel;

-- =========================================================
-- 5. FLAG FINAL
-- =========================================================
drop table if exists t2_omd_pv_conf.tmp_income_anomaly_v2 purge;
create table t2_omd_pv_conf.tmp_income_anomaly_v2 as
select
  *,
  case when abs(modified_z_income) > 3.5 then 1 else 0 end as flag_anomaly_income_v2
from t2_omd_pv_conf.tmp_income_mad;

-- =========================================================
-- 6. CEK HASIL
-- =========================================================
select
  flag_anomaly_income_v2,
  count(1) as n_customer,
  round(count(1) * 100.0 / sum(count(1)) over (), 2) as pct
from t2_omd_pv_conf.tmp_income_anomaly_v2
group by flag_anomaly_income_v2;

-- Validasi manual untuk segmen PSW-Pegawai Swasta / 08-Staff
select
  monthly_income, tenure_years_clean, residual, modified_z_income, flag_anomaly_income_v2
from t2_omd_pv_conf.tmp_income_anomaly_v2
where customerjob = 'PSW-Pegawai Swasta'
  and customerjoblevel = '08-Staff'
order by modified_z_income desc
limit 20;