drop table if exists t2_omd_pv_conf.tmp_sus_income purge;
create temporary table t2_omd_pv_conf.tmp_sus_income as
select cif, Employment_Start_Date, Monthly_Income
from (
    select `cfcif#` as cif,
           cfesd7 as Employment_Start_Date,
           cfecur as Monthly_Income,
           row_number() over (partition by `cfcif#` 
                               order by cfesd7 desc) as rn
    from t15_pv_conf.sibs_cfzemp_v
    where partition_date = ${z_dperiod}
) x
where rn = 1;