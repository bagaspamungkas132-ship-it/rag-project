import pandas as pd

path = "/home/cdsw/parquet_output/result/trx_all_enriched.parquet"

cols = [
    "key1", "disposition_class", "monthly_income", "account_age",
    "flag_account_age_lt9", "flag_anomaly_income",
    "n_sameday_pass_3h_l3m", "n_sameday_pass_3h_l6m",
    "n_sameday_pass_1h_l3m", "n_sameday_pass_1h_l6m",
    "transfer_not_normal_l3m_pct", "transfer_not_normal_l6m_pct",
    "n_crypto", "n_in_10x_income", "flag_in_10x_income",
    "n_out_10x_income", "flag_out_10x_income",
    "n_new_beneficiary", "flag_new_beneficiary",
    "dpn_flag", "screening_flag", "flag"
]

df = pd.read_parquet(path)[cols].copy()

# Filter hanya FDS, lalu drop kolom flag
df_fds = df[df["flag"] == "FDS"].drop(columns=["flag"]).copy()

print("Total baris FDS:", len(df_fds))
print()

# Cek NaN per kolom
nan_summary = pd.DataFrame({
    "n_nan": df_fds.isna().sum(),
    "pct_nan": (df_fds.isna().sum() / len(df_fds) * 100).round(2)
}).sort_values("n_nan", ascending=False)

print("=== Ringkasan NaN per Kolom ===")
print(nan_summary)

# Kolom yang punya NaN saja (biar lebih ringkas kalau kolomnya banyak)
print("\n=== Kolom yang mengandung NaN ===")
print(nan_summary[nan_summary["n_nan"] > 0])