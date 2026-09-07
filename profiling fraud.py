import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib
matplotlib.use("Agg")  # supaya bisa save ke file tanpa GUI, cocok untuk CML
import matplotlib.pyplot as plt

# =========================================================
# 1. BACA DATA (parquet) & FILTER FDS
# =========================================================
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
df_fds = df[df["flag"] == "FDS"].drop(columns=["flag"]).copy()

print("Total baris FDS:", len(df_fds))

# =========================================================
# 2. HANDLE NaN
# =========================================================
flag_cols = [c for c in df_fds.columns if c.startswith("flag_") or c in ["dpn_flag", "screening_flag"]]
numeric_cols = [c for c in df_fds.columns if c not in flag_cols + ["key1", "disposition_class"]]

for c in flag_cols:
    if df_fds[c].isna().sum() > 0:
        df_fds[c] = df_fds[c].fillna(0)

num_imputer = SimpleImputer(strategy="median")
df_fds[numeric_cols] = num_imputer.fit_transform(df_fds[numeric_cols])

# =========================================================
# 3. SIAPKAN X, y
# =========================================================
target = "disposition_class"
feature_cols = [c for c in df_fds.columns if c not in ["key1", target]]

X = df_fds[feature_cols]
y = LabelEncoder().fit_transform(df_fds[target])  # 0=Not Suspicious, 1=Suspicious

# =========================================================
# 4. DECISION TREE UNTUK PROFILING SEGMEN
# =========================================================
tree_profile = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42
)
tree_profile.fit(X, y)

print("\n=== Struktur Pohon Segmentasi ===")
tree_rules = export_text(tree_profile, feature_names=list(X.columns))
print(tree_rules)

# =========================================================
# 5. STATISTIK PER SEGMEN (LEAF NODE)
# =========================================================
leaf_ids = tree_profile.apply(X)
df_fds_seg = df_fds.copy()
df_fds_seg["leaf_id"] = leaf_ids
df_fds_seg["is_suspicious"] = y

segment_summary = df_fds_seg.groupby("leaf_id").agg(
    n_data=("is_suspicious", "count"),
    n_suspicious=("is_suspicious", "sum"),
    pct_suspicious=("is_suspicious", "mean")
).reset_index()
segment_summary["pct_suspicious"] = (segment_summary["pct_suspicious"] * 100).round(2)
segment_summary = segment_summary.sort_values("pct_suspicious", ascending=False)

print("\n=== Ringkasan Segmen (Leaf Node), Urut dari Paling Suspicious ===")
print(segment_summary)

# =========================================================
# 6. VISUALISASI POHON
# =========================================================
plt.figure(figsize=(22, 10))
plot_tree(
    tree_profile,
    feature_names=list(X.columns),
    class_names=["Not Suspicious", "Suspicious"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.tight_layout()
plt.savefig("/home/cdsw/query/segment_tree.png", dpi=150)
print("\nGambar pohon disimpan di: /home/cdsw/query/segment_tree.png")

# =========================================================
# 7. PROFIL DESKRIPTIF SEGMEN PALING SUSPICIOUS
# =========================================================
top_segment_id = segment_summary.iloc[0]["leaf_id"]
top_segment_data = df_fds_seg[df_fds_seg["leaf_id"] == top_segment_id]

print(f"\n=== Profil Segmen Paling Suspicious (leaf_id={int(top_segment_id)}) ===")
print(f"Jumlah data: {len(top_segment_data)}, % Suspicious: {top_segment_data['is_suspicious'].mean()*100:.1f}%")
print("\nRata-rata nilai fitur di segmen ini vs keseluruhan data:")
comparison = pd.DataFrame({
    "segmen_top": top_segment_data[feature_cols].mean(),
    "keseluruhan_data": df_fds_seg[feature_cols].mean()
})
print(comparison.round(3))