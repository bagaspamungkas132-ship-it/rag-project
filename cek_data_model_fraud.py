import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# =========================================================
# 1. LOAD & FILTER FDS
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

target = "disposition_class"
y = LabelEncoder().fit_transform(df_fds[target])  # 0=Not Suspicious, 1=Suspicious (alfabetis)

# =========================================================
# 3. POIN 1 — CEK POTENSI LEAKAGE PADA dpn_flag
#    (dan flag lain yang mencurigakan sebagai "hasil" bukan "input")
# =========================================================
print("\n=== Cek Leakage: Crosstab flag vs target ===")
for c in ["dpn_flag", "screening_flag"]:
    print(f"\n--- {c} vs {target} ---")
    print(pd.crosstab(df_fds[c], df_fds[target], normalize="index").round(3))

print("""
CATATAN: Crosstab di atas hanya bantu lihat asosiasi statistik.
Kalau dpn_flag == 1 HAMPIR SELALU beririsan dengan disposition_class == Suspicious
(atau sebaliknya sangat dominan di satu kelas), ini indikasi kuat bahwa dpn_flag
mungkin di-generate SETELAH proses disposisi (leakage). WAJIB dikonfirmasi ke
pemilik proses/data dictionary sebelum lanjut — bukan cuma dari angka statistik.
""")

# =========================================================
# 4. POIN 2 — FEATURE SELECTION
#    Buang fitur dengan importance mendekati nol di RF, permutation, DAN logreg
# =========================================================
feature_cols_all = [c for c in df_fds.columns if c not in ["key1", target]]
X_all = df_fds[feature_cols_all]

# Fit RF awal untuk permutation importance
rf_check = RandomForestClassifier(n_estimators=300, max_depth=5, class_weight="balanced", random_state=42)
rf_check.fit(X_all, y)
perm = permutation_importance(rf_check, X_all, y, n_repeats=30, random_state=42, scoring="roc_auc")

# Fit logreg untuk cek koefisien
X_scaled_all = StandardScaler().fit_transform(X_all)
logreg_check = LogisticRegression(max_iter=1000, class_weight="balanced")
logreg_check.fit(X_scaled_all, y)

selection_df = pd.DataFrame({
    "feature": feature_cols_all,
    "rf_importance": rf_check.feature_importances_,
    "permutation_importance": perm.importances_mean,
    "logreg_coef_abs": np.abs(logreg_check.coef_[0])
})

# Threshold: buang fitur yang lemah di SEMUA metode sekaligus
selection_df["weak_in_perm"] = selection_df["permutation_importance"] <= 0.001
selection_df["weak_in_rf"] = selection_df["rf_importance"] < selection_df["rf_importance"].median() * 0.3
selection_df["weak_in_logreg"] = selection_df["logreg_coef_abs"] < 0.05

selection_df["drop_candidate"] = (
    selection_df["weak_in_perm"] & selection_df["weak_in_rf"] & selection_df["weak_in_logreg"]
)

print("\n=== Feature Selection Summary ===")
print(selection_df.sort_values("permutation_importance", ascending=False))

dropped_features = selection_df.loc[selection_df["drop_candidate"], "feature"].tolist()
feature_cols = [c for c in feature_cols_all if c not in dropped_features]

print("\nFitur yang DIBUANG (lemah di semua metode):", dropped_features)
print("Fitur yang DIPAKAI untuk model final:", feature_cols)

X = df_fds[feature_cols]

# =========================================================
# 5. POIN 3 — REPEATED STRATIFIED K-FOLD (lebih stabil dari single 5-fold)
# =========================================================
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)  # 50 total fits per model

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]),
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=5, class_weight="balanced", random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42))
    ])
}

# =========================================================
# 6. POIN 4 — EVALUASI DENGAN PRECISION & RECALL PER KELAS (bukan cuma AUC)
# =========================================================
scoring = {
    "roc_auc": "roc_auc",
    "precision_suspicious": "precision",   # kelas positif = 1 = Suspicious (urutan alfabetis)
    "recall_suspicious": "recall",
    "f1_suspicious": "f1"
}

print("\n=== Evaluasi Model (Repeated 5-fold x10, fokus ke recall kelas Suspicious) ===")
final_results = {}
for name, pipe in models.items():
    cv_res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    final_results[name] = cv_res
    print(f"\n--- {name} ---")
    print(f"ROC-AUC        : mean={cv_res['test_roc_auc'].mean():.4f}, std={cv_res['test_roc_auc'].std():.4f}")
    print(f"Precision(Susp): mean={cv_res['test_precision_suspicious'].mean():.4f}, std={cv_res['test_precision_suspicious'].std():.4f}")
    print(f"Recall(Susp)   : mean={cv_res['test_recall_suspicious'].mean():.4f}, std={cv_res['test_recall_suspicious'].std():.4f}")
    print(f"F1(Susp)       : mean={cv_res['test_f1_suspicious'].mean():.4f}, std={cv_res['test_f1_suspicious'].std():.4f}")

print("""
CATATAN INTERPRETASI:
- Bandingkan RECALL kelas Suspicious antar model, bukan cuma AUC — di konteks
  AML/fraud, model dengan recall tinggi (jarang miss kasus Suspicious) biasanya
  lebih diprioritaskan meski precision-nya sedikit lebih rendah (lebih baik
  banyak false alarm yang bisa direview manual, daripada kasus suspicious lolos).
- Std yang mengecil dibanding sebelumnya (karena repeated CV) menandakan estimasi
  performa sekarang lebih bisa dipercaya.
- LANGKAH SETELAH INI: kalau dpn_flag terkonfirmasi leakage dari poin 1, ulangi
  seluruh proses tanpa dpn_flag dan bandingkan penurunan performanya — itu akan
  jadi ukuran seberapa besar model 'curang' mengandalkan fitur tersebut.
""")