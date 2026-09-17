"""
Fraud Scoring Pipeline
=======================
Tujuan:
  1. Menghitung korelasi tiap fitur terhadap target (disposition_class) -> arah & kekuatan pengaruh
  2. Membangun model untuk memprediksi probabilitas suatu transaksi/nasabah Suspicious
  3. Mengeluarkan feature importance (linear & non-linear) agar tim fraud tahu faktor paling berpengaruh

Cara pakai:
  - Sesuaikan PATH di bagian CONFIG
  - Jalankan section per section (bisa pakai #%% kalau di VSCode Interactive Window / Jupyter cell)
  - requirements: pandas, numpy, scikit-learn, scipy, xgboost, matplotlib, seaborn
    pip install pandas numpy scikit-learn scipy xgboost matplotlib seaborn
"""

# %% ------------------------------------------------------------------
# 0. IMPORT & CONFIG
# ------------------------------------------------------------------
import os
os.environ["MPLBACKEND"] = "Agg"  # override SEBELUM matplotlib di-import, krn matplotlib
                                    # baca env var ini langsung di __init__.py-nya

# folder output: pakai folder relatif terhadap lokasi script ini, dan auto-dibuat
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)

import xgboost as xgb

pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 150)

# ---- CONFIG ----
PATH = "/home/cdsw/parquet_output/result/trx_all_enriched.parquet"

TARGET_COL = "disposition_class"          # label: Suspicious / Not Suspicious
TIME_COL = "alert_month"                  # dipakai untuk time-based split
ID_COLS = ["key1", "accountid"]           # bukan fitur, jangan masuk model
FLAG_FILTER_COL = "flag"
FLAG_FILTER_VALUE = "FDS"

COLS = [
    "key1", "alert_month", "accountid", "disposition_class", "account_age",
    "flag_account_age_lt9", "flag_anomaly_income",
    "n_sameday_pass_3h_l3m", "n_sameday_pass_3h_l6m",
    "n_sameday_pass_1h_l3m", "n_sameday_pass_1h_l6m",
    "transfer_not_normal_l3m_pct", "transfer_not_normal_l6m_pct",
    "n_crypto", "n_in_10x_income", "flag_in_10x_income",
    "n_out_10x_income", "flag_out_10x_income",
    "n_new_beneficiary", "flag_new_beneficiary",
    "dpn_flag",
    "screening_flag", "flag",
]

# %% ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
df = pd.read_parquet(PATH)[COLS].copy()
df_fds = df[df[FLAG_FILTER_COL] == FLAG_FILTER_VALUE].drop(columns=[FLAG_FILTER_COL]).copy()

print("Total baris FDS:", len(df_fds))
print(df_fds[TARGET_COL].value_counts(dropna=False))
print(df_fds.groupby(TIME_COL)[TARGET_COL].apply(
    lambda s: pd.Series({"count": s.size, "%sus": (s == "Suspicious").mean()})
))

# %% ------------------------------------------------------------------
# 2. EDA SINGKAT: MISSING VALUE & TIPE DATA
# ------------------------------------------------------------------
print(df_fds.dtypes)
print(df_fds.isna().mean().sort_values(ascending=False))

# encode target -> 1 = Suspicious, 0 = Not Suspicious
df_fds["target"] = (df_fds[TARGET_COL].astype(str).str.strip().str.lower() == "suspicious").astype(int)

# tentukan kolom fitur (numerik) -> exclude id, time, target asli
feature_cols = [
    c for c in df_fds.columns
    if c not in ID_COLS + [TIME_COL, TARGET_COL, "target"]
]
print("Jumlah fitur kandidat:", len(feature_cols))
print(feature_cols)

# %% ------------------------------------------------------------------
# 3. KORELASI FITUR -> TARGET (arah & kekuatan)
#    Point-biserial cocok untuk fitur numerik/biner vs target biner
# ------------------------------------------------------------------
corr_results = []
for col in feature_cols:
    s = df_fds[col]
    if not np.issubdtype(s.dtype, np.number):
        # kalau ada kolom object/boolean non-numerik, coba convert
        s = pd.to_numeric(s, errors="coerce")
    mask = s.notna()
    if mask.sum() < 30 or s[mask].nunique() < 2:
        continue
    r, p = pointbiserialr(df_fds.loc[mask, "target"], s[mask])
    corr_results.append({"feature": col, "corr": r, "p_value": p, "n": mask.sum()})

corr_df = pd.DataFrame(corr_results).sort_values("corr", ascending=False)
corr_df["direction"] = np.where(corr_df["corr"] > 0, "positif (naik -> lebih suspicious)",
                                 "negatif (naik -> lebih tidak suspicious)")
corr_df["signifikan_5pct"] = corr_df["p_value"] < 0.05

print(corr_df.to_string(index=False))

# visualisasi cepat
plt.figure(figsize=(8, max(4, len(corr_df) * 0.3)))
sns.barplot(data=corr_df, x="corr", y="feature",
            palette=["#d62728" if v > 0 else "#1f77b4" for v in corr_df["corr"]])
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Korelasi fitur terhadap target Suspicious (point-biserial)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "corr_to_target.png"), dpi=150)
plt.close()

# %% ------------------------------------------------------------------
# 4. CEK MULTIKOLINEARITAS ANTAR FITUR (biar tau fitur redundan)
# ------------------------------------------------------------------
num_df = df_fds[feature_cols].apply(pd.to_numeric, errors="coerce")
corr_matrix = num_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False)
plt.title("Korelasi antar fitur (cek redundansi)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "corr_matrix_features.png"), dpi=150)
plt.close()

# pasangan fitur dengan korelasi tinggi (>|0.85|) -> kandidat drop salah satu
high_corr_pairs = []
cols_m = corr_matrix.columns
for i in range(len(cols_m)):
    for j in range(i + 1, len(cols_m)):
        v = corr_matrix.iloc[i, j]
        if pd.notna(v) and abs(v) > 0.85:
            high_corr_pairs.append((cols_m[i], cols_m[j], v))
print("Pasangan fitur redundant (|corr| > 0.85):")
for a, b, v in high_corr_pairs:
    print(f"  {a} <-> {b} : {v:.2f}")

# %% ------------------------------------------------------------------
# 5. TIME-BASED TRAIN/TEST SPLIT
#    Jangan random split karena data ini time series per alert_month
#    contoh: train = Jan-Apr 2026, test = Mei-Jun 2026
# ------------------------------------------------------------------
df_fds[TIME_COL] = df_fds[TIME_COL].astype(str)
print("Contoh nilai unik alert_month:", sorted(df_fds[TIME_COL].unique())[:15])
train_months = ["2026-01", "2026-02", "2026-03", "2026-04"]
test_months = ["2026-05", "2026-06"]

train_df = df_fds[df_fds[TIME_COL].str.startswith(tuple(train_months))].copy()
test_df = df_fds[df_fds[TIME_COL].str.startswith(tuple(test_months))].copy()

if len(train_df) == 0 or len(test_df) == 0:
    raise ValueError(
        f"Train/test kosong! Cek format alert_month di atas (Contoh nilai unik), "
        f"lalu sesuaikan train_months/test_months. "
        f"len(train_df)={len(train_df)}, len(test_df)={len(test_df)}"
    )

X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
y_train = train_df["target"]
X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")
y_test = test_df["target"]

print("Train:", X_train.shape, "Positif rate:", y_train.mean())
print("Test :", X_test.shape, "Positif rate:", y_test.mean())

# imputasi NaN sederhana (median dari train, supaya konsisten & tidak leak)
medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

# %% ------------------------------------------------------------------
# 6a. MODEL 1: LOGISTIC REGRESSION (interpretable, buat cek arah pengaruh)
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
logreg.fit(X_train_scaled, y_train)

proba_lr = logreg.predict_proba(X_test_scaled)[:, 1]
print("LogReg ROC-AUC :", roc_auc_score(y_test, proba_lr))
print("LogReg PR-AUC  :", average_precision_score(y_test, proba_lr))

coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coef": logreg.coef_[0]
}).sort_values("coef", ascending=False)
coef_df["direction"] = np.where(coef_df["coef"] > 0, "menaikkan risiko", "menurunkan risiko")
print(coef_df.to_string(index=False))

# %% ------------------------------------------------------------------
# 6b. MODEL 2: RANDOM FOREST (non-linear, buat feature importance & performa)
# ------------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=500, max_depth=6, min_samples_leaf=20,
    class_weight="balanced_subsample", random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
proba_rf = rf.predict_proba(X_test)[:, 1]
print("RF ROC-AUC :", roc_auc_score(y_test, proba_rf))
print("RF PR-AUC  :", average_precision_score(y_test, proba_rf))

rf_imp = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)
print(rf_imp.to_string(index=False))

# %% ------------------------------------------------------------------
# 6c. MODEL 3: XGBOOST (biasanya performa terbaik untuk tabular data begini)
# ------------------------------------------------------------------
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

xgb_model = xgb.XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc", random_state=42
)
xgb_model.fit(X_train, y_train)
proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
print("XGB ROC-AUC :", roc_auc_score(y_test, proba_xgb))
print("XGB PR-AUC  :", average_precision_score(y_test, proba_xgb))

xgb_imp = pd.DataFrame({
    "feature": feature_cols,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=False)
print(xgb_imp.to_string(index=False))

# %% ------------------------------------------------------------------
# 7. EVALUASI LEBIH DETAIL (pilih model terbaik, misal XGB)
# ------------------------------------------------------------------
best_proba = proba_xgb
threshold = 0.5
pred_label = (best_proba >= threshold).astype(int)

print(classification_report(y_test, pred_label, target_names=["Not Suspicious", "Suspicious"]))
print(confusion_matrix(y_test, pred_label))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
RocCurveDisplay.from_predictions(y_test, best_proba, ax=ax[0])
PrecisionRecallDisplay.from_predictions(y_test, best_proba, ax=ax[1])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_eval_curves.png"), dpi=150)
plt.close()

# %% ------------------------------------------------------------------
# 8. SCORING OUTPUT PER NASABAH/AKUN
# ------------------------------------------------------------------
scored = test_df[["key1", "accountid", TIME_COL, TARGET_COL]].copy()
scored["score_suspicious"] = best_proba
scored = scored.sort_values("score_suspicious", ascending=False)
print(scored.head(20))

scored.to_csv(os.path.join(OUTPUT_DIR, "scored_output.csv"), index=False)

print(f"Selesai. Output tersimpan di folder: {OUTPUT_DIR}")
