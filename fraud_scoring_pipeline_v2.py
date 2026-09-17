"""
Fraud Scoring Pipeline v2
==========================
Perbaikan dari v1, sesuai poin-poin review:
  1. Cek stabilitas performa per bulan (concept drift)
  2. Feature engineering: interaksi & rasio antar fitur ber-sinyal
  3. Auto-drop fitur redundant (|corr| > 0.85), keep yang korelasinya ke target paling kuat
  4. Hyperparameter tuning time-series-aware (TimeSeriesSplit + RandomizedSearchCV)
  5. Evaluasi precision@k (buat use-case ranking/prioritas review, bukan cuma threshold 0.5)
  6. Data historis -> tinggal ganti PATH/rentang bulan kalau sudah ada data lebih panjang
  7. Rolling-origin cross-validation (expanding window per bulan)

requirements:
  pip install pandas numpy scikit-learn scipy xgboost matplotlib seaborn
"""

# %% ------------------------------------------------------------------
# 0. IMPORT & CONFIG
# ------------------------------------------------------------------
import os
os.environ["MPLBACKEND"] = "Agg"  # override SEBELUM matplotlib di-import

import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
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

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATH = "/home/cdsw/parquet_output/result/trx_all_enriched.parquet"
TARGET_COL = "disposition_class"
TIME_COL = "alert_month"
ID_COLS = ["key1", "accountid"]
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

RANDOM_STATE = 42

# %% ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
df = pd.read_parquet(PATH)[COLS].copy()
df_fds = df[df[FLAG_FILTER_COL] == FLAG_FILTER_VALUE].drop(columns=[FLAG_FILTER_COL]).copy()
df_fds[TIME_COL] = df_fds[TIME_COL].astype(str)

print("Total baris FDS:", len(df_fds))
print("Rentang alert_month:", df_fds[TIME_COL].min(), "-", df_fds[TIME_COL].max())

df_fds["target"] = (df_fds[TARGET_COL].astype(str).str.strip().str.lower() == "suspicious").astype(int)

feature_cols = [
    c for c in df_fds.columns
    if c not in ID_COLS + [TIME_COL, TARGET_COL, "target"]
]

# %% ------------------------------------------------------------------
# 2. KORELASI FITUR -> TARGET (arah & kekuatan)
# ------------------------------------------------------------------
corr_results = []
for col in feature_cols:
    s = pd.to_numeric(df_fds[col], errors="coerce")
    mask = s.notna()
    if mask.sum() < 30 or s[mask].nunique() < 2:
        continue
    r, p = pointbiserialr(df_fds.loc[mask, "target"], s[mask])
    corr_results.append({"feature": col, "corr": r, "abs_corr": abs(r), "p_value": p, "n": mask.sum()})

corr_df = pd.DataFrame(corr_results).sort_values("corr", ascending=False)
print(corr_df.drop(columns="abs_corr").to_string(index=False))

# %% ------------------------------------------------------------------
# 3. AUTO-DROP FITUR REDUNDANT (poin 3)
#    Untuk tiap pasangan |corr| > 0.85, buang yang korelasinya ke target LEBIH LEMAH
# ------------------------------------------------------------------
num_df = df_fds[feature_cols].apply(pd.to_numeric, errors="coerce")
corr_matrix = num_df.corr()

corr_to_target = corr_df.set_index("feature")["abs_corr"].to_dict()

to_drop = set()
cols_m = corr_matrix.columns
for i in range(len(cols_m)):
    for j in range(i + 1, len(cols_m)):
        a, b = cols_m[i], cols_m[j]
        v = corr_matrix.iloc[i, j]
        if pd.notna(v) and abs(v) > 0.85:
            score_a = corr_to_target.get(a, 0)
            score_b = corr_to_target.get(b, 0)
            weaker = b if score_a >= score_b else a
            to_drop.add(weaker)

print("Fitur yang di-drop karena redundant (|corr| > 0.85, sinyal ke target lebih lemah):")
print(sorted(to_drop))

feature_cols_clean = [c for c in feature_cols if c not in to_drop]
print(f"Fitur tersisa: {len(feature_cols_clean)} dari {len(feature_cols)}")
print(feature_cols_clean)

# %% ------------------------------------------------------------------
# 4. FEATURE ENGINEERING: interaksi & rasio (poin 2)
#    Kombinasi fitur dengan sinyal individual lemah, bisa lebih kuat kalau digabung
# ------------------------------------------------------------------
def add_engineered_features(data: pd.DataFrame, cols_available: list) -> pd.DataFrame:
    d = data.copy()
    # interaksi: sinyal income anomaly + crypto (top 2 fitur dari korelasi)
    if "flag_anomaly_income" in cols_available and "n_crypto" in cols_available:
        d["fe_anomaly_income_x_crypto"] = d["flag_anomaly_income"] * d["n_crypto"]
    # interaksi: transfer 10x income (in & out) walau salah satu mungkin sudah di-drop krn redundant,
    # coba pakai yang tersisa dikali account_age (akun baru + transfer besar = lebih berisiko)
    if "n_out_10x_income" in cols_available and "account_age" in cols_available:
        d["fe_out10x_per_account_age"] = d["n_out_10x_income"] / (d["account_age"].fillna(0) + 1)
    if "n_in_10x_income" in cols_available and "account_age" in cols_available:
        d["fe_in10x_per_account_age"] = d["n_in_10x_income"] / (d["account_age"].fillna(0) + 1)
    # rasio same-day pass terhadap account age (aktivitas cepat di akun baru = lebih mencurigakan)
    sameday_cols = [c for c in cols_available if c.startswith("n_sameday_pass")]
    if sameday_cols and "account_age" in cols_available:
        d["fe_sameday_total"] = d[sameday_cols].sum(axis=1)
        d["fe_sameday_per_account_age"] = d["fe_sameday_total"] / (d["account_age"].fillna(0) + 1)
    # flag gabungan: berapa banyak red-flag yang menyala bersamaan
    flag_cols = [c for c in cols_available if c.startswith("flag_")]
    if flag_cols:
        d["fe_total_flags_on"] = d[flag_cols].sum(axis=1)
    return d

df_fds_fe = add_engineered_features(df_fds, feature_cols_clean)
new_fe_cols = [c for c in df_fds_fe.columns if c.startswith("fe_")]
feature_cols_final = feature_cols_clean + new_fe_cols
print("Fitur baru hasil engineering:", new_fe_cols)

# %% ------------------------------------------------------------------
# 5. TIME-BASED TRAIN/TEST SPLIT
# ------------------------------------------------------------------
all_months_sorted = sorted(df_fds_fe[TIME_COL].str[:7].unique())  # ambil YYYY-MM unik
print("Bulan yang tersedia:", all_months_sorted)

n_test_months = 2
train_months = all_months_sorted[:-n_test_months]
test_months = all_months_sorted[-n_test_months:]
print("Train months:", train_months, " | Test months:", test_months)

train_df = df_fds_fe[df_fds_fe[TIME_COL].str.startswith(tuple(train_months))].copy()
test_df = df_fds_fe[df_fds_fe[TIME_COL].str.startswith(tuple(test_months))].copy()

if len(train_df) == 0 or len(test_df) == 0:
    raise ValueError(f"Train/test kosong! len(train)={len(train_df)}, len(test)={len(test_df)}")

X_train = train_df[feature_cols_final].apply(pd.to_numeric, errors="coerce")
y_train = train_df["target"]
X_test = test_df[feature_cols_final].apply(pd.to_numeric, errors="coerce")
y_test = test_df["target"]

medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

print("Train:", X_train.shape, "Positif rate:", round(y_train.mean(), 3))
print("Test :", X_test.shape, "Positif rate:", round(y_test.mean(), 3))

# %% ------------------------------------------------------------------
# 6. HYPERPARAMETER TUNING TIME-SERIES-AWARE (poin 4)
#    TimeSeriesSplit -> fold selalu train di masa lalu, validasi di masa depan (bukan random)
# ------------------------------------------------------------------
tscv = TimeSeriesSplit(n_splits=3)

rf_param_dist = {
    "n_estimators": [200, 300, 500, 800],
    "max_depth": [3, 4, 5, 6, 8, None],
    "min_samples_leaf": [5, 10, 20, 30],
    "max_features": ["sqrt", "log2", 0.5, 0.8],
}
rf_search = RandomizedSearchCV(
    RandomForestClassifier(class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1),
    param_distributions=rf_param_dist, n_iter=25, scoring="roc_auc",
    cv=tscv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
)
rf_search.fit(X_train, y_train)
print("Best RF params:", rf_search.best_params_)
print("Best RF CV ROC-AUC:", rf_search.best_score_)
rf_best = rf_search.best_estimator_

scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
xgb_param_dist = {
    "n_estimators": [200, 300, 400, 600],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 5, 10],
}
xgb_search = RandomizedSearchCV(
    xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="auc", random_state=RANDOM_STATE),
    param_distributions=xgb_param_dist, n_iter=25, scoring="roc_auc",
    cv=tscv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
)
xgb_search.fit(X_train, y_train)
print("Best XGB params:", xgb_search.best_params_)
print("Best XGB CV ROC-AUC:", xgb_search.best_score_)
xgb_best = xgb_search.best_estimator_

# %% ------------------------------------------------------------------
# 7. LOGISTIC REGRESSION (baseline interpretable, tetap dipertahankan)
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=3000, class_weight="balanced")
logreg.fit(X_train_scaled, y_train)
proba_lr = logreg.predict_proba(X_test_scaled)[:, 1]

coef_df = pd.DataFrame({"feature": feature_cols_final, "coef": logreg.coef_[0]}).sort_values("coef", ascending=False)
coef_df["direction"] = np.where(coef_df["coef"] > 0, "menaikkan risiko", "menurunkan risiko")

# %% ------------------------------------------------------------------
# 8. EVALUASI DI TEST SET (model tuned)
# ------------------------------------------------------------------
proba_rf = rf_best.predict_proba(X_test)[:, 1]
proba_xgb = xgb_best.predict_proba(X_test)[:, 1]

results_summary = pd.DataFrame({
    "model": ["LogReg", "RandomForest (tuned)", "XGBoost (tuned)"],
    "roc_auc": [roc_auc_score(y_test, proba_lr), roc_auc_score(y_test, proba_rf), roc_auc_score(y_test, proba_xgb)],
    "pr_auc": [average_precision_score(y_test, proba_lr), average_precision_score(y_test, proba_rf), average_precision_score(y_test, proba_xgb)],
})
print(results_summary.to_string(index=False))

best_proba = proba_xgb  # ganti manual kalau model lain ternyata lebih baik di results_summary
best_model_name = "XGBoost (tuned)"

pred_label = (best_proba >= 0.5).astype(int)
print(f"\n=== Classification report ({best_model_name}, threshold 0.5) ===")
print(classification_report(y_test, pred_label, target_names=["Not Suspicious", "Suspicious"]))
print(confusion_matrix(y_test, pred_label))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
RocCurveDisplay.from_predictions(y_test, best_proba, ax=ax[0])
PrecisionRecallDisplay.from_predictions(y_test, best_proba, ax=ax[1])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_eval_curves_v2.png"), dpi=150)
plt.close()

# %% ------------------------------------------------------------------
# 9. PRECISION@K (poin 5) -- lebih relevan buat use-case prioritas review
#    "Kalau tim fraud cuma sanggup review top-K% alert dengan score tertinggi,
#     berapa persen dari situ yang benar Suspicious?"
# ------------------------------------------------------------------
def precision_at_k(y_true, y_score, k_pct):
    n = len(y_true)
    k = max(1, int(np.ceil(n * k_pct)))
    order = np.argsort(-y_score)
    top_k_idx = order[:k]
    y_true_arr = np.asarray(y_true)
    return y_true_arr[top_k_idx].mean(), k

print("\n=== Precision@K (ranking top alert by score) ===")
prec_k_rows = []
for k_pct in [0.1, 0.2, 0.3, 0.5]:
    prec, k = precision_at_k(y_test, best_proba, k_pct)
    prec_k_rows.append({"top_pct": k_pct, "n_reviewed": k, "precision": round(prec, 3)})
prec_k_df = pd.DataFrame(prec_k_rows)
print(prec_k_df.to_string(index=False))
print(f"(baseline random pick / base rate suspicious di test = {round(y_test.mean(), 3)})")

# %% ------------------------------------------------------------------
# 10. STABILITAS PER BULAN (poin 1) -- cek concept drift
# ------------------------------------------------------------------
test_df_eval = test_df.copy()
test_df_eval["proba"] = best_proba
test_df_eval["month_only"] = test_df_eval[TIME_COL].str[:7]

monthly_auc = []
for m, grp in test_df_eval.groupby("month_only"):
    if grp["target"].nunique() < 2:
        continue
    auc_m = roc_auc_score(grp["target"], grp["proba"])
    monthly_auc.append({"month": m, "n": len(grp), "positif_rate": grp["target"].mean(), "roc_auc": auc_m})
monthly_auc_df = pd.DataFrame(monthly_auc)
print("\n=== ROC-AUC per bulan (test set) ===")
print(monthly_auc_df.to_string(index=False))

# %% ------------------------------------------------------------------
# 11. ROLLING-ORIGIN CROSS-VALIDATION (poin 7)
#    Expanding window: train bulan 1..k, test bulan k+1, geser terus
# ------------------------------------------------------------------
print("\n=== Rolling-origin CV (expanding window per bulan) ===")
rolling_results = []
for i in range(2, len(all_months_sorted)):  # minimal 2 bulan buat training awal
    train_m = all_months_sorted[:i]
    test_m = all_months_sorted[i]

    tr = df_fds_fe[df_fds_fe[TIME_COL].str.startswith(tuple(train_m))]
    te = df_fds_fe[df_fds_fe[TIME_COL].str.startswith(test_m)]

    if te["target"].nunique() < 2 or len(tr) < 30:
        continue

    Xtr = tr[feature_cols_final].apply(pd.to_numeric, errors="coerce").fillna(tr[feature_cols_final].median(numeric_only=True))
    ytr = tr["target"]
    Xte = te[feature_cols_final].apply(pd.to_numeric, errors="coerce").fillna(tr[feature_cols_final].median(numeric_only=True))
    yte = te["target"]

    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    m = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=spw, eval_metric="auc", random_state=RANDOM_STATE
    )
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    rolling_results.append({"train_until": train_m[-1], "test_month": test_m, "n_train": len(tr), "n_test": len(te), "roc_auc": auc})

rolling_df = pd.DataFrame(rolling_results)
print(rolling_df.to_string(index=False))
print(f"Rata-rata ROC-AUC rolling: {rolling_df['roc_auc'].mean():.3f} (std: {rolling_df['roc_auc'].std():.3f})")

# %% ------------------------------------------------------------------
# 12. FEATURE IMPORTANCE GABUNGAN & SIMPAN
# ------------------------------------------------------------------
rf_imp = pd.DataFrame({"feature": feature_cols_final, "importance_rf": rf_best.feature_importances_})
xgb_imp = pd.DataFrame({"feature": feature_cols_final, "importance_xgb": xgb_best.feature_importances_})

importance_combined = (
    coef_df[["feature", "coef", "direction"]]
    .merge(rf_imp, on="feature", how="outer")
    .merge(xgb_imp, on="feature", how="outer")
    .sort_values("importance_xgb", ascending=False)
)
print("\n=== Feature importance gabungan ===")
print(importance_combined.to_string(index=False))
importance_combined.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_v2.csv"), index=False)

# %% ------------------------------------------------------------------
# 13. SCORING OUTPUT PER NASABAH/AKUN
# ------------------------------------------------------------------
scored = test_df[["key1", "accountid", TIME_COL, TARGET_COL]].copy()
scored["score_suspicious"] = best_proba
scored["rank_pct"] = scored["score_suspicious"].rank(pct=True, ascending=False)
scored = scored.sort_values("score_suspicious", ascending=False)
scored.to_csv(os.path.join(OUTPUT_DIR, "scored_output_v2.csv"), index=False)

print(f"\nSelesai. Output tersimpan di folder: {OUTPUT_DIR}")
print("File: model_eval_curves_v2.png, feature_importance_v2.csv, scored_output_v2.csv")
