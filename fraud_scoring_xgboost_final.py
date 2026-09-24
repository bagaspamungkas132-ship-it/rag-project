"""
Fraud Scoring - Finalisasi Model XGBoost (buat presentasi ke atasan)
=======================================================================
Beda dari versi LogReg:
  - Data di-clean OTOMATIS: fitur redundant (|corr|>0.85) langsung dibuang, sisa 1
    representatif per cluster (yang korelasinya ke target paling kuat)
  - "Signifikansi" fitur dicek pakai PERMUTATION IMPORTANCE, bukan p-value -- karena
    XGBoost/tree model tidak punya p-value seperti regresi linear. Cara kerja: acak
    1 fitur, lihat performa turun berapa banyak. Turun besar = fitur penting beneran.
  - Ada MANUAL_ALWAYS_KEEP -- fitur yang secara bisnis red-flag klasik (kripto, dst)
    tetap dipertahankan meski sinyal statistiknya lemah -- ini keputusan domain,
    bukan keputusan algoritma, jadi ditulis eksplisit di CONFIG biar bisa didiskusikan.

Output:
  1. scored_all_key1_xgb.csv       -> score tiap key1 pakai model final
  2. threshold_analysis_xgb.csv    -> tabel precision/recall per threshold
  3. feature_cleaning_report.csv   -> fitur mana dibuang/dipertahankan dan alasannya
  4. feature_importance_xgb.csv    -> importance bawaan + permutation importance
  5. model_eval_curves_xgb.png     -> ROC & PR curve

requirements:
  pip install pandas numpy scikit-learn scipy xgboost matplotlib seaborn
"""

# %% ------------------------------------------------------------------
# 0. IMPORT & CONFIG
# ------------------------------------------------------------------
import os
os.environ["MPLBACKEND"] = "Agg"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score, f1_score,
    RocCurveDisplay, PrecisionRecallDisplay
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
REDUNDANT_CORR_THRESHOLD = 0.85
PERM_IMPORTANCE_DROP_THRESHOLD = 0.0  # fitur dgn permutation importance <= ini dianggap tidak kontributif

# Fitur yang TETAP dipertahankan meski sinyal statistiknya lemah, karena secara
# bisnis fraud/AML ini red-flag klasik yang tetap relevan walau datanya belum
# cukup banyak buat "membuktikan" secara statistik. Ini keputusan domain -- CEK
# ULANG list ini sama tim fraud, jangan cuma percaya keputusan aku.
MANUAL_ALWAYS_KEEP = ["n_crypto", "n_new_beneficiary", "screening_flag"]

# Threshold final -- isi setelah lihat threshold_analysis_xgb.csv & diskusi kapasitas review
FINAL_THRESHOLD = 0.5

# %% ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
df = pd.read_parquet(PATH)[COLS].copy()
df_fds = df[df[FLAG_FILTER_COL] == FLAG_FILTER_VALUE].drop(columns=[FLAG_FILTER_COL]).copy()
df_fds[TIME_COL] = df_fds[TIME_COL].astype(str)
df_fds["target"] = (df_fds[TARGET_COL].astype(str).str.strip().str.lower() == "suspicious").astype(int)

feature_cols_raw = [c for c in df_fds.columns if c not in ID_COLS + [TIME_COL, TARGET_COL, "target"]]
print(f"Total baris FDS: {len(df_fds)} | Fitur mentah: {len(feature_cols_raw)}")

# %% ------------------------------------------------------------------
# 2. AUTO-CLEAN FITUR REDUNDANT (|corr| > 0.85)
#    Kalau ada cluster (bukan cuma pasangan) yang saling redundant, kelompokkan dulu
#    (union-find sederhana), lalu simpan 1 representatif per cluster: yang korelasinya
#    ke target PALING KUAT.
# ------------------------------------------------------------------
from scipy.stats import pointbiserialr

corr_results = []
for col in feature_cols_raw:
    s = pd.to_numeric(df_fds[col], errors="coerce")
    mask = s.notna()
    if mask.sum() < 30 or s[mask].nunique() < 2:
        continue
    r, p = pointbiserialr(df_fds.loc[mask, "target"], s[mask])
    corr_results.append({"feature": col, "corr_to_target": r, "abs_corr_to_target": abs(r)})
corr_df = pd.DataFrame(corr_results)
corr_to_target = corr_df.set_index("feature")["abs_corr_to_target"].to_dict()

num_df = df_fds[feature_cols_raw].apply(pd.to_numeric, errors="coerce")
corr_matrix = num_df.corr()

# union-find sederhana buat kelompokkan cluster fitur yang saling redundant
parent = {c: c for c in feature_cols_raw}
def find(x):
    while parent[x] != x:
        x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb

redundant_pairs_log = []
cols_m = corr_matrix.columns
for i in range(len(cols_m)):
    for j in range(i + 1, len(cols_m)):
        a, b = cols_m[i], cols_m[j]
        v = corr_matrix.iloc[i, j]
        if pd.notna(v) and abs(v) > REDUNDANT_CORR_THRESHOLD:
            union(a, b)
            redundant_pairs_log.append({"feature_a": a, "feature_b": b, "corr_ab": round(v, 3)})

clusters = {}
for c in feature_cols_raw:
    root = find(c)
    clusters.setdefault(root, []).append(c)

cleaning_report = []
feature_cols_clean = []
for root, members in clusters.items():
    if len(members) == 1:
        feature_cols_clean.append(members[0])
        cleaning_report.append({"feature": members[0], "keputusan": "DIPERTAHANKAN",
                                 "alasan": "tidak redundant dengan fitur lain"})
    else:
        # pilih representatif: korelasi ke target paling kuat
        best = max(members, key=lambda f: corr_to_target.get(f, 0))
        feature_cols_clean.append(best)
        cleaning_report.append({"feature": best, "keputusan": "DIPERTAHANKAN",
                                 "alasan": f"representatif cluster redundant {members} (corr ke target paling kuat)"})
        for m in members:
            if m != best:
                cleaning_report.append({"feature": m, "keputusan": "DIBUANG",
                                         "alasan": f"redundant dengan '{best}' (satu cluster: {members})"})

print(f"\n=== Cluster fitur redundant (|corr| > {REDUNDANT_CORR_THRESHOLD}) ===")
for root, members in clusters.items():
    if len(members) > 1:
        best = max(members, key=lambda f: corr_to_target.get(f, 0))
        print(f"  Cluster: {members} -> keep: '{best}'")

print(f"\nFitur setelah cleaning redundansi: {len(feature_cols_clean)} dari {len(feature_cols_raw)}")
print(sorted(feature_cols_clean))

# --- Flag khusus: kasus in/out 10x_income (redundant scr statistik TAPI beda arah bisnis) ---
if "n_in_10x_income" in [r["feature"] for r in cleaning_report if r["keputusan"] == "DIBUANG"] or \
   "n_out_10x_income" in [r["feature"] for r in cleaning_report if r["keputusan"] == "DIBUANG"]:
    print("\n>> PERHATIAN: n_in_10x_income & n_out_10x_income kedeteksi redundant secara statistik "
          "(korelasi tinggi), tapi 'uang masuk' vs 'uang keluar' itu peristiwa ekonomi yang beda "
          "secara bisnis. Salah satunya otomatis dibuang oleh aturan cluster -- CEK LAGI keputusan "
          "ini sama tim fraud sebelum final, jangan cuma percaya statistik di sini.")

# %% ------------------------------------------------------------------
# 3. TIME-BASED SPLIT
# ------------------------------------------------------------------
all_months_sorted = sorted(df_fds[TIME_COL].str[:7].unique())
n_test_months = 2
train_months = all_months_sorted[:-n_test_months]
test_months = all_months_sorted[-n_test_months:]

train_df = df_fds[df_fds[TIME_COL].str.startswith(tuple(train_months))].copy()
test_df = df_fds[df_fds[TIME_COL].str.startswith(tuple(test_months))].copy()

X_train = train_df[feature_cols_clean].apply(pd.to_numeric, errors="coerce")
y_train = train_df["target"]
X_test = test_df[feature_cols_clean].apply(pd.to_numeric, errors="coerce")
y_test = test_df["target"]

medians_train = X_train.median()
X_train = X_train.fillna(medians_train)
X_test = X_test.fillna(medians_train)

print(f"\nTrain: {X_train.shape} (bulan {train_months}) | Test: {X_test.shape} (bulan {test_months})")

# %% ------------------------------------------------------------------
# 4. TRAIN XGBOOST (tuning konservatif -- lihat catatan overfitting di run LogReg sebelumnya)
# ------------------------------------------------------------------
tscv = TimeSeriesSplit(n_splits=3)
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

xgb_param_dist = {
    "n_estimators": [200, 300, 400],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.02, 0.05, 0.08],
    "subsample": [0.6, 0.8],
    "colsample_bytree": [0.6, 0.8],
    "min_child_weight": [5, 10, 15],
    "reg_lambda": [1, 3, 5, 10],
}
xgb_search = RandomizedSearchCV(
    xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="auc", random_state=RANDOM_STATE),
    param_distributions=xgb_param_dist, n_iter=10, scoring="roc_auc",
    cv=tscv, random_state=RANDOM_STATE, n_jobs=-1,
)
xgb_search.fit(X_train, y_train)
xgb_eval = xgb_search.best_estimator_
proba_test = xgb_eval.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, proba_test)
test_pr_auc = average_precision_score(y_test, proba_test)
cv_test_gap = xgb_search.best_score_ - test_auc

print(f"\n=== Performa model (validasi time-based split) ===")
print(f"Best params: {xgb_search.best_params_}")
print(f"CV ROC-AUC  : {xgb_search.best_score_:.3f}")
print(f"Test ROC-AUC: {test_auc:.3f}")
print(f"Test PR-AUC : {test_pr_auc:.3f}")
print(f"Gap CV-test : {cv_test_gap:.3f}", "(>0.05 = indikasi overfit ke CV)" if cv_test_gap > 0.05 else "(sehat)")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
RocCurveDisplay.from_predictions(y_test, proba_test, ax=ax[0])
PrecisionRecallDisplay.from_predictions(y_test, proba_test, ax=ax[1])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_eval_curves_xgb.png"), dpi=150)
plt.close()

# %% ------------------------------------------------------------------
# 5. PERMUTATION IMPORTANCE -- "signifikansi" versi tree model
#    Dihitung di TEST SET (bukan train), biar mengukur kontribusi fitur ke
#    generalisasi asli, bukan ke data yang sudah dihafal model.
# ------------------------------------------------------------------
perm_result = permutation_importance(
    xgb_eval, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=RANDOM_STATE, n_jobs=-1
)
perm_df = pd.DataFrame({
    "feature": feature_cols_clean,
    "perm_importance_mean": perm_result.importances_mean,
    "perm_importance_std": perm_result.importances_std,
    "builtin_importance": xgb_eval.feature_importances_,
}).sort_values("perm_importance_mean", ascending=False)

print("\n=== Permutation importance (di test set) ===")
print(perm_df.to_string(index=False))

# gabungkan hasil dengan laporan cleaning fitur
cleaning_df = pd.DataFrame(cleaning_report)
final_decision_rows = []
not_significant_drop = []
for _, row in perm_df.iterrows():
    feat = row["feature"]
    if row["perm_importance_mean"] <= PERM_IMPORTANCE_DROP_THRESHOLD and feat not in MANUAL_ALWAYS_KEEP:
        final_decision_rows.append({"feature": feat, "keputusan": "DIBUANG",
                                     "alasan": f"permutation importance <= {PERM_IMPORTANCE_DROP_THRESHOLD} "
                                               f"({row['perm_importance_mean']:.4f}), model tidak terganggu kalau fitur ini diacak"})
        not_significant_drop.append(feat)
    elif row["perm_importance_mean"] <= PERM_IMPORTANCE_DROP_THRESHOLD and feat in MANUAL_ALWAYS_KEEP:
        final_decision_rows.append({"feature": feat, "keputusan": "DIPERTAHANKAN (override manual)",
                                     "alasan": f"sinyal statistik lemah (perm_imp={row['perm_importance_mean']:.4f}) "
                                               f"tapi red-flag bisnis, masuk MANUAL_ALWAYS_KEEP"})
    else:
        final_decision_rows.append({"feature": feat, "keputusan": "DIPERTAHANKAN",
                                     "alasan": f"permutation importance positif ({row['perm_importance_mean']:.4f})"})

final_decision_df = pd.DataFrame(final_decision_rows)
full_report = pd.concat([cleaning_df, final_decision_df], ignore_index=True)
full_report.to_csv(os.path.join(OUTPUT_DIR, "feature_cleaning_report.csv"), index=False)
perm_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_xgb.csv"), index=False)

feature_cols_final = [f for f in feature_cols_clean if f not in not_significant_drop]
print(f"\nFitur final dipakai model: {len(feature_cols_final)} dari {len(feature_cols_raw)} mentah")
print(f"  -> {len(clusters) - len(feature_cols_clean)} dibuang krn redundant")
print(f"  -> {len(not_significant_drop)} dibuang krn permutation importance tidak positif: {not_significant_drop}")
print(feature_cols_final)

# %% ------------------------------------------------------------------
# 6. RE-TRAIN DENGAN FITUR FINAL (buat konfirmasi performa tidak turun setelah cleaning)
# ------------------------------------------------------------------
X_train_final = X_train[feature_cols_final]
X_test_final = X_test[feature_cols_final]

xgb_search_final = RandomizedSearchCV(
    xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="auc", random_state=RANDOM_STATE),
    param_distributions=xgb_param_dist, n_iter=10, scoring="roc_auc",
    cv=tscv, random_state=RANDOM_STATE, n_jobs=-1,
)
xgb_search_final.fit(X_train_final, y_train)
xgb_final_eval = xgb_search_final.best_estimator_
proba_test_final = xgb_final_eval.predict_proba(X_test_final)[:, 1]
test_auc_final = roc_auc_score(y_test, proba_test_final)

print(f"\n=== Bandingkan performa: sebelum vs sesudah buang fitur tidak signifikan ===")
print(f"  Sebelum ({len(feature_cols_clean)} fitur) : test ROC-AUC = {test_auc:.3f}")
print(f"  Sesudah ({len(feature_cols_final)} fitur) : test ROC-AUC = {test_auc_final:.3f}")
if test_auc_final >= test_auc - 0.01:
    print("  >> AMAN -- performa tidak turun berarti setelah fitur dipangkas. "
          "Model lebih ramping, lebih gampang dijelaskan, tanpa mengorbankan akurasi.")
else:
    print("  >> PERHATIAN -- performa turun cukup berarti setelah dipangkas. "
          "Pertimbangkan kembalikan sebagian fitur yang dibuang, terutama yang "
          "permutation importance-nya deket 0 (bukan jelas negatif).")

# %% ------------------------------------------------------------------
# 7. ANALISIS THRESHOLD
# ------------------------------------------------------------------
print("\n=== Analisis Threshold (di test set, model final) ===")
threshold_rows = []
for t in np.arange(0.1, 0.95, 0.05):
    pred = (proba_test_final >= t).astype(int)
    n_flagged = pred.sum()
    prec = precision_score(y_test, pred, zero_division=0) if n_flagged > 0 else None
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    threshold_rows.append({
        "threshold": round(t, 2), "n_alert_ke_flag": n_flagged,
        "pct_dari_total_alert": round(n_flagged / len(y_test), 3),
        "precision": round(prec, 3) if prec is not None else None,
        "recall": round(rec, 3), "f1_score": round(f1, 3),
    })
threshold_df = pd.DataFrame(threshold_rows)
print(threshold_df.to_string(index=False))
threshold_df.to_csv(os.path.join(OUTPUT_DIR, "threshold_analysis_xgb.csv"), index=False)
print(f"\n>> FINAL_THRESHOLD saat ini: {FINAL_THRESHOLD} -- ubah di CONFIG sesuai kapasitas review tim")

# %% ------------------------------------------------------------------
# 8. MODEL FINAL -- retrain pakai SEMUA data historis + fitur final, score semua key1
# ------------------------------------------------------------------
X_all = df_fds[feature_cols_final].apply(pd.to_numeric, errors="coerce")
y_all = df_fds["target"]
X_all = X_all.fillna(X_all.median())

scale_pos_weight_all = (y_all == 0).sum() / max((y_all == 1).sum(), 1)
final_model = xgb.XGBClassifier(
    **{k: v for k, v in xgb_search_final.best_params_.items()},
    scale_pos_weight=scale_pos_weight_all, eval_metric="auc", random_state=RANDOM_STATE
)
final_model.fit(X_all, y_all)
score_all = final_model.predict_proba(X_all)[:, 1]

scored_all = df_fds[["key1", "accountid", TIME_COL, TARGET_COL]].copy()
scored_all["score_suspicious"] = score_all
scored_all["rank_pct"] = scored_all["score_suspicious"].rank(pct=True, ascending=False)
scored_all["flag_recommended"] = np.where(scored_all["score_suspicious"] >= FINAL_THRESHOLD, "Suspicious", "Not Suspicious")
scored_all = scored_all.sort_values("score_suspicious", ascending=False)
scored_all.to_csv(os.path.join(OUTPUT_DIR, "scored_all_key1_xgb.csv"), index=False)

print(f"\nTotal key1 di-score: {len(scored_all)}")
print(f"Di-flag Suspicious pada threshold {FINAL_THRESHOLD}: "
      f"{(scored_all['flag_recommended'] == 'Suspicious').sum()} "
      f"({(scored_all['flag_recommended'] == 'Suspicious').mean():.1%})")
print(scored_all.head(10).to_string(index=False))

# %% ------------------------------------------------------------------
# RINGKASAN AKHIR
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("RINGKASAN UNTUK PRESENTASI")
print("=" * 70)
print(f"Model                : XGBoost ({xgb_search_final.best_params_})")
print(f"Data training final   : {len(df_fds)} baris ({all_months_sorted[0]} s/d {all_months_sorted[-1]})")
print(f"Validasi (holdout)    : ROC-AUC={test_auc_final:.3f}, PR-AUC={average_precision_score(y_test, proba_test_final):.3f} (bulan {test_months})")
print(f"Fitur: {len(feature_cols_raw)} mentah -> {len(feature_cols_clean)} setelah buang redundant -> "
      f"{len(feature_cols_final)} setelah buang tidak signifikan")
print(f"Threshold direkomendasikan: {FINAL_THRESHOLD}")
print(f"\nFile tersimpan di: {OUTPUT_DIR}")
print("  - scored_all_key1_xgb.csv, threshold_analysis_xgb.csv,")
print("  - feature_cleaning_report.csv, feature_importance_xgb.csv, model_eval_curves_xgb.png")
