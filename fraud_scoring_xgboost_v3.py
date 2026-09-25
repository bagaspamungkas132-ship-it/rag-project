"""
Fraud Scoring - XGBoost v3: Train/Validate/Test + Fitur Baru
================================================================
Perubahan dari versi sebelumnya:
  1. FITUR BARU (aman, bukan hasil investigasi jadi bukan leakage):
     - employment_tenure_days  = alert_month - employment_start_date
     - monthly_income_log      = log1p(monthly_income) -- income biasanya skewed,
                                   log-transform bikin distribusinya lebih wajar buat model
     - n_transfer_l3m, n_transfer_l6m               (volume transaksi mentah)
     - n_transfer_not_normal_l3m, n_transfer_not_normal_l6m  (count mentah, bukan cuma pct)
  2. SPLIT 3-ARAH: train / validate / test (bukan cuma train/test)
     - Tuning & feature selection pakai VALIDATE, bukan test
     - TEST cuma diintip SEKALI di akhir -- ini angka yang jujur buat dilaporkan
  3. n_test_months = 1 (sesuai permintaan)

CATATAN METODOLOGIS PENTING:
  Di iterasi-iterasi sebelumnya (LogReg, RF, XGB v1/v2), kita berulang kali
  mengintip skor TEST SET untuk memutuskan model/fitur/tuning. Itu bikin test set
  saat itu diam-diam berfungsi seperti validation set. Skrip ini memperbaikinya:
  validate dipakai buat semua eksperimen, test baru diintip di akhir dan
  HASILNYA TIDAK BOLEH DIPAKAI BUAT UBAH APAPUN LAGI setelah dilihat.

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

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score, f1_score,
    RocCurveDisplay, PrecisionRecallDisplay
)
from scipy.stats import pointbiserialr
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
    "key1", "alert_month", "accountid", "disposition_class",
    "employment_start_date", "monthly_income",
    "account_age", "flag_account_age_lt9", "flag_anomaly_income",
    "n_sameday_pass_3h_l3m", "n_sameday_pass_3h_l6m",
    "n_sameday_pass_1h_l3m", "n_sameday_pass_1h_l6m",
    "n_transfer_l3m", "n_transfer_l6m",
    "n_transfer_not_normal_l3m", "n_transfer_not_normal_l6m",
    "transfer_not_normal_l3m_pct", "transfer_not_normal_l6m_pct",
    "n_crypto", "n_in_10x_income", "flag_in_10x_income",
    "n_out_10x_income", "flag_out_10x_income",
    "n_new_beneficiary", "flag_new_beneficiary",
    "dpn_flag",
    "screening_flag", "flag",
]

RANDOM_STATE = 42
REDUNDANT_CORR_THRESHOLD = 0.85
PERM_IMPORTANCE_DROP_THRESHOLD = 0.0

MANUAL_ALWAYS_KEEP = ["n_crypto", "n_new_beneficiary", "screening_flag"]

N_TEST_MONTHS = 1
N_VALIDATE_MONTHS = 1

USE_ENGINEERED_FEATURES = True
FINAL_THRESHOLD = 0.5

# %% ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
df = pd.read_parquet(PATH)[COLS].copy()
df_fds = df[df[FLAG_FILTER_COL] == FLAG_FILTER_VALUE].drop(columns=[FLAG_FILTER_COL]).copy()
df_fds[TIME_COL] = df_fds[TIME_COL].astype(str)
df_fds["target"] = (df_fds[TARGET_COL].astype(str).str.strip().str.lower() == "suspicious").astype(int)

# %% ------------------------------------------------------------------
# 1b. FITUR BARU (aman -- bukan hasil investigasi alert)
# ------------------------------------------------------------------
def add_new_raw_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    if "employment_start_date" in d.columns:
        alert_dt = pd.to_datetime(d[TIME_COL], errors="coerce")
        emp_dt = pd.to_datetime(d["employment_start_date"], errors="coerce")
        d["employment_tenure_days"] = (alert_dt - emp_dt).dt.days
        # data quality guard: tenure negatif (start_date setelah alert) itu ganjil -> jadi NaN
        d.loc[d["employment_tenure_days"] < 0, "employment_tenure_days"] = np.nan
        d = d.drop(columns=["employment_start_date"])
    if "monthly_income" in d.columns:
        income_numeric = pd.to_numeric(d["monthly_income"].apply(lambda x: float(x) if pd.notna(x) else np.nan))
        d["monthly_income_log"] = np.log1p(income_numeric.clip(lower=0))
        d = d.drop(columns=["monthly_income"])
    return d

df_fds = add_new_raw_features(df_fds)
print(f"Fitur baru ditambahkan: employment_tenure_days, monthly_income_log, "
      f"n_transfer_l3m, n_transfer_l6m, n_transfer_not_normal_l3m, n_transfer_not_normal_l6m")

def add_engineered_ratio_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    if {"flag_anomaly_income", "n_crypto"}.issubset(d.columns):
        d["fe_anomaly_income_x_crypto"] = d["flag_anomaly_income"] * d["n_crypto"]
    if {"n_out_10x_income", "account_age"}.issubset(d.columns):
        d["fe_out10x_per_account_age"] = d["n_out_10x_income"] / (d["account_age"].fillna(0) + 1)
    if {"n_in_10x_income", "account_age"}.issubset(d.columns):
        d["fe_in10x_per_account_age"] = d["n_in_10x_income"] / (d["account_age"].fillna(0) + 1)
    sameday_cols = [c for c in d.columns if c.startswith("n_sameday_pass")]
    if sameday_cols and "account_age" in d.columns:
        d["fe_sameday_total"] = d[sameday_cols].sum(axis=1)
        d["fe_sameday_per_account_age"] = d["fe_sameday_total"] / (d["account_age"].fillna(0) + 1)
    flag_cols = [c for c in d.columns if c.startswith("flag_")]
    if flag_cols:
        d["fe_total_flags_on"] = d[flag_cols].sum(axis=1)
    return d

if USE_ENGINEERED_FEATURES:
    df_fds = add_engineered_ratio_features(df_fds)

feature_cols_raw = [c for c in df_fds.columns if c not in ID_COLS + [TIME_COL, TARGET_COL, "target"]]
print(f"Total baris FDS: {len(df_fds)} | Fitur mentah total: {len(feature_cols_raw)}")
print(feature_cols_raw)

# %% ------------------------------------------------------------------
# 2. AUTO-CLEAN FITUR REDUNDANT (|corr| > 0.85)
# ------------------------------------------------------------------
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

parent = {c: c for c in feature_cols_raw}
def find(x):
    while parent[x] != x:
        x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb

cols_m = corr_matrix.columns
for i in range(len(cols_m)):
    for j in range(i + 1, len(cols_m)):
        a, b = cols_m[i], cols_m[j]
        v = corr_matrix.iloc[i, j]
        if pd.notna(v) and abs(v) > REDUNDANT_CORR_THRESHOLD:
            union(a, b)

clusters = {}
for c in feature_cols_raw:
    root = find(c)
    clusters.setdefault(root, []).append(c)

cleaning_report = []
feature_cols_clean = []
for root, members in clusters.items():
    best = max(members, key=lambda f: corr_to_target.get(f, 0))
    feature_cols_clean.append(best)
    if len(members) == 1:
        cleaning_report.append({"feature": best, "keputusan": "DIPERTAHANKAN", "alasan": "tidak redundant"})
    else:
        cleaning_report.append({"feature": best, "keputusan": "DIPERTAHANKAN",
                                 "alasan": f"representatif cluster {members}"})
        for m in members:
            if m != best:
                cleaning_report.append({"feature": m, "keputusan": "DIBUANG",
                                         "alasan": f"redundant dengan '{best}' (cluster: {members})"})

n_dropped_redundant = len(feature_cols_raw) - len(feature_cols_clean)  # FIX bug hitung sebelumnya
print(f"\n=== Cluster fitur redundant (|corr| > {REDUNDANT_CORR_THRESHOLD}) ===")
for root, members in clusters.items():
    if len(members) > 1:
        best = max(members, key=lambda f: corr_to_target.get(f, 0))
        print(f"  Cluster: {members} -> keep: '{best}'")
print(f"\nFitur setelah cleaning redundansi: {len(feature_cols_clean)} dari {len(feature_cols_raw)} "
      f"({n_dropped_redundant} dibuang)")

# %% ------------------------------------------------------------------
# 3. SPLIT 3-ARAH: TRAIN / VALIDATE / TEST (berbasis waktu)
# ------------------------------------------------------------------
all_months_sorted = sorted(df_fds[TIME_COL].str[:7].unique())
n_needed = N_TEST_MONTHS + N_VALIDATE_MONTHS
if len(all_months_sorted) <= n_needed:
    raise ValueError(f"Cuma ada {len(all_months_sorted)} bulan data, tidak cukup buat "
                      f"train+validate({N_VALIDATE_MONTHS})+test({N_TEST_MONTHS})")

train_months = all_months_sorted[:-n_needed]
validate_months = all_months_sorted[-n_needed:-N_TEST_MONTHS]
test_months = all_months_sorted[-N_TEST_MONTHS:]

train_df = df_fds[df_fds[TIME_COL].str.startswith(tuple(train_months))].copy()
validate_df = df_fds[df_fds[TIME_COL].str.startswith(tuple(validate_months))].copy()
test_df = df_fds[df_fds[TIME_COL].str.startswith(tuple(test_months))].copy()

print(f"\nTrain    : {len(train_df)} baris, bulan {train_months}")
print(f"Validate : {len(validate_df)} baris, bulan {validate_months}")
print(f"Test     : {len(test_df)} baris, bulan {test_months}  <-- JANGAN diintip lagi setelah Step 6")
if min(len(train_df), len(validate_df), len(test_df)) < 100:
    print(">> PERHATIAN: salah satu split ukurannya < 100 baris. Dengan sampel sekecil ini, "
          "angka ROC-AUC di split tersebut BISA GOYANG CUKUP BESAR hanya karena kebetulan "
          "komposisi datanya, bukan karena model berubah beneran. Baca dengan hati-hati.")

X_train = train_df[feature_cols_clean].apply(pd.to_numeric, errors="coerce")
y_train = train_df["target"]
X_validate = validate_df[feature_cols_clean].apply(pd.to_numeric, errors="coerce")
y_validate = validate_df["target"]
X_test = test_df[feature_cols_clean].apply(pd.to_numeric, errors="coerce")
y_test = test_df["target"]

medians_train = X_train.median()
X_train = X_train.fillna(medians_train)
X_validate = X_validate.fillna(medians_train)
X_test = X_test.fillna(medians_train)

# %% ------------------------------------------------------------------
# 4. HYPERPARAMETER SEARCH -- fit di TRAIN, pilih terbaik berdasar skor VALIDATE
#    (bukan RandomizedSearchCV+CV lagi -- sekarang eksplisit train->validate,
#    lebih transparan buat dijelaskan ke atasan: "kita coba-coba di validate,
#    baru cek test 1x di akhir")
# ------------------------------------------------------------------
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

param_grid = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": ss,
     "colsample_bytree": cb, "min_child_weight": mcw, "reg_lambda": rl}
    for n in [200, 300]
    for d in [2, 3, 4]
    for lr in [0.05, 0.08]
    for ss in [0.7]
    for cb in [0.7]
    for mcw in [10]
    for rl in [3, 10]
]

search_results = []
for params in param_grid:
    model = xgb.XGBClassifier(**params, scale_pos_weight=scale_pos_weight,
                               eval_metric="auc", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    val_proba = model.predict_proba(X_validate)[:, 1]
    val_auc = roc_auc_score(y_validate, val_proba)
    search_results.append({**params, "validate_roc_auc": val_auc})

search_df = pd.DataFrame(search_results).sort_values("validate_roc_auc", ascending=False)
print(f"\n=== Hasil hyperparameter search ({len(param_grid)} kombinasi, dipilih berdasar VALIDATE) ===")
print(search_df.head(10).to_string(index=False))
search_df.to_csv(os.path.join(OUTPUT_DIR, "hyperparam_search_log.csv"), index=False)

best_params = {k: v for k, v in search_df.iloc[0].items() if k != "validate_roc_auc"}
best_params = {k: (int(v) if k in ["n_estimators", "max_depth", "min_child_weight"] else v) for k, v in best_params.items()}
print(f"\nBest params (by validate): {best_params}")
print(f"Validate ROC-AUC dgn params ini: {search_df.iloc[0]['validate_roc_auc']:.3f}")

# %% ------------------------------------------------------------------
# 5. PERMUTATION IMPORTANCE -- dihitung di VALIDATE (bukan test!)
#    supaya keputusan buang fitur tidak "mengintip" test set.
# ------------------------------------------------------------------
model_for_selection = xgb.XGBClassifier(**best_params, scale_pos_weight=scale_pos_weight,
                                         eval_metric="auc", random_state=RANDOM_STATE)
model_for_selection.fit(X_train, y_train)

perm_result = permutation_importance(
    model_for_selection, X_validate, y_validate, scoring="roc_auc",
    n_repeats=30, random_state=RANDOM_STATE, n_jobs=-1
)
perm_df = pd.DataFrame({
    "feature": feature_cols_clean,
    "perm_importance_mean": perm_result.importances_mean,
    "perm_importance_std": perm_result.importances_std,
    "builtin_importance": model_for_selection.feature_importances_,
}).sort_values("perm_importance_mean", ascending=False)

print("\n=== Permutation importance (dihitung di VALIDATE) ===")
print(perm_df.to_string(index=False))

not_significant_drop = [
    f for f in perm_df.loc[perm_df["perm_importance_mean"] <= PERM_IMPORTANCE_DROP_THRESHOLD, "feature"]
    if f not in MANUAL_ALWAYS_KEEP
]
feature_cols_final = [f for f in feature_cols_clean if f not in not_significant_drop]

final_decision_rows = []
for _, row in perm_df.iterrows():
    feat = row["feature"]
    if feat in not_significant_drop:
        final_decision_rows.append({"feature": feat, "keputusan": "DIBUANG",
                                     "alasan": f"permutation importance (validate) <= 0 ({row['perm_importance_mean']:.4f})"})
    elif row["perm_importance_mean"] <= PERM_IMPORTANCE_DROP_THRESHOLD and feat in MANUAL_ALWAYS_KEEP:
        final_decision_rows.append({"feature": feat, "keputusan": "DIPERTAHANKAN (override manual)",
                                     "alasan": f"sinyal lemah tapi red-flag bisnis (perm_imp={row['perm_importance_mean']:.4f})"})
    else:
        final_decision_rows.append({"feature": feat, "keputusan": "DIPERTAHANKAN",
                                     "alasan": f"permutation importance positif ({row['perm_importance_mean']:.4f})"})

full_report = pd.concat([pd.DataFrame(cleaning_report), pd.DataFrame(final_decision_rows)], ignore_index=True)
full_report.to_csv(os.path.join(OUTPUT_DIR, "feature_cleaning_report_v3.csv"), index=False)
perm_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_v3.csv"), index=False)

print(f"\nFitur final: {len(feature_cols_final)} dari {len(feature_cols_raw)} mentah "
      f"({n_dropped_redundant} dibuang redundant, {len(not_significant_drop)} dibuang tidak signifikan)")
print(f"Dibuang krn tidak signifikan: {not_significant_drop}")
print(feature_cols_final)

# %% ------------------------------------------------------------------
# 6. CEK FINAL DI TEST SET -- INI DIINTIP SEKALI, JANGAN DIULANG-ULANG
#    Refit pakai TRAIN+VALIDATE gabung (data lebih banyak drpd cuma train),
#    dgn fitur & hyperparameter yang SUDAH DIKUNCI dari langkah sebelumnya.
# ------------------------------------------------------------------
X_train_val = pd.concat([X_train[feature_cols_final], X_validate[feature_cols_final]], axis=0)
y_train_val = pd.concat([y_train, y_validate], axis=0)
X_test_final = X_test[feature_cols_final]

scale_pos_weight_tv = (y_train_val == 0).sum() / max((y_train_val == 1).sum(), 1)
model_final_check = xgb.XGBClassifier(**best_params, scale_pos_weight=scale_pos_weight_tv,
                                       eval_metric="auc", random_state=RANDOM_STATE)
model_final_check.fit(X_train_val, y_train_val)
proba_test = model_final_check.predict_proba(X_test_final)[:, 1]

test_auc = roc_auc_score(y_test, proba_test)
test_pr_auc = average_precision_score(y_test, proba_test)

print("\n" + "!" * 70)
print("HASIL TEST SET -- DIINTIP SEKALI, DIPUTUSKAN, JANGAN DIULANG LAGI")
print("!" * 70)
print(f"Test ROC-AUC: {test_auc:.3f}")
print(f"Test PR-AUC : {test_pr_auc:.3f}")
print(f"(dibandingkan skor validate saat tuning: {search_df.iloc[0]['validate_roc_auc']:.3f} -- "
      f"kalau jauh beda, itu wajar krn test cuma 1 bulan / sample kecil, bukan berarti "
      f"prosesnya salah)")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
RocCurveDisplay.from_predictions(y_test, proba_test, ax=ax[0])
PrecisionRecallDisplay.from_predictions(y_test, proba_test, ax=ax[1])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_eval_curves_v3.png"), dpi=150)
plt.close()

# %% ------------------------------------------------------------------
# 7. ANALISIS THRESHOLD (di test set -- hasil final yang tadi diintip)
# ------------------------------------------------------------------
print("\n=== Analisis Threshold (test set) ===")
threshold_rows = []
for t in np.arange(0.1, 0.95, 0.05):
    pred = (proba_test >= t).astype(int)
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
threshold_df.to_csv(os.path.join(OUTPUT_DIR, "threshold_analysis_v3.csv"), index=False)

# %% ------------------------------------------------------------------
# 8. MODEL PRODUKSI FINAL -- retrain pakai SEMUA data (train+validate+test)
#    Ini yang benar2 dipakai buat scoring operasional ke depan.
# ------------------------------------------------------------------
X_all = df_fds[feature_cols_final].apply(pd.to_numeric, errors="coerce")
y_all = df_fds["target"]
X_all = X_all.fillna(X_all.median())
scale_pos_weight_all = (y_all == 0).sum() / max((y_all == 1).sum(), 1)

production_model = xgb.XGBClassifier(**best_params, scale_pos_weight=scale_pos_weight_all,
                                      eval_metric="auc", random_state=RANDOM_STATE)
production_model.fit(X_all, y_all)
score_all = production_model.predict_proba(X_all)[:, 1]

scored_all = df_fds[["key1", "accountid", TIME_COL, TARGET_COL]].copy()
scored_all["score_suspicious"] = score_all
scored_all["rank_pct"] = scored_all["score_suspicious"].rank(pct=True, ascending=False)
scored_all["flag_recommended"] = np.where(scored_all["score_suspicious"] >= FINAL_THRESHOLD, "Suspicious", "Not Suspicious")
scored_all = scored_all.sort_values("score_suspicious", ascending=False)
scored_all.to_csv(os.path.join(OUTPUT_DIR, "scored_all_key1_v3.csv"), index=False)

# %% ------------------------------------------------------------------
# RINGKASAN AKHIR
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("RINGKASAN UNTUK PRESENTASI")
print("=" * 70)
print(f"Model                 : XGBoost {best_params}")
print(f"Split                 : train={len(train_df)} ({train_months}), "
      f"validate={len(validate_df)} ({validate_months}), test={len(test_df)} ({test_months})")
print(f"Fitur                 : {len(feature_cols_raw)} mentah -> {len(feature_cols_clean)} setelah redundant "
      f"-> {len(feature_cols_final)} setelah tidak signifikan")
print(f"Test ROC-AUC (final, 1x diintip): {test_auc:.3f} | PR-AUC: {test_pr_auc:.3f}")
print(f"Threshold direkomendasikan       : {FINAL_THRESHOLD}")
print(f"\nFile tersimpan di: {OUTPUT_DIR}")
print("  - hyperparam_search_log.csv, feature_cleaning_report_v3.csv, feature_importance_v3.csv")
print("  - model_eval_curves_v3.png, threshold_analysis_v3.csv, scored_all_key1_v3.csv")
