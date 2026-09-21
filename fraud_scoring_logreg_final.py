"""
Fraud Scoring - Finalisasi Model LogReg (buat presentasi ke atasan)
=====================================================================
Output dari script ini:
  1. scored_all_key1.csv       -> score_suspicious untuk SEMUA key1 (bukan cuma test set)
  2. threshold_analysis.csv    -> tabel trade-off precision/recall di berbagai threshold
  3. feature_significance.csv  -> tiap fitur: signifikan secara statistik atau tidak,
                                   dan apakah L1-regularization mempertahankan atau membuang fitur itu
  4. ringkasan tercetak di terminal, siap di-screenshot/copy buat slide

Alur:
  - Step 1-2  : load data, hitung ulang fitur (tanpa feature engineering rasio -- sudah
                terbukti bikin importance ambigu, lihat kesimpulan sebelumnya)
  - Step 3    : evaluasi model dengan time-based split (buat validasi performa, JANGAN
                dipakai buat scoring final -- cuma buat lapor "seberapa bagus model ini")
  - Step 4    : uji apakah semua fitur perlu dipakai (p-value + L1 regularization)
  - Step 5    : analisis threshold (precision/recall di berbagai titik potong)
  - Step 6    : re-train model final pakai SEMUA data historis (train+test digabung),
                lalu score semua key1 -- ini model yang benar-benar dipakai ke depan

requirements:
  pip install pandas numpy scikit-learn scipy matplotlib seaborn
  pip install statsmodels     # opsional, buat p-value formal (kalau tidak ada, pakai fallback)
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score, f1_score,
    RocCurveDisplay, PrecisionRecallDisplay
)

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("statsmodels tidak terinstal -- p-value formal di-skip, pakai fallback "
          "(coef / std-error dari bootstrap sederhana). Install dengan: "
          "pip install statsmodels  (opsional, tidak wajib)")

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

# Threshold yang direkomendasikan buat production -- isi manual setelah lihat
# tabel di Step 5 (threshold_analysis.csv), diskusikan dulu sama atasan/tim fraud
# soal berapa kapasitas review harian mereka.
FINAL_THRESHOLD = 0.5

# %% ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
df = pd.read_parquet(PATH)[COLS].copy()
df_fds = df[df[FLAG_FILTER_COL] == FLAG_FILTER_VALUE].drop(columns=[FLAG_FILTER_COL]).copy()
df_fds[TIME_COL] = df_fds[TIME_COL].astype(str)
df_fds["target"] = (df_fds[TARGET_COL].astype(str).str.strip().str.lower() == "suspicious").astype(int)

feature_cols = [c for c in df_fds.columns if c not in ID_COLS + [TIME_COL, TARGET_COL, "target"]]
print(f"Total baris FDS: {len(df_fds)} | Fitur dipakai: {len(feature_cols)}")
print(feature_cols)

# %% ------------------------------------------------------------------
# 2. TIME-BASED SPLIT (buat evaluasi/validasi performa)
# ------------------------------------------------------------------
all_months_sorted = sorted(df_fds[TIME_COL].str[:7].unique())
n_test_months = 2
train_months = all_months_sorted[:-n_test_months]
test_months = all_months_sorted[-n_test_months:]

train_df = df_fds[df_fds[TIME_COL].str.startswith(tuple(train_months))].copy()
test_df = df_fds[df_fds[TIME_COL].str.startswith(tuple(test_months))].copy()

X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
y_train = train_df["target"]
X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")
y_test = test_df["target"]

medians_train = X_train.median()
X_train = X_train.fillna(medians_train)
X_test = X_test.fillna(medians_train)

print(f"Train: {X_train.shape} (bulan {train_months}) | Test: {X_test.shape} (bulan {test_months})")

# %% ------------------------------------------------------------------
# 3. EVALUASI MODEL (buat validasi performa, dilaporkan ke atasan)
# ------------------------------------------------------------------
scaler_eval = StandardScaler()
X_train_scaled = scaler_eval.fit_transform(X_train)
X_test_scaled = scaler_eval.transform(X_test)

tscv = TimeSeriesSplit(n_splits=3)
lr_search = RandomizedSearchCV(
    LogisticRegression(max_iter=3000, class_weight="balanced"),
    param_distributions={"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    n_iter=6, scoring="roc_auc", cv=tscv, random_state=RANDOM_STATE, n_jobs=-1,
)
lr_search.fit(X_train_scaled, y_train)
logreg_eval = lr_search.best_estimator_
proba_test = logreg_eval.predict_proba(X_test_scaled)[:, 1]

test_auc = roc_auc_score(y_test, proba_test)
test_pr_auc = average_precision_score(y_test, proba_test)
print(f"\n=== Performa model (validasi time-based split) ===")
print(f"Best C: {lr_search.best_params_['C']}")
print(f"CV ROC-AUC : {lr_search.best_score_:.3f}")
print(f"Test ROC-AUC: {test_auc:.3f}")
print(f"Test PR-AUC : {test_pr_auc:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
RocCurveDisplay.from_predictions(y_test, proba_test, ax=ax[0])
PrecisionRecallDisplay.from_predictions(y_test, proba_test, ax=ax[1])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "final_model_eval_curves.png"), dpi=150)
plt.close()

# %% ------------------------------------------------------------------
# 4. APAKAH SEMUA FITUR PERLU DIPAKAI?
#    Cara 1: signifikansi statistik (p-value) pakai statsmodels (kalau ada)
#    Cara 2: L1-regularization (Lasso) -- otomatis "matikan" (coef=0) fitur lemah
# ------------------------------------------------------------------
print("\n=== Cek fitur: signifikan secara statistik? ===")
if HAS_STATSMODELS:
    X_train_sm = sm.add_constant(X_train_scaled)
    logit_model = sm.Logit(y_train.values, X_train_sm)
    logit_result = logit_model.fit(disp=0)
    pvalues = logit_result.pvalues[1:]  # buang konstanta
    sig_df = pd.DataFrame({"feature": feature_cols, "p_value": pvalues.values})
    sig_df["signifikan_5pct"] = sig_df["p_value"] < 0.05
    print(sig_df.sort_values("p_value").to_string(index=False))
else:
    # fallback: pakai |coef| standardized dari LogReg biasa sebagai proxy kepentingan
    logreg_plain = LogisticRegression(max_iter=3000, class_weight="balanced", C=lr_search.best_params_["C"])
    logreg_plain.fit(X_train_scaled, y_train)
    sig_df = pd.DataFrame({"feature": feature_cols, "abs_coef": np.abs(logreg_plain.coef_[0])})
    sig_df = sig_df.sort_values("abs_coef", ascending=False)
    sig_df["signifikan_5pct"] = None  # tidak bisa dihitung tanpa statsmodels
    print("(fallback tanpa statsmodels -- urutan berdasarkan |koefisien| standardized, "
          "bukan p-value formal)")
    print(sig_df.to_string(index=False))

print("\n=== Cek fitur: mana yang di-drop otomatis oleh L1 (Lasso) regularization? ===")
l1_search = RandomizedSearchCV(
    LogisticRegression(max_iter=3000, class_weight="balanced", penalty="l1", solver="liblinear"),
    param_distributions={"C": [0.01, 0.03, 0.1, 0.3, 1.0]},
    n_iter=5, scoring="roc_auc", cv=tscv, random_state=RANDOM_STATE, n_jobs=-1,
)
l1_search.fit(X_train_scaled, y_train)
logreg_l1 = l1_search.best_estimator_
l1_coef_df = pd.DataFrame({"feature": feature_cols, "coef_l1": logreg_l1.coef_[0]})
l1_coef_df["dipertahankan_l1"] = l1_coef_df["coef_l1"] != 0
print(f"Best L1 C: {l1_search.best_params_['C']}")
print(l1_coef_df.sort_values("coef_l1", key=abs, ascending=False).to_string(index=False))

dropped_by_l1 = l1_coef_df.loc[~l1_coef_df["dipertahankan_l1"], "feature"].tolist()
kept_by_l1 = l1_coef_df.loc[l1_coef_df["dipertahankan_l1"], "feature"].tolist()
print(f"\nFitur yang di-drop otomatis oleh L1: {dropped_by_l1 if dropped_by_l1 else '(tidak ada)'}")

# --- Bandingkan performa: semua fitur (L2) vs fitur hasil seleksi L1 ---
proba_l1_test = logreg_l1.predict_proba(X_test_scaled)[:, 1]
l1_test_auc = roc_auc_score(y_test, proba_l1_test)
print(f"\nPerbandingan test ROC-AUC:")
print(f"  Semua {len(feature_cols)} fitur (L2)     : {test_auc:.3f}")
print(f"  Fitur hasil seleksi L1 ({len(kept_by_l1)} fitur): {l1_test_auc:.3f}")
if abs(test_auc - l1_test_auc) < 0.01:
    print("  >> Selisih kecil -- fitur yang di-drop L1 memang tidak banyak membantu, "
          "AMAN untuk dipakai versi lebih ringkas kalau mau model lebih simpel.")
else:
    print("  >> Ada selisih berarti -- pertimbangkan tetap pakai semua fitur (L2) "
          "meskipun sebagian koefisiennya kecil.")

# merge info signifikansi + L1 buat 1 tabel ringkas
feature_report = l1_coef_df.merge(sig_df[["feature"] + (["p_value", "signifikan_5pct"] if HAS_STATSMODELS else ["abs_coef"])], on="feature", how="left")
feature_report.to_csv(os.path.join(OUTPUT_DIR, "feature_significance.csv"), index=False)

# %% ------------------------------------------------------------------
# 5. ANALISIS THRESHOLD -- trade-off precision/recall di berbagai titik potong
# ------------------------------------------------------------------
print("\n=== Analisis Threshold (di test set) ===")
threshold_rows = []
for t in np.arange(0.1, 0.95, 0.05):
    pred = (proba_test >= t).astype(int)
    n_flagged = pred.sum()
    if n_flagged == 0:
        prec = np.nan
    else:
        prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    threshold_rows.append({
        "threshold": round(t, 2),
        "n_alert_ke_flag": n_flagged,
        "pct_dari_total_alert": round(n_flagged / len(y_test), 3),
        "precision": round(prec, 3) if not np.isnan(prec) else None,
        "recall": round(rec, 3),
        "f1_score": round(f1, 3),
    })
threshold_df = pd.DataFrame(threshold_rows)
print(threshold_df.to_string(index=False))
threshold_df.to_csv(os.path.join(OUTPUT_DIR, "threshold_analysis.csv"), index=False)

print(f"\n>> FINAL_THRESHOLD saat ini di-set: {FINAL_THRESHOLD}")
print("   Ubah nilai FINAL_THRESHOLD di CONFIG (atas) berdasarkan tabel ini, "
      "sesuai kapasitas review tim fraud, lalu run ulang.")

# %% ------------------------------------------------------------------
# 6. MODEL FINAL -- re-train pakai SEMUA data historis, lalu score SEMUA key1
#    (bukan cuma test set -- ini yang dipakai buat kebutuhan operasional/presentasi)
# ------------------------------------------------------------------
X_all = df_fds[feature_cols].apply(pd.to_numeric, errors="coerce")
y_all = df_fds["target"]
medians_all = X_all.median()
X_all = X_all.fillna(medians_all)

scaler_final = StandardScaler()
X_all_scaled = scaler_final.fit_transform(X_all)

final_model = LogisticRegression(max_iter=3000, class_weight="balanced", C=lr_search.best_params_["C"])
final_model.fit(X_all_scaled, y_all)

score_all = final_model.predict_proba(X_all_scaled)[:, 1]

final_coef_df = pd.DataFrame({
    "feature": feature_cols, "coef": final_model.coef_[0]
}).sort_values("coef", ascending=False)
final_coef_df["direction"] = np.where(final_coef_df["coef"] > 0, "menaikkan risiko", "menurunkan risiko")
print("\n=== Koefisien model final (dilatih di semua data historis) ===")
print(final_coef_df.to_string(index=False))

scored_all = df_fds[["key1", "accountid", TIME_COL, TARGET_COL]].copy()
scored_all["score_suspicious"] = score_all
scored_all["rank_pct"] = scored_all["score_suspicious"].rank(pct=True, ascending=False)
scored_all["flag_recommended"] = np.where(scored_all["score_suspicious"] >= FINAL_THRESHOLD, "Suspicious", "Not Suspicious")
scored_all = scored_all.sort_values("score_suspicious", ascending=False)
scored_all.to_csv(os.path.join(OUTPUT_DIR, "scored_all_key1.csv"), index=False)

print(f"\nTotal key1 di-score: {len(scored_all)}")
print(f"Jumlah yang di-flag Suspicious pada threshold {FINAL_THRESHOLD}: "
      f"{(scored_all['flag_recommended'] == 'Suspicious').sum()} "
      f"({(scored_all['flag_recommended'] == 'Suspicious').mean():.1%})")
print("\nContoh 10 skor tertinggi:")
print(scored_all.head(10).to_string(index=False))

# %% ------------------------------------------------------------------
# RINGKASAN AKHIR -- buat disalin ke slide presentasi
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("RINGKASAN UNTUK PRESENTASI")
print("=" * 70)
print(f"Model               : Logistic Regression (C={lr_search.best_params_['C']}, class_weight=balanced)")
print(f"Data training final  : {len(df_fds)} baris ({all_months_sorted[0]} s/d {all_months_sorted[-1]})")
print(f"Validasi (holdout)   : ROC-AUC={test_auc:.3f}, PR-AUC={test_pr_auc:.3f} (di bulan {test_months})")
print(f"Jumlah fitur dipakai : {len(feature_cols)} dari {len(feature_cols)} tersedia")
print(f"  -> {len(kept_by_l1)} fitur terbukti kontributif (L1), "
      f"{len(dropped_by_l1)} fitur kontribusinya minim tapi tetap disertakan (lihat feature_significance.csv)")
print(f"Threshold direkomendasikan: {FINAL_THRESHOLD} "
      f"(lihat threshold_analysis.csv untuk opsi lain sesuai kapasitas review)")
print(f"\nFile tersimpan di: {OUTPUT_DIR}")
print("  - scored_all_key1.csv       (score tiap key1, siap dipakai operasional)")
print("  - threshold_analysis.csv    (tabel precision/recall per threshold)")
print("  - feature_significance.csv  (signifikansi & seleksi fitur)")
print("  - final_model_eval_curves.png (ROC & PR curve validasi)")
