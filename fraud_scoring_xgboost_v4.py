"""
Fraud Scoring - XGBoost v4 (data gabungan: KYC + rule lama + transaksi mentah)
=================================================================================
Perubahan utama dari v3:
  1. WHITELIST fitur (bukan "semua kolom kecuali ID") -> kolom baru yang tidak
     sengaja ikut (customername, teks bebas, skor screening, dll) tidak bisa
     nyelonong jadi fitur / leakage.
  2. monthly_income di-parse ROBUST jadi numerik (Decimal / "5.000.000" / "Rp 7,500,000").
  3. XGBoost dibiarkan menangani NaN sendiri (tanpa median imputation) -> "data
     kosong" tidak dikaburkan, dan justru bisa jadi sinyal (KYC tidak lengkap).
  4. Fitur turunan baru: transaksi vs income (profile mismatch), flow balance
     (uang masuk ~ keluar = pass-through/mule), rasio gagal, rasio jam ganjil, burst.
  5. Redundansi dicek pakai SPEARMAN (data ekor panjang: total_amount max 36 M vs
     median 830 jt -> Pearson bisa menyesatkan), representatif cluster dipilih
     berdasar AUC univariat.
  6. Hyperparameter, seleksi fitur & THRESHOLD dipilih dari ROLLING-ORIGIN
     (beberapa fold out-of-time), bukan 1 bulan validate yang goyang.
  7. TEST (bulan terakhir) diintip SEKALI di akhir, dilaporkan dengan 95% CI bootstrap.
  8. Output tambahan buat presentasi/governance: lift table, PSI (drift),
     ablation per kelompok data, reason codes per alert, model card.

requirements:
  pip install pandas numpy scikit-learn scipy xgboost matplotlib pyarrow
"""

# %% ------------------------------------------------------------------
# 0. IMPORT & CONFIG
# ------------------------------------------------------------------
import os
os.environ["MPLBACKEND"] = "Agg"

import re
import json
import fnmatch
import datetime
import warnings
from decimal import Decimal

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, fbeta_score, RocCurveDisplay, PrecisionRecallDisplay,
)
import xgboost as xgb

pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# >>> GANTI ke file hasil gabungan (parquet atau csv) <<<
PATH = "/home/cdsw/parquet_output/result/trx_all_enriched_merged.parquet"

KEY_COL, ACCOUNT_COL, TIME_COL, TARGET_COL = "key1", "accountid", "alert_month", "disposition_class"
FLAG_FILTER_COL, FLAG_FILTER_VALUE = "flag", "FDS"
EMPLOYMENT_START_COL = "employment_start_date"

# Hanya kolom di list ini yang boleh jadi fitur (yang tidak ada di data otomatis di-skip).
CANDIDATE_FEATURES = [
    # KYC / profil
    "account_age", "flag_account_age_lt9", "flag_anomaly_income", "monthly_income",
    # rule/behavior lama
    "n_sameday_pass_3h_l3m", "n_sameday_pass_3h_l6m", "n_sameday_pass_1h_l3m", "n_sameday_pass_1h_l6m",
    "n_transfer_l3m", "n_transfer_l6m", "n_transfer_not_normal_l3m", "n_transfer_not_normal_l6m",
    "transfer_not_normal_l3m_pct", "transfer_not_normal_l6m_pct",
    "n_crypto", "n_in_10x_income", "flag_in_10x_income", "n_out_10x_income", "flag_out_10x_income",
    "n_new_beneficiary", "flag_new_beneficiary", "dpn_flag", "screening_flag",
    # transaksi mentah (SQL v1)
    "n_trx_l3m", "total_amount_l3m", "avg_amount_l3m", "max_amount_l3m",
    "total_amount_out_l3m", "total_amount_in_l3m",
    "n_unique_counterparty_l3m", "n_unique_bank_l3m", "n_channel_l3m",
    "n_failed_trx_l3m", "n_trx_odd_hour_l3m",
    "max_counterparty_popularity_l3m", "avg_counterparty_popularity_l3m",   # LEGACY (lihat catatan SQL v2)
    # SQL v2 (opsional, dipakai kalau kolomnya ada)
    "max_cp_popularity_asof_l3m", "avg_cp_popularity_asof_l3m", "n_cp_shared_ge3_l3m",
    "n_trx_l30d", "n_trx_l7d", "total_out_l30d", "total_out_l7d",
    "n_unique_bank_norm_l3m", "n_trx_night_l3m",
]
LEGACY_POPULARITY = ["max_counterparty_popularity_l3m", "avg_counterparty_popularity_l3m"]
ASOF_POPULARITY = ["max_cp_popularity_asof_l3m", "avg_cp_popularity_asof_l3m", "n_cp_shared_ge3_l3m"]

# Skor match watchlist: DEFAULT OFF sampai provenance-nya dikonfirmasi (lihat output diagnostik).
USE_DPN_MATCH_SCORE = False
USE_SCREENING_MATCH_SCORE = False

RANDOM_STATE = 42
N_TEST_MONTHS = 1
N_VALIDATE_MONTHS = 1
REDUNDANT_CORR_THRESHOLD = 0.85
PERM_IMPORTANCE_DROP_THRESHOLD = 0.0
PERM_REPEATS = 10
MIN_FEATURES = 6
PARSIMONY_TOLERANCE = 0.01          # set fitur lebih kecil dipakai kalau AUC OOT turun < ini
MANUAL_ALWAYS_KEEP = ["n_crypto", "n_new_beneficiary", "screening_flag"]

# Pemilihan threshold (dihitung dari prediksi out-of-time SEBELUM test)
THRESHOLD_STRATEGY = "target_precision"   # "target_precision" | "fbeta" | "fixed"
TARGET_PRECISION = 0.70
MIN_FLAGGED_FOR_THRESHOLD = 30            # minimal alert ter-flag di pooled OOT biar precision tidak cuma kebetulan
FBETA = 0.5                               # <1 = lebih menghargai precision
FIXED_THRESHOLD = 0.5
N_BOOT = 1000

PARAM_GRID = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr, "subsample": 0.7,
     "colsample_bytree": 0.7, "min_child_weight": 10, "reg_lambda": rl}
    for n in [200, 300] for d in [2, 3, 4] for lr in [0.05, 0.08] for rl in [3, 10]
]
INT_PARAMS = ["n_estimators", "max_depth", "min_child_weight"]

FEATURE_GROUPS = {          # urutan = prioritas (first match)
    "rasio_turunan": ["fe_*"],
    "jaringan_counterparty": ["*popularity*", "n_cp_shared*"],
    "transaksi_mentah_SQL": ["n_trx_*", "total_*", "avg_amount_*", "max_amount_*",
                             "n_unique_*", "n_channel_*", "n_failed_*"],
    "KYC_profil": ["account_age", "flag_account_age_lt9", "monthly_income",
                   "employment_tenure_days", "flag_income_missing"],
    "rule_lama": ["*"],
}


# %% ------------------------------------------------------------------
# HELPER
# ------------------------------------------------------------------
def parse_number(x):
    """Parse angka robust: Decimal, int/float, '5.000.000', '5,000,000', 'Rp 7.500.000,50'."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating, Decimal)):
        try:
            return float(x)
        except Exception:
            return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "null", "-"):
        return np.nan
    s = re.sub(r"[^0-9,.\-]", "", s)
    if s in ("", "-", ".", ","):
        return np.nan
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        s = s.replace(",", "") if (len(parts) > 2 or len(parts[-1]) == 3) else s.replace(",", ".")
    elif "." in s:
        parts = s.split(".")
        if len(parts) > 2 or (len(parts[-1]) == 3 and 1 <= len(parts[0]) <= 3):
            s = s.replace(".", "")
    try:
        return float(s)
    except Exception:
        return np.nan


def sdiv(a, b):
    """Pembagian aman: penyebut 0 -> NaN (dibiarkan, XGBoost paham NaN)."""
    return a / b.where(b != 0)


def univariate_auc(data, feats, target="target"):
    rows = []
    for f in feats:
        x = data[f]
        if x.nunique(dropna=True) < 2:
            rows.append({"feature": f, "auc": 0.5, "strength": 0.0, "n_missing": int(x.isna().sum())})
            continue
        auc = roc_auc_score(data[target], x.fillna(x.median()))
        rows.append({"feature": f, "auc": auc, "strength": abs(auc - 0.5), "n_missing": int(x.isna().sum())})
    return pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)


def psi(expected, actual, bins=10):
    expected, actual = expected.dropna(), actual.dropna()
    if len(expected) == 0 or len(actual) == 0 or expected.nunique() < 2:
        return np.nan
    qs = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(qs) < 4:   # fitur biner / sedikit nilai unik
        cats = sorted(set(expected.unique()) | set(actual.unique()))
        e = expected.value_counts(normalize=True).reindex(cats).fillna(0) + 1e-4
        a = actual.value_counts(normalize=True).reindex(cats).fillna(0) + 1e-4
    else:
        qs[0], qs[-1] = -np.inf, np.inf
        e = pd.cut(expected, qs).value_counts(normalize=True, sort=False) + 1e-4
        a = pd.cut(actual, qs).value_counts(normalize=True, sort=False) + 1e-4
    return float(((a.values - e.values) * np.log(a.values / e.values)).sum())


def bootstrap_ci(y, p, fn, n_boot=N_BOOT, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    y, p = np.asarray(y), np.asarray(p)
    n, vals = len(y), []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], p[idx]))
    return tuple(np.percentile(vals, [2.5, 97.5])) if vals else (np.nan, np.nan)


def gains_table(y, p, n_bins=10):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p", ascending=False).reset_index(drop=True)
    d["bin"] = pd.qcut(np.arange(len(d)), q=min(n_bins, len(d)), labels=False) + 1
    base = d["y"].mean()
    g = d.groupby("bin").agg(n=("y", "size"), n_suspicious=("y", "sum"),
                             actual_rate=("y", "mean"), mean_score=("p", "mean"),
                             min_score=("p", "min")).reset_index()
    g["lift"] = g["actual_rate"] / base if base > 0 else np.nan
    g["cum_capture"] = g["n_suspicious"].cumsum() / max(d["y"].sum(), 1)
    g["cum_precision"] = g["n_suspicious"].cumsum() / g["n"].cumsum()
    return g


def threshold_table(y, p, thresholds=None):
    thresholds = np.arange(0.10, 0.96, 0.05) if thresholds is None else thresholds
    y, p, rows = np.asarray(y), np.asarray(p), []
    for t in thresholds:
        pred = (p >= t).astype(int)
        n_flag = int(pred.sum())
        rows.append({
            "threshold": round(float(t), 2), "n_alert_ke_flag": n_flag,
            "pct_dari_total_alert": round(n_flag / len(y), 3),
            "precision": precision_score(y, pred, zero_division=0) if n_flag > 0 else np.nan,
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "f_beta": fbeta_score(y, pred, beta=FBETA, zero_division=0),
        })
    return pd.DataFrame(rows)


def top_reasons_from_contribs(contribs, feature_names, X, k=3):
    """contribs: (n, n_fitur+1) hasil pred_contribs (kolom terakhir = bias)."""
    c = np.asarray(contribs)[:, :-1]
    order = np.argsort(-c, axis=1)[:, :k]
    out = []
    for i in range(c.shape[0]):
        parts = []
        for j in order[i]:
            if c[i, j] > 0:
                v = X.iloc[i, j]
                parts.append(f"{feature_names[j]}={v:.4g}" if pd.notna(v) else f"{feature_names[j]}=NA")
        out.append(" | ".join(parts))
    return out


def assign_group(f):
    for g, pats in FEATURE_GROUPS.items():
        if any(fnmatch.fnmatch(f, p) for p in pats):
            return g
    return "lainnya"


# %% ------------------------------------------------------------------
# 1. LOAD & DATA QUALITY
# ------------------------------------------------------------------
df_raw = pd.read_parquet(PATH) if PATH.endswith(".parquet") else pd.read_csv(PATH)
print(f"Baris mentah: {len(df_raw)} | kolom: {df_raw.shape[1]}")

missing_required = [c for c in [KEY_COL, ACCOUNT_COL, TIME_COL, TARGET_COL] if c not in df_raw.columns]
if missing_required:
    raise ValueError(f"Kolom wajib tidak ada: {missing_required}")

df = df_raw.copy()
if FLAG_FILTER_COL in df.columns:
    df = df[df[FLAG_FILTER_COL] == FLAG_FILTER_VALUE].copy()
df[TIME_COL] = df[TIME_COL].astype(str)
df["month"] = df[TIME_COL].str[:7]
df["target"] = (df[TARGET_COL].astype(str).str.strip().str.lower() == "suspicious").astype(int)
print(f"Baris FDS: {len(df)} | base rate suspicious: {df['target'].mean():.3f}")
if df[KEY_COL].duplicated().any():
    print(f">> PERHATIAN: key1 duplikat = {int(df[KEY_COL].duplicated().sum())} baris. Cek proses merge (join 1-ke-banyak?).")

candidates = list(CANDIDATE_FEATURES)
if USE_DPN_MATCH_SCORE:
    candidates.append("dpn_match_score")
if USE_SCREENING_MATCH_SCORE:
    candidates.append("screening_match_score")

present = [c for c in candidates if c in df.columns]
missing = [c for c in candidates if c not in df.columns]
print(f"Fitur kandidat ada di data: {len(present)} | tidak ada (di-skip): {missing}")

income_raw = df["monthly_income"].copy() if "monthly_income" in df.columns else None
for c in present:
    df[c] = df[c].map(parse_number) if c == "monthly_income" else pd.to_numeric(df[c], errors="coerce")

if income_raw is not None:
    inc = df["monthly_income"]
    failed = income_raw.notna() & inc.isna() & (income_raw.astype(str).str.strip() != "")
    print("\n=== Cek monthly_income ===")
    print(f"  gagal parse: {int(failed.sum())} | kosong: {int(inc.isna().sum())} | <=0: {int((inc <= 0).sum())} | nilai unik: {inc.nunique()}")
    if failed.any():
        print("  contoh nilai gagal parse:", income_raw[failed].astype(str).head(5).tolist())
    print("  nilai paling sering (indikasi bucket/placeholder):")
    print(inc.value_counts(normalize=True).head(5).round(3).to_string())

print("\n=== Cek konsistensi hasil merge ===")
if {"n_transfer_l3m", "n_trx_l3m"} <= set(df.columns):
    print(f"  n_transfer_l3m > n_trx_l3m: {(df['n_transfer_l3m'] > df['n_trx_l3m']).mean():.1%} baris "
          f"(idealnya ~0% jika transfer adalah bagian dari semua transaksi; beda definisi kalau tinggi)")
if {"total_amount_l3m", "total_amount_out_l3m", "total_amount_in_l3m"} <= set(df.columns):
    rel = (df["total_amount_l3m"] - df["total_amount_out_l3m"] - df["total_amount_in_l3m"]).abs() / df["total_amount_l3m"].where(df["total_amount_l3m"] != 0)
    print(f"  total != out+in (>1%): {(rel > 0.01).mean():.1%} baris")
if {"avg_amount_l3m", "n_trx_l3m", "total_amount_l3m"} <= set(df.columns):
    rel = (df["avg_amount_l3m"] * df["n_trx_l3m"] - df["total_amount_l3m"]).abs() / df["total_amount_l3m"].where(df["total_amount_l3m"] != 0)
    print(f"  avg*n != total (>1%): {(rel > 0.01).mean():.1%} baris")

print("\n=== Diagnostik skor match watchlist (dipakai HANYA jika toggle ON) ===")
for sc, fl in [("dpn_match_score", "dpn_flag"), ("screening_match_score", "screening_flag")]:
    if sc in df.columns:
        s = pd.to_numeric(df[sc], errors="coerce")
        n100 = (s >= 100).mean()
        fl_rate = df[fl].mean() if fl in df.columns else np.nan
        auc_s = roc_auc_score(df["target"], s.fillna(0)) if s.nunique() > 1 else 0.5
        print(f"  {sc}: rata-rata={s.mean():.1f} | skor>=100: {n100:.1%} | {fl} menyala: {fl_rate:.1%} | AUC univariat={auc_s:.3f}")
        if n100 > 5 * max(fl_rate, 0.001):
            print(f"  >> JANGGAL: skor sempurna jauh lebih sering dari flag. Kemungkinan fuzzy-match nama (false positive tinggi), "
                  f"terisi saat investigasi, atau memakai watchlist versi HARI INI (bukan versi saat alert). Konfirmasi dulu.")

# %% ------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ------------------------------------------------------------------
fe_cols = []

def add_fe(name, series):
    df[name] = series.replace([np.inf, -np.inf], np.nan)
    fe_cols.append(name)

if EMPLOYMENT_START_COL in df.columns:
    tenure = (pd.to_datetime(df[TIME_COL], errors="coerce") - pd.to_datetime(df[EMPLOYMENT_START_COL], errors="coerce")).dt.days
    df["employment_tenure_days"] = tenure.where(tenure >= 0)      # tenure negatif = data ganjil -> NaN
    fe_cols_kyc = ["employment_tenure_days"]
    present.append("employment_tenure_days")
else:
    print("employment_start_date tidak ada di data -> employment_tenure_days di-skip")

if "monthly_income" in df.columns:
    df["flag_income_missing"] = (df["monthly_income"].isna() | (df["monthly_income"] <= 0)).astype(int)
    present.append("flag_income_missing")
    inc_pos = df["monthly_income"].where(df["monthly_income"] > 0)
    if "total_amount_out_l3m" in df.columns:
        add_fe("fe_out_to_income", sdiv(df["total_amount_out_l3m"], inc_pos * 3))
    if "total_amount_in_l3m" in df.columns:
        add_fe("fe_in_to_income", sdiv(df["total_amount_in_l3m"], inc_pos * 3))
    if "max_amount_l3m" in df.columns:
        add_fe("fe_maxamt_to_income", sdiv(df["max_amount_l3m"], inc_pos))

if {"total_amount_out_l3m", "total_amount_in_l3m"} <= set(df.columns):
    mx = np.maximum(df["total_amount_out_l3m"], df["total_amount_in_l3m"])
    mn = np.minimum(df["total_amount_out_l3m"], df["total_amount_in_l3m"])
    add_fe("fe_flow_balance", sdiv(mn, mx))                      # ~1 = uang masuk ~ keluar (pass-through)
if {"n_failed_trx_l3m", "n_trx_l3m"} <= set(df.columns):
    add_fe("fe_failed_ratio", sdiv(df["n_failed_trx_l3m"], df["n_trx_l3m"]))
odd_col = next((c for c in ["n_trx_odd_hour_l3m", "n_trx_night_l3m"] if c in df.columns), None)
if odd_col and "n_trx_l3m" in df.columns:
    add_fe("fe_odd_hour_ratio", sdiv(df[odd_col], df["n_trx_l3m"]))
if {"n_trx_l3m", "n_unique_counterparty_l3m"} <= set(df.columns):
    add_fe("fe_trx_per_counterparty", sdiv(df["n_trx_l3m"], df["n_unique_counterparty_l3m"]))
if {"n_trx_l30d", "n_trx_l3m"} <= set(df.columns):
    add_fe("fe_burst_trx_30d", sdiv(df["n_trx_l30d"] * 3, df["n_trx_l3m"]))
if {"n_trx_l7d", "n_trx_l3m"} <= set(df.columns):
    add_fe("fe_burst_trx_7d", sdiv(df["n_trx_l7d"] * 13, df["n_trx_l3m"]))
if {"total_out_l30d", "total_amount_out_l3m"} <= set(df.columns):
    add_fe("fe_burst_out_30d", sdiv(df["total_out_l30d"] * 3, df["total_amount_out_l3m"]))
# fitur turunan lama yang sebelumnya terbukti berguna
if {"n_out_10x_income", "account_age"} <= set(df.columns):
    add_fe("fe_out10x_per_account_age", df["n_out_10x_income"] / (df["account_age"].fillna(0) + 1))
if {"n_in_10x_income", "account_age"} <= set(df.columns):
    add_fe("fe_in10x_per_account_age", df["n_in_10x_income"] / (df["account_age"].fillna(0) + 1))
orig_flags = [c for c in ["flag_account_age_lt9", "flag_anomaly_income", "flag_in_10x_income",
                          "flag_out_10x_income", "flag_new_beneficiary", "dpn_flag", "screening_flag"] if c in df.columns]
if orig_flags:
    add_fe("fe_total_flags_on", df[orig_flags].sum(axis=1))

if any(c in df.columns for c in ASOF_POPULARITY):
    dropped = [c for c in LEGACY_POPULARITY if c in present]
    present = [c for c in present if c not in LEGACY_POPULARITY]
    if dropped:
        print(f"\nFitur popularity versi as-of tersedia -> versi legacy di-drop otomatis: {dropped}")
else:
    if any(c in present for c in LEGACY_POPULARITY):
        print("\n>> CATATAN: popularity legacy dihitung dari snapshot BULANAN (termasuk hari setelah tanggal alert). "
              "Ada risiko leakage ringan. Gunakan SQL v2 (as-of) dan gabungkan kolomnya; ablation di bawah "
              "akan menunjukkan seberapa besar kontribusinya.")

feature_pool = [c for c in dict.fromkeys(present + fe_cols) if c in df.columns]
const_cols = [c for c in feature_pool if df[c].nunique(dropna=True) < 2]
if const_cols:
    print(f"Fitur konstan (di-drop): {const_cols}")
feature_pool = [c for c in feature_pool if c not in const_cols]
for c in feature_pool:
    df[c] = pd.to_numeric(df[c], errors="coerce")
print(f"\nTotal kandidat fitur (mentah + turunan): {len(feature_pool)}")

# %% ------------------------------------------------------------------
# 3. LEAKAGE RADAR (AUC univariat) & CLEANING REDUNDANSI (Spearman)
# ------------------------------------------------------------------
uni = univariate_auc(df, feature_pool)
uni.to_csv(os.path.join(OUTPUT_DIR, "univariate_auc_v4.csv"), index=False)
print("\n=== Leakage radar: 10 fitur dengan AUC univariat terjauh dari 0.5 ===")
print(uni.head(10).to_string(index=False))
sus = uni[uni["strength"] > 0.30]
if len(sus):
    print(f">> WASPADA: fitur ini AUC univariat > 0.80/< 0.20 -> cek provenance & timing datanya: {sus['feature'].tolist()}")
strength = uni.set_index("feature")["strength"].to_dict()

corr = df[feature_pool].corr(method="spearman")
parent = {c: c for c in feature_pool}
def _find(x):
    while parent[x] != x:
        x = parent[x]
    return x
for i, a in enumerate(feature_pool):
    for b in feature_pool[i + 1:]:
        v = corr.loc[a, b]
        if pd.notna(v) and abs(v) > REDUNDANT_CORR_THRESHOLD:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[ra] = rb
clusters = {}
for c in feature_pool:
    clusters.setdefault(_find(c), []).append(c)

decision_rows, feature_cols_clean = [], []
for members in clusters.values():
    best = max(members, key=lambda f: strength.get(f, 0))
    feature_cols_clean.append(best)
    if len(members) == 1:
        decision_rows.append({"feature": best, "keputusan": "DIPERTAHANKAN", "tahap": "redundansi", "alasan": "tidak redundant"})
    else:
        decision_rows.append({"feature": best, "keputusan": "DIPERTAHANKAN", "tahap": "redundansi",
                              "alasan": f"representatif cluster {members}"})
        for m in members:
            if m != best:
                decision_rows.append({"feature": m, "keputusan": "DIBUANG", "tahap": "redundansi",
                                      "alasan": f"redundant (Spearman>{REDUNDANT_CORR_THRESHOLD}) dgn '{best}'"})
        print(f"  Cluster redundant: {members} -> keep '{best}'")
        if "n_in_10x_income" in members and "n_out_10x_income" in members:
            print("  >> CEK DOMAIN: uang MASUK vs KELUAR di atas 10x income beda makna bisnis; konfirmasi ke tim fraud.")
print(f"Setelah cleaning redundansi: {len(feature_cols_clean)} dari {len(feature_pool)}")

# %% ------------------------------------------------------------------
# 4. SPLIT WAKTU: TRAIN / VALIDATE / TEST
# ------------------------------------------------------------------
all_months = sorted(df["month"].unique())
if len(all_months) < N_TEST_MONTHS + N_VALIDATE_MONTHS + 1:
    raise ValueError(f"Bulan tersedia ({len(all_months)}) tidak cukup untuk train+validate+test")
test_months = all_months[-N_TEST_MONTHS:]
validate_months = all_months[-(N_TEST_MONTHS + N_VALIDATE_MONTHS):-N_TEST_MONTHS]
train_months = all_months[:-(N_TEST_MONTHS + N_VALIDATE_MONTHS)]
pre_test_months = train_months + validate_months

train_df = df[df["month"].isin(train_months)]
validate_df = df[df["month"].isin(validate_months)]
test_df = df[df["month"].isin(test_months)]
df_pre = df[df["month"].isin(pre_test_months)].copy()

print(f"\nTrain    : {len(train_df):4d} baris, bulan {train_months}")
print(f"Validate : {len(validate_df):4d} baris, bulan {validate_months}")
print(f"Test     : {len(test_df):4d} baris, bulan {test_months}  <-- diintip SEKALI di Step 9")
seen_accounts = set(df_pre[ACCOUNT_COL])
overlap = test_df[ACCOUNT_COL].isin(seen_accounts).mean()
print(f"Akun di test yang sudah muncul di train/validate: {overlap:.1%} "
      f"({'aman' if overlap < 0.15 else 'tinggi -> lihat AUC khusus akun baru di Step 9'})")
epv = df_pre["target"].sum() / max(len(feature_cols_clean), 1)
print(f"Events-per-variable (positif pre-test / jumlah fitur): {epv:.1f} "
      f"({'ok' if epv >= 10 else 'tipis -> waspada overfit, pertahankan model dangkal + regularisasi'})")

min_train = 2 if len(pre_test_months) >= 3 else 1
folds = [(pre_test_months[:i], pre_test_months[i]) for i in range(min_train, len(pre_test_months))]
print(f"Rolling-origin folds: {[(f[0][0] + '..' + f[0][-1], f[1]) for f in folds]}")

# %% ------------------------------------------------------------------
# 5. FUNGSI MODEL & ROLLING-ORIGIN
# ------------------------------------------------------------------
def fit_model(params, X, y):
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    m = xgb.XGBClassifier(**params, scale_pos_weight=spw, eval_metric="auc", random_state=RANDOM_STATE)
    m.fit(X, y)
    return m


def rolling_eval(params, features, want_preds=False, want_perm=False):
    aucs, info, preds, perms = [], [], [], []
    for tr_m, te_m in folds:
        tr, te = df_pre[df_pre["month"].isin(tr_m)], df_pre[df_pre["month"] == te_m]
        if tr["target"].nunique() < 2 or te["target"].nunique() < 2:
            continue
        m = fit_model(params, tr[features], tr["target"])
        p = m.predict_proba(te[features])[:, 1]
        auc = roc_auc_score(te["target"], p)
        aucs.append(auc)
        info.append({"train_sampai": tr_m[-1], "test_bulan": te_m, "n_train": len(tr), "n_test": len(te), "roc_auc": auc})
        if want_preds:
            preds.append(pd.DataFrame({KEY_COL: te[KEY_COL].values, "month": te_m, "y": te["target"].values, "p": p}))
        if want_perm:
            r = permutation_importance(m, te[features], te["target"], scoring="roc_auc",
                                       n_repeats=PERM_REPEATS, random_state=RANDOM_STATE)
            perms.append(pd.Series(r.importances_mean, index=features))
    out = {"mean_auc": float(np.mean(aucs)) if aucs else np.nan,
           "std_auc": float(np.std(aucs)) if aucs else np.nan,
           "fold_aucs": aucs, "fold_info": pd.DataFrame(info)}
    if want_preds:
        out["preds"] = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(columns=[KEY_COL, "month", "y", "p"])
    if want_perm:
        out["perm"] = pd.concat(perms, axis=1).mean(axis=1) if perms else pd.Series(dtype=float)
    return out


# %% ------------------------------------------------------------------
# 6. HYPERPARAMETER SEARCH (rata-rata AUC beberapa fold out-of-time)
# ------------------------------------------------------------------
rows = []
for params in PARAM_GRID:
    r = rolling_eval(params, feature_cols_clean)
    rows.append({**params, "mean_auc": r["mean_auc"], "std_auc": r["std_auc"],
                 "min_auc": min(r["fold_aucs"]) if r["fold_aucs"] else np.nan})
search_df = pd.DataFrame(rows).sort_values("mean_auc", ascending=False).reset_index(drop=True)
search_df.to_csv(os.path.join(OUTPUT_DIR, "hyperparam_search_v4.csv"), index=False)
print(f"\n=== Hyperparameter search ({len(PARAM_GRID)} kombinasi x {len(folds)} fold out-of-time) ===")
print(search_df.head(8).to_string(index=False))
best_params = {k: (int(search_df.iloc[0][k]) if k in INT_PARAMS else float(search_df.iloc[0][k])) for k in PARAM_GRID[0]}
print(f"Best params: {best_params}")
print(f"Selisih AUC kombinasi terbaik vs median semua kombinasi: {search_df['mean_auc'].iloc[0] - search_df['mean_auc'].median():.3f} "
      f"(kecil = tuning tidak terlalu berpengaruh)")

# %% ------------------------------------------------------------------
# 7. SELEKSI FITUR (permutation importance rata-rata lintas fold) + PARSIMONY RULE
# ------------------------------------------------------------------
base = rolling_eval(best_params, feature_cols_clean, want_perm=True)
perm_s = base["perm"].sort_values(ascending=False)
print("\n=== Permutation importance (rata-rata lintas fold OOT) ===")
print(perm_s.round(4).to_string())

worst_first = perm_s.sort_values().index.tolist()
drop_candidates = [f for f in worst_first
                   if perm_s[f] <= PERM_IMPORTANCE_DROP_THRESHOLD and f not in MANUAL_ALWAYS_KEEP]
drop_candidates = drop_candidates[:max(0, len(feature_cols_clean) - MIN_FEATURES)]
pruned = [f for f in feature_cols_clean if f not in drop_candidates]
after = rolling_eval(best_params, pruned)

if after["mean_auc"] >= base["mean_auc"] - PARSIMONY_TOLERANCE:
    final_features = pruned
    print(f"\nParsimony rule: set ringkas ({len(pruned)} fitur) AUC OOT {after['mean_auc']:.3f} vs "
          f"penuh ({len(feature_cols_clean)}) {base['mean_auc']:.3f} -> pakai yang ringkas")
else:
    final_features = feature_cols_clean
    drop_candidates = []
    print(f"\nParsimony rule: set ringkas turun terlalu banyak ({after['mean_auc']:.3f} vs {base['mean_auc']:.3f}) "
          f"-> pertahankan set penuh")

for f in feature_cols_clean:
    if f in drop_candidates:
        decision_rows.append({"feature": f, "keputusan": "DIBUANG", "tahap": "permutation",
                              "alasan": f"permutation importance OOT <= 0 ({perm_s[f]:.4f})"})
    elif perm_s.get(f, 0) <= PERM_IMPORTANCE_DROP_THRESHOLD and f in MANUAL_ALWAYS_KEEP:
        decision_rows.append({"feature": f, "keputusan": "DIPERTAHANKAN (override manual)", "tahap": "permutation",
                              "alasan": f"sinyal lemah tapi red-flag bisnis ({perm_s.get(f, 0):.4f})"})
pd.DataFrame(decision_rows).to_csv(os.path.join(OUTPUT_DIR, "feature_decisions_v4.csv"), index=False)
print(f"Fitur final: {len(final_features)} -> {final_features}")

final_eval = rolling_eval(best_params, final_features, want_preds=True)
print("\n=== Performa per fold out-of-time (fitur final) ===")
print(final_eval["fold_info"].round(3).to_string(index=False))
print(f"Rata-rata AUC OOT: {final_eval['mean_auc']:.3f} (std antar fold {final_eval['std_auc']:.3f})")

# %% ------------------------------------------------------------------
# 8. ABLATION PER KELOMPOK DATA + THRESHOLD DARI PREDIKSI OOT
# ------------------------------------------------------------------
group_rows = []
groups = {}
for f in final_features:
    groups.setdefault(assign_group(f), []).append(f)
for g, members in groups.items():
    rest = [f for f in final_features if f not in members]
    if len(rest) < 2:
        continue
    r = rolling_eval(best_params, rest)
    group_rows.append({"kelompok": g, "n_fitur": len(members), "fitur": ", ".join(members),
                       "auc_tanpa_kelompok": r["mean_auc"], "kontribusi_auc": final_eval["mean_auc"] - r["mean_auc"]})
ablation_df = pd.DataFrame(group_rows).sort_values("kontribusi_auc", ascending=False)
ablation_df.to_csv(os.path.join(OUTPUT_DIR, "ablation_by_group_v4.csv"), index=False)
print("\n=== Ablation: seberapa besar AUC turun kalau 1 kelompok data dibuang ===")
print(ablation_df[["kelompok", "n_fitur", "auc_tanpa_kelompok", "kontribusi_auc"]].round(3).to_string(index=False))
print("(kontribusi_auc positif = kelompok itu membantu; ~0 atau negatif = tidak menambah nilai)")

oot = final_eval["preds"]
oot_tbl = threshold_table(oot["y"], oot["p"])
oot_tbl.to_csv(os.path.join(OUTPUT_DIR, "threshold_selection_oot_v4.csv"), index=False)
print(f"\n=== Threshold dari pooled prediksi OOT ({len(oot)} alert, bulan {sorted(oot['month'].unique())}) ===")
print(oot_tbl.round(3).to_string(index=False))

chosen_thr, thr_rule = FIXED_THRESHOLD, "fixed"
if THRESHOLD_STRATEGY == "target_precision":
    ok = oot_tbl[(oot_tbl["precision"] >= TARGET_PRECISION) & (oot_tbl["n_alert_ke_flag"] >= MIN_FLAGGED_FOR_THRESHOLD)]
    if len(ok):
        chosen_thr = float(ok.sort_values("threshold").iloc[0]["threshold"])
        thr_rule = f"threshold terendah dgn precision OOT >= {TARGET_PRECISION} (min {MIN_FLAGGED_FOR_THRESHOLD} alert ter-flag)"
    else:
        THRESHOLD_STRATEGY = "fbeta"
        print(f">> Target precision {TARGET_PRECISION} tidak tercapai di OOT pada threshold manapun -> fallback ke F{FBETA}. "
              f"Artinya dgn fitur sekarang precision setinggi itu belum realistis; turunkan target atau cari sinyal baru.")
if THRESHOLD_STRATEGY == "fbeta":
    row = oot_tbl.sort_values("f_beta", ascending=False).iloc[0]
    chosen_thr, thr_rule = float(row["threshold"]), f"maksimum F{FBETA} di OOT"
print(f"\n>> THRESHOLD TERPILIH: {chosen_thr:.2f}  ({thr_rule})")

# %% ------------------------------------------------------------------
# 9. PSI (DRIFT) + EVALUASI TEST -- DIINTIP SEKALI
# ------------------------------------------------------------------
psi_rows = [{"fitur": f, "psi": psi(df_pre[f], test_df[f])} for f in final_features]
psi_df = pd.DataFrame(psi_rows).sort_values("psi", ascending=False)
psi_df.to_csv(os.path.join(OUTPUT_DIR, "psi_train_vs_test_v4.csv"), index=False)
print("\n=== PSI fitur (pre-test vs test): >0.25 = shift besar, 0.10-0.25 = sedang ===")
print(psi_df[psi_df["psi"] >= 0.10].round(3).to_string(index=False) if (psi_df["psi"] >= 0.10).any() else "  tidak ada fitur dengan PSI >= 0.10")

test_model = fit_model(best_params, df_pre[final_features], df_pre["target"])
p_test = test_model.predict_proba(test_df[final_features])[:, 1]
y_test = test_df["target"].values

test_auc = roc_auc_score(y_test, p_test)
test_pr = average_precision_score(y_test, p_test)
auc_lo, auc_hi = bootstrap_ci(y_test, p_test, roc_auc_score)
pred_t = (p_test >= chosen_thr).astype(int)
test_prec = precision_score(y_test, pred_t, zero_division=0)
test_rec = recall_score(y_test, pred_t, zero_division=0)
prec_lo, prec_hi = bootstrap_ci(y_test, p_test, lambda yy, pp: precision_score(yy, (pp >= chosen_thr).astype(int), zero_division=0))
rec_lo, rec_hi = bootstrap_ci(y_test, p_test, lambda yy, pp: recall_score(yy, (pp >= chosen_thr).astype(int), zero_division=0))

print("\n" + "!" * 72)
print("HASIL TEST -- DIINTIP SEKALI. JANGAN ubah fitur/parameter/threshold berdasar angka ini.")
print("!" * 72)
print(f"n test = {len(y_test)} | base rate suspicious = {y_test.mean():.3f}")
print(f"ROC-AUC : {test_auc:.3f}  (95% CI {auc_lo:.3f} - {auc_hi:.3f})")
print(f"PR-AUC  : {test_pr:.3f}")
print(f"Pada threshold {chosen_thr:.2f}: di-flag {int(pred_t.sum())} dari {len(y_test)} alert "
      f"({pred_t.mean():.1%}) | precision {test_prec:.3f} (CI {prec_lo:.3f}-{prec_hi:.3f}) | "
      f"recall {test_rec:.3f} (CI {rec_lo:.3f}-{rec_hi:.3f})")
print(f"Rata-rata AUC OOT sebelum test: {final_eval['mean_auc']:.3f} -> "
      f"{'konsisten' if abs(test_auc - final_eval['mean_auc']) < 0.06 else 'selisih cukup besar; baca CI, sample test kecil'}")

unseen = ~test_df[ACCOUNT_COL].isin(seen_accounts).values
if unseen.sum() >= 40 and len(np.unique(y_test[unseen])) == 2:
    print(f"AUC khusus AKUN BARU (belum pernah di train): {roc_auc_score(y_test[unseen], p_test[unseen]):.3f} (n={int(unseen.sum())})")

lift_df = gains_table(y_test, p_test)
lift_df.to_csv(os.path.join(OUTPUT_DIR, "lift_table_test_v4.csv"), index=False)
print("\n=== Lift / gains table (test, decile skor tertinggi -> terendah) ===")
print(lift_df.round(3).to_string(index=False))
print("(mean_score vs actual_rate menunjukkan kalibrasi; skor dgn scale_pos_weight bagus buat RANKING, bukan probabilitas literal)")

test_tbl = threshold_table(y_test, p_test)
test_tbl.to_csv(os.path.join(OUTPUT_DIR, "threshold_table_test_v4.csv"), index=False)
print("\n=== Tabel threshold di test (INFORMASI SAJA, jangan dipakai memilih ulang threshold) ===")
print(test_tbl.round(3).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
RocCurveDisplay.from_predictions(y_test, p_test, ax=ax[0])
PrecisionRecallDisplay.from_predictions(y_test, p_test, ax=ax[1])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_eval_curves_v4.png"), dpi=150)
plt.close()

# %% ------------------------------------------------------------------
# 10. MODEL PRODUKSI (semua data) + SCORING + REASON CODES + MODEL CARD
# ------------------------------------------------------------------
X_all = df[final_features]
production_model = fit_model(best_params, X_all, df["target"])
score_all = production_model.predict_proba(X_all)[:, 1]

try:
    contribs = production_model.get_booster().predict(xgb.DMatrix(X_all), pred_contribs=True)
    reasons = top_reasons_from_contribs(contribs, list(X_all.columns), X_all)
except Exception as e:
    print(f"\n(reason codes dilewati: {type(e).__name__}: {e})")
    reasons = [""] * len(X_all)

oot_scores = pd.concat([oot[[KEY_COL, "p"]], pd.DataFrame({KEY_COL: test_df[KEY_COL].values, "p": p_test})]).drop_duplicates(KEY_COL)
scored = df[[KEY_COL, ACCOUNT_COL, TIME_COL, TARGET_COL]].copy()
scored["score_suspicious"] = score_all
scored["score_oot"] = scored[KEY_COL].map(oot_scores.set_index(KEY_COL)["p"])
scored["rank_pct"] = scored["score_suspicious"].rank(pct=True, ascending=False)
scored["flag_recommended"] = np.where(scored["score_suspicious"] >= chosen_thr, "Suspicious", "Not Suspicious")
scored["top_reasons"] = reasons
scored = scored.sort_values("score_suspicious", ascending=False)
scored.to_csv(os.path.join(OUTPUT_DIR, "scored_all_key1_v4.csv"), index=False)

try:
    production_model.save_model(os.path.join(OUTPUT_DIR, "xgb_production_v4.json"))
except Exception as e:
    print(f"(save_model dilewati: {type(e).__name__})")

def _js(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    return str(o)

model_card = {
    "dibuat": datetime.datetime.now().isoformat(timespec="seconds"),
    "data": {"n_baris": int(len(df)), "bulan": all_months, "base_rate": float(df["target"].mean()),
             "train": train_months, "validate": validate_months, "test": test_months},
    "fitur_final": final_features,
    "params": best_params,
    "threshold": {"nilai": chosen_thr, "aturan": thr_rule},
    "metrik": {"auc_oot_rata2": final_eval["mean_auc"], "auc_oot_std": final_eval["std_auc"],
               "test_auc": test_auc, "test_auc_ci95": [auc_lo, auc_hi], "test_pr_auc": test_pr,
               "test_precision": test_prec, "test_recall": test_rec},
    "toggle": {"USE_DPN_MATCH_SCORE": USE_DPN_MATCH_SCORE, "USE_SCREENING_MATCH_SCORE": USE_SCREENING_MATCH_SCORE},
    "keterbatasan": [
        "Label = keputusan disposisi analis (bukan fraud terkonfirmasi): model meniru penilaian analis, termasuk bias-nya.",
        "Sample kecil (ratusan baris): CI lebar, hindari klaim presisi berlebihan.",
        "Skor bagus untuk ranking; bukan probabilitas terkalibrasi (scale_pos_weight).",
        "Skor historis di scored_all_key1 bersifat in-sample; gunakan score_oot untuk evaluasi jujur.",
    ],
}
with open(os.path.join(OUTPUT_DIR, "model_card_v4.json"), "w", encoding="utf-8") as fh:
    json.dump(model_card, fh, indent=2, ensure_ascii=False, default=_js)

print("\n" + "=" * 72)
print("RINGKASAN UNTUK PRESENTASI")
print("=" * 72)
print(f"Fitur       : {len(feature_pool)} kandidat -> {len(feature_cols_clean)} (redundansi) -> {len(final_features)} final")
print(f"Model       : XGBoost {best_params}")
print(f"OOT rolling : AUC {final_eval['mean_auc']:.3f} +/- {final_eval['std_auc']:.3f} ({len(final_eval['fold_aucs'])} fold)")
print(f"Test (1x)   : AUC {test_auc:.3f} (CI95 {auc_lo:.3f}-{auc_hi:.3f}) | precision {test_prec:.3f} | recall {test_rec:.3f} @ threshold {chosen_thr:.2f}")
print(f"Output      : {OUTPUT_DIR}")
