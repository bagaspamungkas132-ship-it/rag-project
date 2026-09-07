import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

# 1. Load & prep (lanjutan dari sebelumnya)
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
    "dpn_flag", "screening_flag"
]
df = pd.read_parquet(path)[cols].copy()

target = "disposition_class"
feature_cols = [c for c in cols if c not in ["key1", target]]
X = df[feature_cols]
y = LabelEncoder().fit_transform(df[target])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. Bandingkan beberapa model dengan cross-validation (ROC-AUC)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=5, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
}

print("=== Perbandingan Model (5-fold CV, ROC-AUC) ===")
results = {}
for name, model in models.items():
    if name == "Logistic Regression":
        X_use = StandardScaler().fit_transform(X)
    else:
        X_use = X
    scores = cross_val_score(model, X_use, y, cv=cv, scoring="roc_auc")
    results[name] = scores
    print(f"{name}: mean AUC={scores.mean():.4f}, std={scores.std():.4f}")

# 3. Feature importance dari Random Forest (cross-validated permutation importance)
rf = RandomForestClassifier(n_estimators=300, max_depth=5, class_weight="balanced", random_state=42)
rf.fit(X, y)
perm = permutation_importance(rf, X, y, n_repeats=30, random_state=42, scoring="roc_auc")

perm_df = pd.DataFrame({
    "feature": feature_cols,
    "rf_importance": rf.feature_importances_,
    "permutation_importance": perm.importances_mean
}).sort_values("permutation_importance", ascending=False).reset_index(drop=True)

print("\n=== Ranking Variabel Paling Berpengaruh (RF + Permutation) ===")
print(perm_df)

# 4. Koefisien Logistic Regression sebagai pembanding arah pengaruh
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
logreg.fit(X_scaled, y)

coef_df = pd.DataFrame({
    "feature": feature_cols,
    "logreg_coef": logreg.coef_[0]
}).sort_values("logreg_coef", key=abs, ascending=False).reset_index(drop=True)

print("\n=== Ranking Variabel (Logistic Regression coef, |value|) ===")
print(coef_df)