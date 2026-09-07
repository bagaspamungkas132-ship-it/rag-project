import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib
matplotlib.use("Agg")  # supaya bisa save ke file tanpa GUI, cocok untuk CML
import matplotlib.pyplot as plt

# =========================================================
# Asumsi: df_fds, feature_cols, X, y sudah ada dari script sebelumnya
# (kalau belum, load ulang dulu bagian LOAD & FILTER FDS + HANDLE NaN)
# =========================================================

# =========================================================
# 1. DECISION TREE UNTUK PROFILING SEGMEN
#    (bukan untuk prediksi akurat, tapi untuk EXPLAIN pola segmen)
# =========================================================
# Depth dibatasi kecil (3-4) supaya hasilnya masih bisa dibaca manusia
tree_profile = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=20,   # tiap segmen minimal 20 baris, biar cukup representatif
    class_weight="balanced",
    random_state=42
)
tree_profile.fit(X, y)

# Cetak aturan pohon dalam bentuk teks (mudah dibaca)
print("=== Struktur Pohon Segmentasi ===")
tree_rules = export_text(tree_profile, feature_names=list(X.columns))
print(tree_rules)

# =========================================================
# 2. HITUNG STATISTIK PER SEGMEN (LEAF NODE)
# =========================================================
leaf_ids = tree_profile.apply(X)
df_fds_seg = df_fds.copy()
df_fds_seg["leaf_id"] = leaf_ids
df_fds_seg["is_suspicious"] = y  # 1 = Suspicious

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
# 3. VISUALISASI POHON (disimpan sebagai gambar)
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
# 4. PROFIL DESKRIPTIF PER SEGMEN TERATAS
#    (biar tahu karakteristik segmen paling berisiko secara konkret)
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