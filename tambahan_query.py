def expand_query_with_synonyms(question, term_synonyms):
    """Cek apakah pertanyaan mengandung istilah bisnis yang ada di kamus
    sinonim (term_synonyms di config.yaml). Kalau ada, buat 1 variasi
    tambahan berupa pertanyaan asli + kata-kata alternatifnya, supaya
    TF-IDF (yang murni cocokkan kata) tetap bisa menemukan dokumen yang
    memakai istilah internal berbeda — misal "OCBC Mobile" di pertanyaan
    harus tetap ketemu dokumen yang pakai kata "online" di isinya.

    Ini kamus yang bisa terus kamu tambah sendiri di config.yaml setiap
    kali menemukan istilah yang penyebutannya beda antara pertanyaan
    sehari-hari dan istilah resmi di dokumen."""
    if not term_synonyms:
        return []

    q_lower = question.lower()
    extra_terms = []
    for term, synonyms in term_synonyms.items():
        if term.lower() in q_lower:
            extra_terms.extend(synonyms)

    if not extra_terms:
        return []

    augmented = question + " " + " ".join(extra_terms)
    return [augmented]


part2
candidate_top_k = cfg["retrieval_config"].get("candidate_top_k", 15)
    paraphrase_count = cfg["retrieval_config"].get("paraphrase_count", 3)
    term_synonyms = cfg.get("term_synonyms", {})

    client = build_llm_client(cfg)

    # Kumpulkan kandidat dari pertanyaan asli + beberapa variasi makna serupa
    all_results = list(retrieve(index, question, candidate_top_k))

    # Ekspansi berdasarkan kamus istilah bisnis (misal "OCBC Mobile" -> "online")
    for augmented in expand_query_with_synonyms(question, term_synonyms):
        all_results.extend(retrieve(index, augmented, candidate_top_k))

    if paraphrase_count > 0:
        variants = generate_query_variants(cfg, client, question, n=paraphrase_count)
        for variant in variants:
            all_results.extend(retrieve(index, variant, candidate_top_k))

python3 -c "
import pickle
with open('tfidf_index/tfidf_index.pkl', 'rb') as f:
    data = pickle.load(f)
matches = [m['source'] for m in data['chunk_metadatas'] if 'online' in m['source'].lower()]
print(set(matches))
"
cat > debug_search2.py << 'PYEOF'
import pickle
from sklearn.metrics.pairwise import cosine_similarity

with open('tfidf_index/tfidf_index.pkl', 'rb') as f:
    data = pickle.load(f)

vectorizer = data['vectorizer']
tfidf_matrix = data['tfidf_matrix']
chunk_metadatas = data['chunk_metadatas']
chunk_texts = data['chunk_texts']

query = "Berapa minimum saldo penempatan deposito idr menggunakan OCBC Mobile? online digital"
q_vec = vectorizer.transform([query])
scores = cosine_similarity(q_vec, tfidf_matrix)[0]

ranked = scores.argsort()[::-1]

print("=== Rank untuk semua chunk dari file 'online_umum' (di mana pun posisinya) ===")
for rank, idx in enumerate(ranked, 1):
    source = chunk_metadatas[idx]['source']
    if "online_umum" in source.lower() or "deposito_idr_online" in source.lower():
        print(f"\nRank {rank} | skor={scores[idx]:.4f} | {source}")
        print(f"Isi chunk:\n{chunk_texts[idx][:500]}")
        print("-" * 60)
PYEOF
python3 debug_search2.py
cat > debug_search3.py << 'PYEOF'
import pickle
from sklearn.metrics.pairwise import cosine_similarity

with open('tfidf_index/tfidf_index.pkl', 'rb') as f:
    data = pickle.load(f)

vectorizer = data['vectorizer']
tfidf_matrix = data['tfidf_matrix']
chunk_metadatas = data['chunk_metadatas']
chunk_texts = data['chunk_texts']

print(f"Total chunk di index: {len(chunk_texts)}")

query = "Berapa minimum saldo penempatan deposito idr menggunakan OCBC Mobile? online digital"
q_vec = vectorizer.transform([query])
scores = cosine_similarity(q_vec, tfidf_matrix)[0]

ranked = scores.argsort()[::-1]

target_keyword = "deposito_idr_online"

with open("hasil_debug.txt", "w") as out:
    out.write(f"Total chunk di index: {len(chunk_texts)}\n\n")
    found = 0
    for rank, idx in enumerate(ranked, 1):
        source = chunk_metadatas[idx]['source']
        if target_keyword in source.lower():
            found += 1
            out.write(f"Rank {rank} dari {len(chunk_texts)} | skor={scores[idx]:.4f} | {source}\n")
            out.write(f"Isi chunk:\n{chunk_texts[idx][:600]}\n")
            out.write("-" * 60 + "\n\n")
            if found >= 5:
                break
    if found == 0:
        out.write("TIDAK ADA chunk yang path-nya mengandung 'deposito_idr_online' sama sekali.\n")

print("Selesai, hasil disimpan di hasil_debug.txt")
PYEOF
python3 debug_search3.py
