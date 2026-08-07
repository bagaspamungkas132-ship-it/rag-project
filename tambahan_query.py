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
