"""
Tool untuk mengecek apakah suatu pertanyaan/topik ADA jawabannya di
dalam folder dokumen (PDF & Markdown). AI akan memberi jawaban utuh
DAN wajib menyebutkan nama dokumen sumber untuk setiap informasi yang
disampaikan (bisa lebih dari satu dokumen), supaya bisa di-cross-check
manual.

Cara kerja (3 tahap, didesain untuk minim halu & hemat memori):
1. TF-IDF (scikit-learn) — tanpa model AI, cari kandidat chunk yang
   kata/istilahnya mirip pertanyaan ASLI + beberapa VARIASI pertanyaan
   (dibuat oleh LLM) yang maksudnya sama tapi pakai istilah berbeda.
   Ini supaya dokumen yang membahas hal sama dengan kata berbeda tetap
   ketemu. Kalau tidak ada kecocokan sama sekali (bahkan dengan
   variasi), langsung dijawab "TIDAK DITEMUKAN" tanpa panggil LLM lagi.
2. Untuk setiap file kandidat, LLM diminta jawab HANYA "YA"/"TIDAK"
   (temperature=0, deterministik) apakah potongan teks itu benar-benar
   relevan. Ini menyaring kandidat TF-IDF yang cuma kebetulan mirip
   kata tapi tidak relevan.
3. Untuk file yang lolos verifikasi, LLM menyusun jawaban akhir dari
   potongan-potongan itu, dengan instruksi ketat: hanya pakai info
   dari konteks, dan WAJIB mencantumkan nama dokumen sumber di setiap
   informasi yang disampaikan.

Batasan yang perlu disadari: pencarian tetap berbasis kecocokan kata
(langsung maupun lewat variasi), bukan pemahaman makna murni seperti
model embedding. Sistem ini didesain untuk seakurat mungkin, tapi
tidak ada sistem retrieval yang bisa dijamin 100% sempurna.

Tidak memakai model embedding neural (sentence-transformers/torch)
karena package besar seperti itu sering kena quarantine oleh firewall
keamanan internal (Nexus).

Didesain untuk environment terbatas (CML/CDSW):
- Setiap file yang gagal dibaca (rusak/format aneh) hanya dilewati,
  tidak menghentikan seluruh proses build.
- Hanya 1 chunk terbaik per file yang dikirim ke LLM (bukan semua
  chunk), membatasi jumlah panggilan API dan penggunaan memori.
- Tidak ada dependency besar/berat — hanya scikit-learn, pypdf,
  pyyaml, openai.

Cara pakai (dari terminal CML / Jupyter):
    python rag.py build                     # membangun / memperbarui index
    python rag.py check "pertanyaan kamu"    # cek apakah ada di dokumen

Sebelum dijalankan, set environment variable untuk API key (JANGAN
ditulis langsung di config.yaml atau di-commit ke git):
    export QWEN_API_KEY="isi_api_key_kamu"
(di CML: Project Settings > Advanced > Environment Variables)
"""

import os
import sys
import pickle
import argparse
from pathlib import Path

import yaml
import pdfplumber
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


class TableAwarePDFLoader:
    """Loader PDF pengganti PyPDFLoader.

    PyPDFLoader (pypdf) mengekstrak teks murni berdasarkan urutan
    stream PDF, BUKAN posisi visual. Untuk PDF yang isinya tabel
    2 kolom (seperti dokumen RIPLAY: "Fitur | Nilai"), ini bikin
    teks dari kolom kiri dan kanan ke-interleave/ketuker jadi satu
    baris yang salah nyambung.

    Loader ini, per halaman:
    1. Mendeteksi tabel lewat garis/struktur tabel (page.find_tables()).
    2. Mengekstrak tabel tsb sebagai grid sel yang benar, lalu
       menuliskannya ulang sebagai tabel Markdown (baris & kolom
       tetap sejajar dengan pasangannya).
    3. Mengekstrak teks non-tabel (paragraf biasa di luar area
       tabel) secara terpisah, direkonstruksi per baris berdasarkan
       posisi (top, x0) supaya urutan bacanya tetap benar.
    4. Menggabungkan: teks non-tabel dahulu, lalu tabel markdown.

    Dipakai dengan API yang sama seperti loader langchain lain:
    TableAwarePDFLoader(path).load() -> list[Document]
    """

    def __init__(self, file_path):
        self.file_path = str(file_path)

    def load(self):
        documents = []
        with pdfplumber.open(self.file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                content = self._extract_page_content(page)
                if content.strip():
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={"source": self.file_path, "page": page_num},
                        )
                    )
        return documents

    def _extract_page_content(self, page):
        tables = page.find_tables()
        table_bboxes = [t.bbox for t in tables]

        def in_table(word):
            x0, top, x1, bottom = word["x0"], word["top"], word["x1"], word["bottom"]
            for (tx0, ttop, tx1, tbottom) in table_bboxes:
                if x0 >= tx0 - 2 and x1 <= tx1 + 2 and top >= ttop - 2 and bottom <= tbottom + 2:
                    return True
            return False

        # Teks di luar area tabel, direkonstruksi urut baris berdasarkan posisi
        words = [w for w in page.extract_words() if not in_table(w)]
        non_table_text = self._words_to_lines(words)

        # Tabel jadi Markdown, kolom kiri-kanan tetap pasangannya
        table_md_blocks = []
        for t in tables:
            rows = t.extract()
            if rows:
                md = self._table_to_markdown(rows)
                if md:
                    table_md_blocks.append(md)

        parts = []
        if non_table_text.strip():
            parts.append(non_table_text.strip())
        parts.extend(table_md_blocks)
        return "\n\n".join(parts)

    @staticmethod
    def _words_to_lines(words, line_tolerance=3):
        if not words:
            return ""
        words_sorted = sorted(words, key=lambda w: (w["top"], w["x0"]))
        lines, current, current_top = [], [words_sorted[0]], words_sorted[0]["top"]
        for w in words_sorted[1:]:
            if abs(w["top"] - current_top) <= line_tolerance:
                current.append(w)
            else:
                lines.append(current)
                current, current_top = [w], w["top"]
        lines.append(current)
        return "\n".join(
            " ".join(w["text"] for w in sorted(line, key=lambda w: w["x0"]))
            for line in lines
        )

    @staticmethod
    def _table_to_markdown(rows):
        cleaned = [
            [(cell or "").strip().replace("\n", " ") for cell in row]
            for row in rows
        ]
        cleaned = [r for r in cleaned if any(c for c in r)]
        if not cleaned:
            return ""
        n_cols = max(len(r) for r in cleaned)
        header = cleaned[0] + [""] * (n_cols - len(cleaned[0]))
        lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * n_cols) + "|"]
        for row in cleaned[1:]:
            row = row + [""] * (n_cols - len(row))
            lines.append("| " + " | ".join(row[:n_cols]) + " |")
        return "\n".join(lines)

REQUIRED_CONFIG_KEYS = [
    "llm_config",
    "vectorstore_config",
    "chunking_config",
    "retrieval_config",
    "document_folders",
]

INDEX_FILENAME = "tfidf_index.pkl"


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise ValueError(
            f"config.yaml tidak lengkap, key berikut belum ada: {', '.join(missing)}"
        )

    def resolve(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_name = value[2:-1]
            resolved = os.getenv(env_name)
            if resolved is None:
                raise EnvironmentError(
                    f"Environment variable '{env_name}' belum di-set. "
                    f"Jalankan: export {env_name}=<api_key_kamu>"
                )
            return resolved
        return value

    cfg["llm_config"]["api_key"] = resolve(cfg["llm_config"]["api_key"])
    return cfg


def iter_source_files(folder_path):
    """Menghasilkan (file_path, loader_class) satu per satu secara
    rekursif untuk semua .pdf dan .md di dalam folder."""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"[WARN] Folder tidak ditemukan, dilewati: {folder_path}")
        return

    for file_path in sorted(folder.rglob("*.pdf")):
        yield file_path, TableAwarePDFLoader
    for file_path in sorted(folder.rglob("*.md")):
        yield file_path, TextLoader


def load_and_split_file(file_path, loader_cls, splitter, knowledge_base_name):
    """Memuat satu file, memberi label sumbernya, lalu langsung
    dipecah jadi chunk. File yang gagal dibaca dilewati saja."""
    try:
        docs = loader_cls(str(file_path)).load()
    except Exception as e:
        print(f"[WARN] Gagal memuat {file_path.name}: {e}")
        return []

    for d in docs:
        d.metadata["knowledge_base"] = knowledge_base_name
        d.metadata["source"] = str(file_path)

    return splitter.split_documents(docs)


def build_index(cfg):
    folders = cfg["document_folders"]
    if isinstance(folders, str):
        folders = [folders]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunking_config"]["chunk_size"],
        chunk_overlap=cfg["chunking_config"]["chunk_overlap"],
    )

    chunk_texts = []
    chunk_metadatas = []
    total_files = 0
    skipped_files = 0

    for folder_path in folders:
        knowledge_base_name = Path(folder_path).name
        print(f"\nMemproses folder: {folder_path}")

        for file_path, loader_cls in iter_source_files(folder_path):
            chunks = load_and_split_file(file_path, loader_cls, splitter, knowledge_base_name)

            if not chunks:
                skipped_files += 1
                continue

            for c in chunks:
                chunk_texts.append(c.page_content)
                chunk_metadatas.append(c.metadata)

            total_files += 1
            print(f"  - {file_path.name}: {len(chunks)} chunk (total chunk: {len(chunk_texts)})")

    if not chunk_texts:
        print("\n[WARN] Tidak ada dokumen yang berhasil diindeks. Index tidak dibuat.")
        return None

    print(f"\nMembangun index TF-IDF dari {len(chunk_texts)} chunk...")
    vectorizer = TfidfVectorizer(max_features=50000)
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)

    persist_dir = Path(cfg["vectorstore_config"]["persist_directory"])
    persist_dir.mkdir(parents=True, exist_ok=True)
    index_path = persist_dir / INDEX_FILENAME

    with open(index_path, "wb") as f:
        pickle.dump(
            {
                "vectorizer": vectorizer,
                "tfidf_matrix": tfidf_matrix,
                "chunk_texts": chunk_texts,
                "chunk_metadatas": chunk_metadatas,
            },
            f,
        )

    print(f"\nSelesai. {total_files} file berhasil diindeks, {skipped_files} file dilewati.")
    print(f"Total chunk: {len(chunk_texts)}")
    print(f"Index tersimpan di: {index_path}")


def load_index(cfg):
    persist_dir = Path(cfg["vectorstore_config"]["persist_directory"])
    index_path = persist_dir / INDEX_FILENAME

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index belum ada di '{index_path}'. Jalankan 'python rag.py build' dahulu."
        )

    with open(index_path, "rb") as f:
        return pickle.load(f)


def retrieve(index, question, top_k):
    vectorizer = index["vectorizer"]
    tfidf_matrix = index["tfidf_matrix"]
    chunk_texts = index["chunk_texts"]
    chunk_metadatas = index["chunk_metadatas"]

    question_vec = vectorizer.transform([question])
    scores = cosine_similarity(question_vec, tfidf_matrix)[0]

    ranked_idx = scores.argsort()[::-1][:top_k]
    results = []
    for idx in ranked_idx:
        if scores[idx] <= 0:
            continue
        results.append((chunk_texts[idx], chunk_metadatas[idx], float(scores[idx])))
    return results


def group_best_chunk_per_file(results):
    """Dari gabungan beberapa hasil retrieve (bisa dari beberapa variasi
    pertanyaan), ambil HANYA 1 chunk dengan skor TERTINGGI per file.
    Ini membatasi jumlah panggilan LLM verifikasi seminimal mungkin
    (1 file = 1 panggilan), sekaligus menjaga penggunaan memori tetap
    kecil karena tidak menumpuk banyak chunk per file di memori."""
    best_per_file = {}
    for text, meta, score in results:
        source = meta.get("source", "unknown")
        if source not in best_per_file or score > best_per_file[source][2]:
            best_per_file[source] = (text, meta, score)
    return best_per_file


def generate_query_variants(cfg, client, question, n=3):
    """Minta LLM membuat beberapa variasi kalimat pertanyaan dengan
    MAKSUD YANG SAMA tapi memakai kata/istilah berbeda. Ini supaya
    pencarian TF-IDF (yang murni berbasis kecocokan kata) tetap bisa
    menemukan dokumen yang membahas hal sama walau istilahnya beda.
    Kalau langkah ini gagal (misal error koneksi), tidak masalah —
    pencarian tetap lanjut hanya dengan pertanyaan asli."""
    prompt = f"""Buat {n} variasi kalimat pertanyaan dengan MAKSUD YANG SAMA seperti
pertanyaan di bawah, tapi memakai kata/istilah alternatif atau sinonim yang
mungkin dipakai di dokumen resmi/formal. JANGAN menjawab pertanyaannya.
Tulis HANYA variasi kalimatnya, satu per baris, tanpa penomoran dan tanpa
penjelasan apapun.

Pertanyaan asli: {question}"""

    try:
        response = client.chat.completions.create(
            model=cfg["llm_config"]["model"],
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        lines = response.choices[0].message.content.strip().split("\n")
        variants = [line.strip("- ").strip() for line in lines if line.strip()]
        return variants[:n]
    except Exception as e:
        print(f"[WARN] Gagal membuat variasi pertanyaan: {e.__class__.__name__}. Lanjut tanpa variasi.")
        return []


def verify_relevance(cfg, client, question, chunk_text):
    """Minta LLM menjawab HANYA 'YA' atau 'TIDAK' apakah potongan teks
    ini relevan untuk pertanyaan. Ini jauh mengurangi risiko halu
    dibanding meminta LLM menyusun jawaban bebas, karena tugasnya
    dibatasi jadi klasifikasi biner sederhana."""
    verify_prompt = f"""Kamu adalah pemeriksa relevansi dokumen yang SANGAT ketat dan literal.
Tugasmu HANYA menilai apakah POTONGAN TEKS di bawah ini benar-benar membahas
atau menjawab PERTANYAAN yang diberikan. Jangan menyimpulkan, jangan menambah
informasi apapun di luar apa yang tertulis di potongan teks.

Jawab HANYA dengan satu kata: "YA" jika potongan teks tersebut relevan dan
memuat informasi terkait pertanyaan, atau "TIDAK" jika tidak relevan.
Jangan menjawab apapun selain "YA" atau "TIDAK".

Potongan teks:
\"\"\"
{chunk_text}
\"\"\"

Pertanyaan: {question}

Jawaban (YA/TIDAK):"""

    response = client.chat.completions.create(
        model=cfg["llm_config"]["model"],
        temperature=0,  # deterministik, minimalkan variasi/halu
        messages=[{"role": "user", "content": verify_prompt}],
    )
    answer = response.choices[0].message.content.strip().upper()
    return answer.startswith("YA")


def generate_final_answer(cfg, client, question, confirmed_chunks):
    """Minta LLM menyusun jawaban dari potongan-potongan yang sudah
    terverifikasi relevan. Setiap potongan diberi label nama dokumen
    asalnya, dan model DIWAJIBKAN mencantumkan nama dokumen itu di
    jawabannya untuk setiap informasi yang disampaikan — supaya kamu
    bisa langsung cross-check ke file aslinya."""
    context = "\n\n---\n\n".join(
        f"[Dokumen: {c['file']}]\n{c['text']}" for c in confirmed_chunks
    )

    prompt = f"""Kamu adalah asisten yang HANYA menjawab berdasarkan KONTEKS di bawah ini.
Setiap potongan konteks sudah diberi label nama dokumen asalnya dalam format [Dokumen: nama_file].

Aturan ketat yang WAJIB diikuti:
1. Jawab pertanyaan HANYA menggunakan informasi yang ada di konteks. Jangan mengarang
   atau menambahkan informasi apapun di luar konteks yang diberikan.
2. WAJIB sebutkan nama dokumen sumber untuk setiap informasi yang kamu sampaikan,
   dengan format: (Sumber: nama_file). Kalau satu jawaban diambil dari beberapa
   dokumen, sebutkan semua nama dokumennya.
3. Jika konteks yang ada ternyata tidak cukup untuk menjawab pertanyaan secara utuh,
   katakan dengan jelas bagian mana yang tidak bisa dijawab, jangan ditutupi atau ditebak.

Konteks:
{context}

Pertanyaan: {question}

Jawaban (WAJIB sertakan nama dokumen sumber di setiap informasi):"""

    response = client.chat.completions.create(
        model=cfg["llm_config"]["model"],
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def check_question(cfg, index, question):
    """Cek apakah pertanyaan ada jawabannya di dokumen — termasuk kalau
    dokumen membahas hal yang MAKSUDNYA SAMA walau kata-katanya beda
    (lewat variasi pertanyaan). Mengembalikan (confirmed, answer):
    - confirmed: daftar file yang terverifikasi relevan (bisa kosong)
    - answer: jawaban dari LLM dengan sitasi nama dokumen, atau None
      kalau tidak ada dokumen yang relevan sama sekali."""
    candidate_top_k = cfg["retrieval_config"].get("candidate_top_k", 15)
    paraphrase_count = cfg["retrieval_config"].get("paraphrase_count", 3)

    client = OpenAI(
        base_url=cfg["llm_config"]["base_url"],
        api_key=cfg["llm_config"]["api_key"],
    )

    # Kumpulkan kandidat dari pertanyaan asli + beberapa variasi makna serupa
    all_results = list(retrieve(index, question, candidate_top_k))

    if paraphrase_count > 0:
        variants = generate_query_variants(cfg, client, question, n=paraphrase_count)
        for variant in variants:
            all_results.extend(retrieve(index, variant, candidate_top_k))

    if not all_results:
        # Tidak ada kecocokan kata sama sekali (bahkan dengan variasi) -> tidak ada.
        return [], None

    best_per_file = group_best_chunk_per_file(all_results)

    confirmed = []
    for source, (text, meta, score) in best_per_file.items():
        try:
            is_relevant = verify_relevance(cfg, client, question, text)
        except Exception as e:
            print(f"[WARN] Gagal verifikasi {Path(source).name}: {e.__class__.__name__}")
            continue

        if is_relevant:
            confirmed.append(
                {
                    "file": Path(source).name,
                    "knowledge_base": meta.get("knowledge_base", "unknown"),
                    "tfidf_score": round(score, 4),
                    "text": text,
                }
            )

    if not confirmed:
        return [], None

    try:
        answer = generate_final_answer(cfg, client, question, confirmed)
    except Exception as e:
        answer = f"[ERROR] Gagal menyusun jawaban: {e.__class__.__name__}. Tapi dokumen relevan tetap ditemukan (lihat daftar file di bawah)."

    return confirmed, answer


def main():
    parser = argparse.ArgumentParser(
        description="Cek apakah pertanyaan ada jawabannya di folder dokumen (PDF & Markdown)"
    )
    parser.add_argument("command", choices=["build", "check"])
    parser.add_argument("question", nargs="?", help="Pertanyaan (untuk command 'check')")
    parser.add_argument("--config", default="config.yaml", help="Path ke file config")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except (ValueError, EnvironmentError, FileNotFoundError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if args.command == "build":
        build_index(cfg)

    elif args.command == "check":
        if not args.question:
            print('Gunakan: python rag.py check "pertanyaan kamu"')
            sys.exit(1)

        index = load_index(cfg)
        confirmed, answer = check_question(cfg, index, args.question)

        print(f"\nPertanyaan: {args.question}")
        if not confirmed:
            print("Hasil: TIDAK DITEMUKAN dalam dokumen.")
        else:
            print(f"Hasil: DITEMUKAN di {len(confirmed)} file.\n")
            print("=== JAWABAN ===")
            print(answer)
            print("\n=== DAFTAR FILE TERVERIFIKASI (untuk cross-check manual) ===")
            for c in confirmed:
                print(f"  - {c['file']}  (knowledge base: {c['knowledge_base']}, skor TF-IDF: {c['tfidf_score']})")


if __name__ == "__main__":
    main()
