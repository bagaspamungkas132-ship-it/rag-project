"""
RAG (Retrieval-Augmented Generation) pipeline untuk mengecek apakah
jawaban dari suatu pertanyaan tersedia di dalam folder dokumen
(PDF & Markdown).

Didesain untuk environment terbatas (CML/CDSW):
- Dokumen diproses satu file pada satu waktu (tidak semua dimuat ke
  memori sekaligus), jadi penggunaan memori tetap rendah walau jumlah
  dokumen banyak.
- Setiap file yang gagal dibaca (rusak/format aneh) hanya dilewati,
  tidak menghentikan seluruh proses build.
- Embedding dijalankan di CPU dengan batch kecil agar tidak membengkak.

Cara pakai (dari terminal CML / Jupyter):
    python rag.py build                    # membangun / memperbarui index
    python rag.py ask "pertanyaan kamu"     # tanya sekali
    python rag.py chat                      # mode tanya-jawab interaktif

Sebelum dijalankan, set environment variable untuk API key (JANGAN
ditulis langsung di config.yaml atau di-commit ke git):
    export QWEN_API_KEY="isi_api_key_kamu"
(di CML: Project Settings > Advanced > Environment Variables)
"""

import os
import sys
import gc
import argparse
from pathlib import Path

import yaml
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

REQUIRED_CONFIG_KEYS = [
    "llm_config",
    "embedding_config",
    "vectorstore_config",
    "chunking_config",
    "retrieval_config",
    "document_folders",
    "system_prompt",
]


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
    rekursif untuk semua .pdf dan .md di dalam folder — tanpa memuat
    isinya, hanya path saja, supaya ringan."""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"[WARN] Folder tidak ditemukan, dilewati: {folder_path}")
        return

    for file_path in sorted(folder.rglob("*.pdf")):
        yield file_path, PyPDFLoader
    for file_path in sorted(folder.rglob("*.md")):
        yield file_path, TextLoader


def load_and_split_file(file_path, loader_cls, splitter, knowledge_base_name):
    """Memuat satu file, memberi label sumbernya, lalu langsung
    dipecah jadi chunk. Kalau file gagal dibaca, dilewati (tidak
    menghentikan proses build secara keseluruhan)."""
    try:
        docs = loader_cls(str(file_path)).load()
    except Exception as e:
        print(f"[WARN] Gagal memuat {file_path.name}: {e}")
        return []

    for d in docs:
        d.metadata["knowledge_base"] = knowledge_base_name
        d.metadata["source"] = str(file_path)

    return splitter.split_documents(docs)


def build_embeddings(cfg):
    batch_size = cfg["embedding_config"].get("batch_size", 32)
    return HuggingFaceEmbeddings(
        model_name=cfg["embedding_config"]["model"],
        model_kwargs={"device": cfg["embedding_config"].get("device", "cpu")},
        encode_kwargs={"batch_size": batch_size},
    )


def build_index(cfg):
    folders = cfg["document_folders"]
    if isinstance(folders, str):
        folders = [folders]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunking_config"]["chunk_size"],
        chunk_overlap=cfg["chunking_config"]["chunk_overlap"],
    )
    embeddings = build_embeddings(cfg)

    vectordb = None
    total_files = 0
    total_chunks = 0
    skipped_files = 0

    for folder_path in folders:
        knowledge_base_name = Path(folder_path).name
        print(f"\nMemproses folder: {folder_path}")

        for file_path, loader_cls in iter_source_files(folder_path):
            chunks = load_and_split_file(file_path, loader_cls, splitter, knowledge_base_name)

            if not chunks:
                skipped_files += 1
                continue

            try:
                if vectordb is None:
                    vectordb = FAISS.from_documents(chunks, embeddings)
                else:
                    vectordb.add_documents(chunks)
            except Exception as e:
                print(f"[WARN] Gagal mengindeks {file_path.name}: {e}")
                skipped_files += 1
                continue

            total_files += 1
            total_chunks += len(chunks)
            print(f"  - {file_path.name}: {len(chunks)} chunk (total chunk: {total_chunks})")

            # Bebaskan memori dokumen & chunk file ini sebelum lanjut ke file berikutnya.
            del chunks
            gc.collect()

    if vectordb is None:
        print("\n[WARN] Tidak ada dokumen yang berhasil diindeks. Index tidak dibuat.")
        return None

    persist_dir = cfg["vectorstore_config"]["persist_directory"]
    vectordb.save_local(persist_dir)

    print(f"\nSelesai. {total_files} file berhasil diindeks, {skipped_files} file dilewati.")
    print(f"Total chunk: {total_chunks}")
    print(f"Index tersimpan di: {persist_dir}")
    return vectordb


def load_index(cfg):
    embeddings = build_embeddings(cfg)
    persist_dir = cfg["vectorstore_config"]["persist_directory"]

    if not Path(persist_dir).exists():
        raise FileNotFoundError(
            f"Index belum ada di '{persist_dir}'. Jalankan 'python rag.py build' dahulu."
        )

    # allow_dangerous_deserialization aman di sini karena file index ini
    # HANYA pernah dibuat oleh build_index() di atas (bukan dari sumber
    # eksternal/tidak dikenal). Jangan load index dari sumber yang tidak
    # kamu percaya dengan flag ini.
    return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)


def ask_question(cfg, vectordb, question):
    top_k = cfg["retrieval_config"]["top_k"]
    results = vectordb.similarity_search(question, k=top_k)

    if not results:
        return "TIDAK DITEMUKAN dalam dokumen.", []

    context = "\n\n---\n\n".join(
        f"[Knowledge base: {r.metadata.get('knowledge_base', 'unknown')} | "
        f"Sumber: {Path(r.metadata.get('source', 'unknown')).name}]\n{r.page_content}"
        for r in results
    )

    prompt = f"""{cfg['system_prompt']}

Konteks:
{context}

Pertanyaan: {question}

Jawaban:"""

    try:
        client = OpenAI(
            base_url=cfg["llm_config"]["base_url"],
            api_key=cfg["llm_config"]["api_key"],
        )
        response = client.chat.completions.create(
            model=cfg["llm_config"]["model"],
            temperature=cfg["llm_config"]["temperature"],
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content
    except Exception as e:
        # Tidak menampilkan detail internal (base_url/key) di error,
        # hanya info yang perlu diketahui pengguna.
        return f"[ERROR] Gagal memanggil model LLM: {e.__class__.__name__}. Cek koneksi/endpoint.", []

    sources = sorted(
        set(
            f"[{r.metadata.get('knowledge_base', 'unknown')}] "
            f"{Path(r.metadata.get('source', 'unknown')).name}"
            for r in results
        )
    )
    return answer, sources


def main():
    parser = argparse.ArgumentParser(
        description="RAG QA atas folder dokumen (PDF & Markdown)"
    )
    parser.add_argument("command", choices=["build", "ask", "chat"])
    parser.add_argument("question", nargs="?", help="Pertanyaan (untuk command 'ask')")
    parser.add_argument("--config", default="config.yaml", help="Path ke file config")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except (ValueError, EnvironmentError, FileNotFoundError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if args.command == "build":
        build_index(cfg)

    elif args.command == "ask":
        if not args.question:
            print('Gunakan: python rag.py ask "pertanyaan kamu"')
            sys.exit(1)
        vectordb = load_index(cfg)
        answer, sources = ask_question(cfg, vectordb, args.question)
        print("\n=== JAWABAN ===")
        print(answer)
        if sources:
            print("\n=== SUMBER ===")
            for s in sources:
                print(f"- {s}")

    elif args.command == "chat":
        vectordb = load_index(cfg)
        print("Mode chat aktif. Ketik 'exit' untuk keluar.\n")
        while True:
            question = input("Pertanyaan: ").strip()
            if question.lower() in ("exit", "quit"):
                break
            if not question:
                continue
            answer, sources = ask_question(cfg, vectordb, question)
            print(f"\nJawaban: {answer}")
            if sources:
                print(f"Sumber: {', '.join(sources)}")
            print()


if __name__ == "__main__":
    main()
