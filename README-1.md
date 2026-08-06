# Cek Jawaban di Folder Dokumen (PDF & Markdown)

Tool untuk mengecek apakah suatu pertanyaan/topik ADA jawabannya di
dalam folder dokumen, dan menyebutkan nama dokumen sumbernya (bisa
lebih dari satu file). Didesain ringan (tanpa torch/chromadb) supaya
lolos di environment dengan firewall package yang ketat (Nexus).

## 1. Setup di CML

```bash
pip install -r requirements.txt
```

Set API key sebagai environment variable (JANGAN ditulis langsung di config.yaml):

- Lewat terminal sesi:
  ```bash
  export QWEN_API_KEY="isi_api_key_kamu"
  ```
- Atau lebih permanen: CML Project > **Project Settings** > **Advanced** > **Environment Variables**, tambahkan `QWEN_API_KEY`.

## 2. Sesuaikan config.yaml

- `document_folders`: daftar folder yang berisi PDF/Markdown yang ingin dicek (bisa lebih dari satu, di-scan rekursif termasuk semua subfolder).
- `base_url` & `model`: sesuaikan dengan endpoint Qwen kamu.
- `candidate_top_k`: jumlah kandidat chunk awal dari TF-IDF yang diverifikasi ke LLM (default 15).
- `paraphrase_count`: jumlah variasi kalimat pertanyaan (maksud sama, istilah beda) yang dibuat LLM untuk memperluas pencarian (default 3, set 0 untuk mematikan).

## 3. Build index

Jalankan ini setiap kali dokumen di folder berubah (baru/update):

```bash
python rag.py build
```

Ini membaca semua PDF & MD, memecahnya jadi potongan (chunk), lalu membangun index TF-IDF dan menyimpannya ke `./tfidf_index`.

## 4. Cek pertanyaan

```bash
python rag.py check "apakah ada promo cashback kartu kredit bulan ini?"
```

Cara kerja (3 tahap):
1. **TF-IDF** — cari kandidat chunk yang kata/istilahnya mirip pertanyaan ASLI + beberapa VARIASI pertanyaan (dibuat LLM) yang maksudnya sama tapi istilahnya beda.
2. **Verifikasi YA/TIDAK** — tiap file kandidat diverifikasi LLM (temperature 0) apakah benar-benar relevan, menyaring kecocokan kata yang kebetulan tapi tidak relevan.
3. **Jawaban bersitasi** — untuk file yang lolos, LLM menyusun jawaban HANYA dari konteks tersebut, dan WAJIB menyebutkan nama dokumen sumber di setiap informasi (format `(Sumber: nama_file)`).

Kalau tidak ada dokumen relevan sama sekali, hasilnya **"TIDAK DITEMUKAN dalam dokumen"** — ini sesuai desain (mencegah model mengarang jawaban), bukan bug.

## Catatan

- Tidak pakai model embedding neural (sentence-transformers/torch) karena package besar sering kena quarantine oleh firewall keamanan internal — hanya scikit-learn (TF-IDF), pypdf, pyyaml, openai.
- Setiap file yang gagal dibaca (rusak/format aneh) dilewati saja, tidak menghentikan seluruh proses `build`.
- Index (`tfidf_index/`) sebaiknya disimpan di storage yang persisten antar-sesi CML, supaya tidak perlu `build` ulang tiap sesi baru.
- Batasan: pencarian tetap berbasis kecocokan kata (langsung maupun lewat variasi pertanyaan), bukan pemahaman makna murni seperti model embedding — tidak ada sistem retrieval yang bisa dijamin 100% sempurna.
- **Jangan** hardcode API key di `config.yaml` atau commit ke git — sudah diatasi lewat `${QWEN_API_KEY}` + `.gitignore`.
