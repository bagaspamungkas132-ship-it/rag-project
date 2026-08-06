# RAG QA — Cek Jawaban di Folder Dokumen (PDF & Markdown)

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

- `document_folder`: path folder yang berisi PDF/Markdown yang ingin dicek.
- `base_url` & `model`: sesuaikan dengan endpoint Qwen kamu.
- `top_k`, `chunk_size`, dll bisa disesuaikan sesuai kebutuhan.

## 3. Build index

Jalankan ini setiap kali dokumen di folder berubah (baru/update):

```bash
python rag.py build
```

Ini akan membaca semua PDF & MD, memecahnya jadi potongan (chunk), lalu menyimpan embedding-nya ke `./chroma_db`.

## 4. Tanya jawab

Sekali tanya:
```bash
python rag.py ask "Apakah ada aturan promo cashback untuk kartu kredit?"
```

Mode interaktif:
```bash
python rag.py chat
```

## Catatan

- Kalau folder dokumen besar, proses `build` bisa makan waktu & memory — jalankan sekali di awal, tidak perlu diulang tiap tanya.
- Model akan menjawab **"TIDAK DITEMUKAN dalam dokumen"** kalau konteks yang diambil tidak relevan dengan pertanyaan — ini sesuai desain, bukan bug.
- Kalau butuh akurasi lebih tinggi, coba naikkan `top_k` di config.yaml atau perkecil `chunk_size` agar potongan lebih presisi.
- Vector store (`chroma_db`) sebaiknya disimpan di storage yang persisten antar-sesi CML, bukan folder sementara, supaya tidak perlu `build` ulang tiap kali sesi baru.
