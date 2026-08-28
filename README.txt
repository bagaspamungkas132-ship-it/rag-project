CARA PAKAI
==========

1. Copy file ".env.example" menjadi ".env"
   (di terminal: cp .env.example .env)

2. Buka file ".env", isi baris LLM_API_KEY dengan API key kamu:
   LLM_API_KEY=isi_key_disini

3. Cek juga baris PARQUET_INPUT_PATH di .env, pastikan sudah sesuai
   lokasi file data.parquet kamu.

4. Install library yang dibutuhkan (sekali saja):
   pip install -r requirements.txt

5. Jalankan scriptnya:
   python deteksi_anomali_llm.py

6. Tunggu sampai selesai (ada progress bar). Hasil akan tersimpan
   otomatis ke file baru sesuai PARQUET_OUTPUT_PATH di .env
   (default: data_with_anomali_flag.parquet).

   File data.parquet ASLI tidak akan diubah/ditimpa sama sekali.

CATATAN
=======
- Kalau nama kolom di parquet kamu berbeda dari:
  customerjob, customerjobsector, customerjoblevel, monthly_income
  ubah bagian COL_JOB / COL_SECTOR / COL_LEVEL / COL_INCOME
  di bagian atas file deteksi_anomali_llm.py

- Kalau sering muncul warning "Gagal parse JSON", turunkan nilai
  BATCH_SIZE di file .env (misal dari 25 jadi 10 atau 15).

- File .env JANGAN dikirim/di-share ke orang lain atau di-commit ke
  git, karena berisi API key kamu.
