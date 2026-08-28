"""
Deteksi anomali kombinasi pekerjaan-jabatan-income menggunakan LLM (Qwen).

Cara pakai:
1. Copy file .env.example menjadi .env
2. Isi LLM_API_KEY di file .env dengan API key kamu
3. Sesuaikan PARQUET_INPUT_PATH kalau lokasi file parquet berbeda
4. Install dependency: pip install -r requirements.txt
5. Jalankan: python deteksi_anomali_llm.py

Hasil akan disimpan ke file BARU (PARQUET_OUTPUT_PATH) dengan 3 kolom tambahan:
- llm_status   : "WAJAR" / "ANOMALI" / "GAGAL_CEK"
- llm_alasan   : alasan singkat dari LLM
- is_anomali   : True/False

File parquet asli TIDAK disentuh/ditimpa.
"""

import os
import json
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# ----------------------------------------------------------------------
# 1. LOAD KONFIGURASI DARI .env
# ----------------------------------------------------------------------
load_dotenv()

API_KEY = os.environ.get("LLM_API_KEY")
BASE_URL = os.environ.get("LLM_BASE_URL")
MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3.5-35B-A3B")
PATH_INPUT = os.path.expanduser(os.environ.get("PARQUET_INPUT_PATH", ""))
PATH_OUTPUT = os.path.expanduser(os.environ.get("PARQUET_OUTPUT_PATH", ""))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 25))

if not API_KEY or API_KEY == "paste_api_key_kamu_di_sini":
    raise ValueError(
        "LLM_API_KEY belum diisi. Buat file .env (copy dari .env.example) "
        "dan isi API key kamu di sana."
    )
if not PATH_INPUT:
    raise ValueError("PARQUET_INPUT_PATH belum diisi di file .env")

# Kolom-kolom yang dipakai untuk menilai kewajaran.
# Ganti daftar ini kalau nama kolom di parquet kamu berbeda.
COL_JOB = "customerjob"
COL_SECTOR = "customerjobsector"
COL_LEVEL = "customerjoblevel"
COL_INCOME = "monthly_income"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ----------------------------------------------------------------------
# 2. FUNGSI: cek satu batch baris sekaligus dalam 1 API call
# ----------------------------------------------------------------------
def cek_batch_llm(batch_df: pd.DataFrame, max_retry: int = 3) -> list:
    rows_text = ""
    for i, row in batch_df.iterrows():
        income = row.get(COL_INCOME)
        income_str = f"Rp{income:,.0f}" if pd.notna(income) else "KOSONG/NULL"
        rows_text += (
            f"ID:{i} | Pekerjaan:{row.get(COL_JOB)} | "
            f"Sektor:{row.get(COL_SECTOR)} | "
            f"Jabatan:{row.get(COL_LEVEL)} | "
            f"Income:{income_str}\n"
        )

    prompt = f"""Kamu adalah auditor data. Nilai setiap baris berikut, apakah kombinasi pekerjaan-jabatan-income WAJAR atau ANOMALI (tidak masuk akal secara logika bisnis/umum).

Data:
{rows_text}

Jawab HANYA dalam format JSON array, satu objek per baris, seperti ini:
[{{"id": 0, "status": "WAJAR", "alasan": "..."}}, {{"id": 1, "status": "ANOMALI", "alasan": "..."}}]

Jangan tambahkan teks lain di luar JSON."""

    for attempt in range(max_retry):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except json.JSONDecodeError:
            if attempt == max_retry - 1:
                print(f"[WARNING] Gagal parse JSON setelah {max_retry}x percobaan. "
                      f"Batch mulai index {batch_df.index[0]} ditandai GAGAL_CEK.")
                return []
        except Exception as e:
            if attempt == max_retry - 1:
                print(f"[ERROR] API call gagal setelah {max_retry}x percobaan: {e}")
                return []
            time.sleep(2 * (attempt + 1))  # backoff sebelum retry

    return []


# ----------------------------------------------------------------------
# 3. BACA DATA
# ----------------------------------------------------------------------
print(f"Membaca data dari: {PATH_INPUT}")
df = pd.read_parquet(PATH_INPUT)
df = df.replace("[NULL]", pd.NA)
df[COL_INCOME] = pd.to_numeric(df[COL_INCOME], errors="coerce")
df = df.reset_index(drop=True)
print(f"Total baris: {len(df)}")


# ----------------------------------------------------------------------
# 4. JALANKAN PER BATCH
# ----------------------------------------------------------------------
hasil_semua = {}

for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Memproses batch"):
    batch = df.iloc[start:start + BATCH_SIZE]
    hasil_batch = cek_batch_llm(batch)
    for item in hasil_batch:
        try:
            idx = int(item["id"])
            hasil_semua[idx] = {
                "status": item.get("status", "UNKNOWN"),
                "alasan": item.get("alasan", ""),
            }
        except (KeyError, ValueError, TypeError):
            continue


# ----------------------------------------------------------------------
# 5. GABUNGKAN HASIL KE DATAFRAME (kolom baru, tidak menimpa yang lama)
# ----------------------------------------------------------------------
df["llm_status"] = df.index.map(lambda i: hasil_semua.get(i, {}).get("status", "GAGAL_CEK"))
df["llm_alasan"] = df.index.map(lambda i: hasil_semua.get(i, {}).get("alasan", ""))
df["is_anomali"] = df["llm_status"] == "ANOMALI"


# ----------------------------------------------------------------------
# 6. SIMPAN KE FILE BARU (file asli tidak disentuh)
# ----------------------------------------------------------------------
os.makedirs(os.path.dirname(PATH_OUTPUT), exist_ok=True)
df.to_parquet(PATH_OUTPUT, index=False)

total = len(df)
n_anomali = int(df["is_anomali"].sum())
n_gagal = int((df["llm_status"] == "GAGAL_CEK").sum())

print("\n=== SELESAI ===")
print(f"Total baris     : {total}")
print(f"Anomali         : {n_anomali}")
print(f"Wajar           : {total - n_anomali - n_gagal}")
print(f"Gagal dicek     : {n_gagal}")
print(f"Hasil disimpan  : {PATH_OUTPUT}")

if n_gagal > 0:
    print(f"\nCatatan: {n_gagal} baris gagal dicek (kemungkinan LLM mengembalikan "
          f"format tidak valid). Bisa dijalankan ulang khusus untuk baris "
          f"dengan llm_status == 'GAGAL_CEK'.")
