# Price Flex — Copilot Studio Starter Kit

Paket ini berisi semua yang perlu kamu siapkan sebelum masuk ke Microsoft Copilot Studio: data dummy, rate matrix, instruksi agent, skrip percakapan (topic), dan checklist setup step-by-step.

---

## 1. Sample Dummy Customer Data

Upload tabel ini sebagai **Knowledge Source** (bisa dalam bentuk Excel/CSV). Ini yang bikin demo kelihatan "personalized" karena tiap customer dapat rate berbeda.

| Customer ID | Segment | Saldo (Rp) | Tenor Diinginkan | Histori Transaksi | Produk Existing | Engagement Score |
|---|---|---|---|---|---|---|
| C001 | Emerging Affluent | 150.000.000 | 12 bulan | Aktif, transfer rutin bulanan | Tabungan + Kartu Kredit | Tinggi |
| C002 | Emerging Affluent | 45.000.000 | 6 bulan | Baru buka rekening 3 bulan lalu | Tabungan saja | Rendah |
| C003 | Mass Market | 20.000.000 | 3 bulan | Transaksi sporadis | Tabungan saja | Sedang |
| C004 | Emerging Affluent | 300.000.000 | 24 bulan | Sangat aktif, ada histori deposito sebelumnya | Tabungan + Deposito + KPR | Tinggi |
| C005 | Mass Market | 10.000.000 | 1 bulan | Baru pertama kali coba TD | Tabungan saja | Rendah |
| C006 | Emerging Affluent | 80.000.000 | 12 bulan | Aktif, sering pakai e-wallet top up | Tabungan + Kartu Kredit | Sedang |

> Tips: tambahkan 3–5 baris lagi biar demo lebih variatif, terutama beberapa kombinasi "saldo rendah tapi engagement tinggi" dan sebaliknya — ini bagus untuk menunjukkan model tidak cuma lihat saldo saja.

---

## 2. Rate Matrix / Pricing Rules

Ini tabel aturan yang jadi dasar rekomendasi rate. Simpan sebagai Knowledge Source terpisah, atau masukkan ke Power Automate sebagai lookup table kalau mau logic lebih presisi.

| Kondisi | Rate Dasar (Deposito Online) | Adjustment | Rate Final (Range) |
|---|---|---|---|
| Saldo ≥ 200jt, Tenor ≥ 12 bulan, Engagement Tinggi | 5.00% | +0.50% loyalty bonus | 5.25% – 5.50% |
| Saldo 100–200jt, Tenor ≥ 12 bulan | 5.00% | +0.25% | 5.00% – 5.25% |
| Saldo 50–100jt, Tenor 6–12 bulan | 4.75% | +0.15% jika Engagement Tinggi | 4.75% – 4.90% |
| Saldo < 50jt, Tenor < 6 bulan | 4.50% | Tidak ada adjustment | 4.50% |
| Customer baru (< 3 bulan), tanpa histori TD | 4.50% | Tidak eligible untuk promo dinamis | 4.50% (flat) |

**Guardrail (wajib disebutkan ke judges):**
- Rate final tidak pernah melebihi batas atas yang di-set treasury (misal cap 5.50%)
- Customer dengan profil serupa (saldo & tenor mirip) tidak boleh dapat rate yang beda signifikan → hindari kesan diskriminatif
- Kalau data tidak lengkap/eligibility tidak terpenuhi → fallback ke rate standar, bukan reject

---

## 3. Agent Instructions (untuk di-paste ke Copilot Studio)

Saat bikin agent baru, di kolom "Instructions" masukkan ini:

```
Kamu adalah Price Flex Assistant, asisten AI internal OCBC yang membantu
merekomendasikan personalized interest rate untuk produk TAKA Online dan
Deposito Online.

Tugasmu:
1. Tanyakan data customer yang relevan: saldo yang akan didepositokan,
   tenor yang diinginkan, dan status keaktifan transaksi mereka.
2. Cocokkan data tersebut dengan rate matrix yang ada di knowledge source.
3. Berikan rekomendasi rate dalam bentuk range, bukan angka pasti tunggal,
   kecuali data lengkap dan cocok dengan satu baris matrix.
4. Selalu jelaskan secara singkat KENAPA rate tersebut direkomendasikan
   (contoh: "karena saldo Anda di atas 200 juta dengan tenor 12 bulan,
   Anda eligible untuk loyalty bonus 0.5%").
5. Jangan pernah merekomendasikan rate di luar batas yang ada di rate
   matrix, walau user meminta.
6. Jika data customer tidak lengkap, tanyakan dulu sebelum memberi rate.
7. Gunakan nada profesional, ramah, dan ringkas — seperti customer
   service digital bank.
```

---

## 4. Skrip Percakapan (untuk Topic di Copilot Studio)

Gunakan ini sebagai dasar bikin **Topic** dengan trigger phrases dan conversation nodes.

**Topic Name:** `Rekomendasi Rate Personalisasi`

**Trigger phrases** (kata kunci yang memicu topic ini):
- "saya mau buka deposito"
- "berapa rate untuk saya"
- "cek rate personalisasi"
- "mau tau bunga deposito saya"

**Flow percakapan:**

```
Agent : Halo! Saya bisa bantu cek rate personalisasi untuk Deposito
        Online / TAKA Online Anda. Boleh saya tahu perkiraan saldo yang
        ingin Anda depositokan?

User  : [input saldo, misal: 150 juta]

Agent : Baik, dan tenor yang Anda inginkan berapa lama? (misal: 3, 6,
        12, atau 24 bulan)

User  : [input tenor, misal: 12 bulan]

Agent : Terakhir, apakah Anda sudah cukup aktif bertransaksi di rekening
        OCBC selama ini, atau ini rekening yang baru dibuka?

User  : [input status: aktif / baru]

Agent : [Agent mencocokkan ke rate matrix]
        Berdasarkan profil Anda (saldo Rp150.000.000, tenor 12 bulan,
        nasabah aktif), Anda eligible untuk rate personalisasi di
        kisaran 5.00%–5.25% — lebih tinggi dari rate standar karena
        loyalty & tenor Anda.

        Mau saya bantu lanjutkan proses pembukaan Deposito Online?

User  : [ya / tidak]

Agent : [Jika ya] Baik, saya arahkan ke halaman pembukaan Deposito
        Online. [Jika tidak] Baik, kalau ada pertanyaan lain seputar
        rate, silakan tanya kapan saja!
```

**Variasi skenario tambahan (opsional, bikin demo lebih kaya):**
1. Skenario "customer baru tanpa histori" → agent kasih tau kenapa rate-nya flat/standar
2. Skenario "saldo besar tapi tenor pendek" → agent jelaskan trade-off tenor vs rate
3. Skenario "user tanya kenapa rate teman mereka lebih tinggi" → agent jelaskan guardrail fairness tanpa membocorkan data customer lain

---

## 5. Checklist Setup di Copilot Studio (Step-by-Step)

- [ ] **Buat agent baru** → beri nama "Price Flex Assistant", masukkan deskripsi singkat use case
- [ ] **Paste Agent Instructions** dari bagian 3 di atas ke kolom Instructions
- [ ] **Upload Knowledge Source**:
  - File data dummy customer (bagian 1) — format Excel/CSV
  - File rate matrix (bagian 2) — format Excel/CSV atau tabel di Word
- [ ] **Buat Topic** "Rekomendasi Rate Personalisasi" pakai trigger phrases & flow dari bagian 4
- [ ] **Tambahkan Variables** untuk menyimpan input user (saldo, tenor, status aktif) supaya bisa dipakai lintas node dalam topic
- [ ] **(Opsional) Tambahkan Power Automate Action** kalau mau perhitungan rate lebih presisi daripada sekadar lookup tabel matrix
- [ ] **Test di Test Panel** — coba minimal 3 skenario berbeda (saldo besar, saldo kecil, customer baru) untuk pastikan rate yang keluar konsisten dengan matrix
- [ ] **Publish ke channel** — pilih Teams atau "Demo website" (embed) supaya bisa direkam sebagai video demo atau ditunjukkan live ke judges
- [ ] **Rekam demo 1-2 menit** menunjukkan 2 skenario customer berbeda dapat rate berbeda — ini yang dipakai untuk slide "See it in Action"

---

## Catatan

- Kalau waktu terbatas, fokus dulu ke 1 topic utama (Rekomendasi Rate) dengan 2-3 skenario saja — lebih baik demo simpel yang jalan mulus daripada banyak topic tapi belum matang.
- Simpan hasil rekaman demo dalam format screen recording pendek (bisa pakai Clipchamp atau built-in screen recorder Windows) untuk ditempel di slide PPTX yang sudah dibuat sebelumnya.
