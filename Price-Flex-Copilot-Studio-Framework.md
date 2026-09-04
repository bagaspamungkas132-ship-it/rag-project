# Price Flex — Framework & Materi untuk Hackathon OCBC (Microsoft Copilot Studio)

Ringkasan ide: agent yang menganalisa profil nasabah (saldo, histori transaksi, tenor preference, engagement) untuk merekomendasikan rate/promo TAKA Online atau Deposito Online yang dipersonalisasi, lalu mengirim notifikasi ke nasabah — dalam batas guardrail yang disetujui treasury.

---

## BAGIAN 1 — Isi Slide (mengikuti template PPT kamu)

### 1. Problem Statement
> Indonesia memasuki era digital-first dengan Gen Z & Milenial (≈49,3% populasi) yang terbiasa dengan personalisasi, gamifikasi, dan promo real-time di platform seperti marketplace, hotel, dan tiket. Namun produk simpanan seperti TAKA Online dan Deposito Online masih pakai "satu rate untuk semua", padahal histori transaksi, saldo, dan engagement tiap nasabah berbeda. Ini membuat konversi dan loyalitas segmen Emerging Affluent tidak optimal.

**Background (isi 4 poin di slide):**
1. **Skenario** — dynamic pricing sudah lazim di industri lain (Tiket.com, Traveloka menampilkan harga berbeda berdasarkan demand/timing/segmen); bank belum memanfaatkan pendekatan ini untuk funding products.
2. **Pain point** — rate promo TAKA/Deposito seragam untuk semua nasabah meski profil (saldo, produk yang dimiliki, histori transaksi) berbeda → bank kehilangan peluang konversi & margin optimal.
3. **Kondisi saat ini** — pricing ditetapkan manual/seragam oleh treasury, tidak ada mekanisme personalisasi otomatis.
4. **Pihak terdampak** — nasabah Emerging Affluent (Gen Z/Milenial digital-savvy), tim treasury/pricing, tim funding/product growth.

### 2. Solution
**What it does:** Agent AI yang menganalisa data internal nasabah (saldo, histori funding/lending, pola transaksi) untuk merekomendasikan rate/promo personalisasi TAKA Online & Deposito Online, dalam rentang yang sudah di-approve treasury — lalu memicu notifikasi personalisasi ke nasabah yang eligible.

**How it works:**
- **Data yang dipakai:** saldo tabungan, histori produk funding/lending, pola & jam transaksi, tenor preference, skor engagement.
- **Bagaimana AI memakai data:** model/prompt scoring menilai propensity nasabah untuk membeli TAKA/Deposito, lalu memilih tier rate dari rentang pre-approved (bukan open pricing).
- **Peran user (customer):** menerima notifikasi promo yang relevan di waktu yang tepat (berdasarkan pola jam transaksi mereka).
- **Workflow:** data nasabah → filter eligibility → scoring/recommendation engine → guardrail check → generate pesan notifikasi personalisasi → kirim via channel (push/app/email).

**One-sentence solution statement:** "Price Flex mengganti 'satu rate untuk semua' dengan 'rate yang tepat, untuk nasabah yang tepat, di waktu yang tepat', tanpa melepas kendali pricing dari treasury."

### 3. Benefits and Challenges
**Expected benefits:** peningkatan konversi pembelian TAKA Online/Deposito Online, funding cost tetap terkendali (dibanding menaikkan rate untuk semua nasabah), engagement nasabah digital-savvy meningkat.
**Path to value capture:** integrasi ke sistem notifikasi existing (app/push), perlu proses approval treasury untuk rentang rate, adopsi oleh tim funding/marketing sebagai kanal promo baru.
**Challenges:** kualitas & akses data transaksi nasabah (perlu izin/API internal), guardrail supaya pricing tidak dianggap diskriminatif antar nasabah sejenis, review compliance/risk sebelum live.

### 4. Benefit Estimation (isi tabel)
| Benefit | Tipe | Metric/proxy | Asumsi & sumber | Confidence |
|---|---|---|---|---|
| Peningkatan konversi TAKA/Deposito Online | Quantified | % kenaikan jumlah pembelian per bulan | Baseline dari data pilot/simulasi sample | Medium |
| Efisiensi funding cost | Quantified | Selisih cost of fund vs skenario "rate seragam naik" | Estimasi treasury/benchmark internal | Medium |
| Engagement notifikasi | Qualitative | Open rate / response rate notifikasi personalisasi vs generik | Observasi pilot/A-B test | Medium |
| Kepercayaan & pengalaman nasabah | Qualitative | Rated High/Med/Low dibanding baseline promo generik | Judgement tim (belum dimonetisasi) | Low |

### 5. Architecture & Tech Stack (untuk slide Appendix)
```
[Data Nasabah: saldo, histori produk, transaksi] 
        ↓
[Power Automate / Dataverse: eligibility filter]
        ↓
[Copilot Studio Agent: scoring & rate recommendation
   - Prompt/Knowledge: rules pricing pre-approved treasury
   - Guardrails: min/max rate, eligibility rules]
        ↓
[Generate pesan notifikasi personalisasi]
        ↓
[Power Automate: kirim ke channel notifikasi (app/email/Teams demo)]
```
**Tech stack:** Microsoft Copilot Studio (agent + generative orchestration), Power Automate (trigger & connector), Dataverse atau Excel (sample data untuk demo), Power Fx (guardrail/logic).
**AI model:** model bawaan Copilot Studio (GPT via Copilot Studio's generative answers/prompt), dengan instruksi yang membatasi output ke rentang rate yang di-approve.

### 6. Other supporting information (Appendix)
Poin yang perlu disiapkan untuk Q&A juri:
- Bagaimana guardrail mencegah pricing diskriminatif → jelaskan aturan eligibility yang berbasis parameter objektif (saldo, tenor, cash flow stability), bukan atribut personal yang sensitif.
- Bagaimana kalau data nasabah tidak lengkap → fallback ke rate standar.
- Bagaimana skalanya kalau dipakai ke jutaan nasabah → batching lewat Power Automate/Dataverse, bukan real-time per klik.

---

## BAGIAN 2 — Cara Membangun Agent di Microsoft Copilot Studio

### Langkah 0 — Siapkan sebelum mulai
1. Akses Microsoft Copilot Studio (via lisensi Microsoft 365 Copilot atau trial environment).
2. Siapkan **sample data nasabah** dalam Excel/Dataverse (lihat skema di bawah) — karena kamu tidak akan pakai data nasabah asli saat hackathon.
3. Siapkan dokumen "pricing rules" pre-approved treasury versi simulasi (misalnya: rate dasar TAKA 3,5%–4,75% tergantung tenor & saldo, rate dasar Deposito tergantung tenor 1–12 bulan) — ini jadi rules/guardrail untuk agent, bukan open pricing.

### Langkah 1 — Buat Agent Baru
- Buat agent baru, beri nama & deskripsi jelas (mis. "Price Flex — Personalized Pricing Recommender for TAKA/Deposito Online"), karena deskripsi ini dipakai orchestrator untuk memilih topic/tool yang tepat.
- Aktifkan **generative orchestration** — ini penting karena fitur event trigger (untuk skenario "notifikasi otomatis tanpa user chat duluan") hanya tersedia kalau generative orchestration aktif.

### Langkah 2 — Tambahkan Knowledge Sources
- Upload dokumen pricing rules (rentang rate pre-approved per tenor/produk) sebagai file knowledge source.
- Upload/link halaman produk TAKA Online & Deposito Online sebagai referensi (agar agent tahu detail produk saat generate pesan notifikasi).
- Beri deskripsi jelas di tiap knowledge source, supaya agent tahu kapan harus memakainya.

### Langkah 3 — Buat Topic/Prompt untuk Scoring & Recommendation
Buat sebuah **Prompt** (atau topic dengan node "Create a prompt with Copilot") dengan instruksi kurang lebih:
- Input: profil nasabah (saldo, histori produk, pola transaksi, tenor preference).
- Tugas: (1) cek eligibility berdasarkan parameter (saldo minimum, stabilitas cash flow, tenor fit), (2) pilih rate/promo dari rentang pre-approved sesuai profil, (3) kalau tidak eligible atau data tidak cukup, kembalikan rate standar.
- Guardrail eksplisit di instruksi: "Jangan pernah merekomendasikan rate di luar rentang yang tercantum di knowledge source pricing rules."
- Output: rate/tier yang direkomendasikan + draft pesan notifikasi personalisasi.

### Langkah 4 — Buat Event Trigger untuk Notifikasi Otomatis
Karena skenario ini bersifat **proaktif** (bukan nasabah yang chat duluan, tapi sistem yang mendeteksi nasabah eligible lalu mengirim notifikasi), gunakan **event trigger**:
- Trigger sumber: Dataverse row update / recurrence schedule (misal setiap malam, cek batch nasabah baru yang eligible) — via konektor Power Automate.
- Payload: bawa data profil nasabah sebagai variable ke agent.
- Agent menerima payload → jalankan topic scoring di Langkah 3 → hasilkan pesan notifikasi.
- Catatan: event trigger butuh autentikasi maker (bukan user) supaya bisa jalan otomatis tanpa nasabah login duluan.

### Langkah 5 — Kirim Notifikasi
- Tambahkan action/tool berupa Power Automate flow yang mengirim hasil (pesan notifikasi) ke channel — untuk demo hackathon bisa pakai Teams message atau email ke "customer" simulasi, atau tampilkan di Excel/PPT sebagai contoh output (karena integrasi ke app notification production nasabah asli di luar scope hackathon).

### Langkah 6 — Testing
- Uji dengan beberapa baris sample data nasabah berbeda profil (lihat skema di Bagian 3), pastikan:
  - Nasabah tidak eligible → dapat rate standar, bukan promo.
  - Nasabah eligible saldo besar & tenor panjang → dapat tier rate lebih tinggi (dalam batas guardrail).
  - Tidak pernah keluar dari rentang rate yang di-approve.
- Gunakan tab Activity di Copilot Studio untuk melihat kenapa agent memilih output tertentu (berguna untuk debug & demo ke juri).

---

## BAGIAN 3 — Skema Sample Data Nasabah (untuk Excel/Dataverse)

| Kolom | Contoh isi | Keterangan |
|---|---|---|
| customer_id | CUST-001 | ID unik (dummy) |
| saldo_tabungan | 15,000,000 | Saldo rata-rata bulanan |
| produk_dimiliki | Tabungan, Kartu Kredit | Produk existing |
| histori_transaksi_bulanan | 20 transaksi | Aktivitas transaksi |
| jam_transaksi_favorit | 19:00–21:00 | Untuk timing notifikasi |
| tenor_preference | 6 bulan | Preferensi tenor jika sudah pernah tanya/checkout |
| engagement_score | Tinggi/Sedang/Rendah | Skor internal (dummy) |
| eligible_TAKA | Ya/Tidak | Hasil filter eligibility |
| eligible_Deposito | Ya/Tidak | Hasil filter eligibility |

Buat 15–20 baris dengan variasi kombinasi (saldo besar+tenor panjang, saldo kecil+tenor pendek, dst) supaya demo scoring-nya kelihatan bervariasi.

---

## Catatan Penting untuk Guardrail & Compliance
- Rate yang direkomendasikan **harus** selalu berada dalam rentang yang sudah ditentukan tim treasury/product owner — agent tidak boleh membuat rate baru sendiri.
- Parameter eligibility harus berbasis data objektif (saldo, stabilitas cash flow, tenor fit) — hindari parameter yang bisa dianggap diskriminatif terhadap segmen nasabah tertentu.
- Untuk hackathon, jelaskan bahwa versi production nantinya perlu review risk & compliance sebelum notifikasi benar-benar dikirim ke nasabah asli.
