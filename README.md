# Dashboard SH-RD — Menjalankan di Lokal

Panduan menjalankan dashboard monitoring sentimen di komputer sendiri (tanpa Colab / Cloudflared).

## Versi Python
Gunakan **Python 3.11 atau 3.12**.
- 3.12 sudah diuji jalan penuh (import + LDA + coherence).
- 3.11 juga aman dan paling luas kompatibilitasnya.
- Hindari 3.13 (beberapa paket belum tentu punya wheel) dan di bawah 3.10.

Cek versi:
```
python --version
```

## Perbaikan dari versi Colab
Versi lokal ini berbeda dari notebook Colab pada dua hal:
1. **Path data** tidak lagi menunjuk ke Google Drive. Sekarang otomatis membaca
   `./data/hasil_prediksi.csv` (folder `data/` sejajar dengan `app.py`).
   Bisa dioverride lewat variabel lingkungan `SHRD_DATA_DIR`.
2. **Tanpa mount Drive & tanpa tunnel Cloudflared** — cukup `streamlit run app.py`
   lalu buka di browser lokal. File catatan tersimpan ke `./data/catatan_monitoring.csv`.

## Struktur folder
```
shrd-dashboard-lokal/
├─ app.py
├─ requirements.txt
├─ .streamlit/
│  └─ config.toml
└─ data/
   └─ hasil_prediksi.csv   <-- taruh file datamu di sini
```

## Langkah menjalankan

### Windows (PowerShell / CMD)
```
cd shrd-dashboard-lokal
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Salin `hasil_prediksi.csv` ke folder `data\`, lalu:
```
streamlit run app.py
```
Browser terbuka otomatis; kalau tidak, buka http://localhost:8501

### macOS / Linux
```
cd shrd-dashboard-lokal
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Salin `hasil_prediksi.csv` ke folder `data/`, lalu:
```
streamlit run app.py
```
Buka http://localhost:8501

### Menghentikan
Tekan `Ctrl + C` di terminal. Untuk keluar dari venv: ketik `deactivate`.

## Kalau nama kolom CSV berbeda
Dashboard mengenali otomatis kolom: `content`, `sentimen_pred`, `rating`,
`tanggal`, `platform`, `username`, `stemming`. Kalau namamu beda, atur lewat
menu **"Pemetaan kolom"** di sidebar (tidak perlu edit kode).

## Troubleshooting

**`streamlit: command not found`** → venv belum aktif. Aktifkan dulu
(`venv\Scripts\activate` di Windows, `source venv/bin/activate` di Mac/Linux),
atau jalankan lewat `python -m streamlit run app.py`.

**Error dependensi gensim / scipy / numpy** → buka `requirements.txt`, hapus
tanda pagar pada blok versi PASTI di bawah, lalu jalankan ulang
`pip install -r requirements.txt`.

**Klik irisan donut tidak memunculkan pop-up** → butuh Streamlit ≥ 1.35
(sudah dijamin oleh requirements). Sebagai cadangan, tombol **"Lihat"** di
sebelah legenda selalu berfungsi.

**"File data tidak ditemukan"** → pastikan `hasil_prediksi.csv` benar-benar
berada di folder `data/`, atau set `SHRD_DATA_DIR` ke lokasi filenya:
- Windows: `set SHRD_DATA_DIR=D:\path\ke\folder`
- Mac/Linux: `export SHRD_DATA_DIR=/path/ke/folder`

**Port 8501 dipakai** → jalankan di port lain:
`streamlit run app.py --server.port 8600`
