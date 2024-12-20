# Analisis Attrition Rate di Perusahaan Jaya Jaya Maju

## Deskripsi Proyek
Proyek ini bertujuan untuk menganalisis faktor-faktor yang memengaruhi *attrition rate* (rasio karyawan yang keluar) di perusahaan Jaya Jaya Maju. Dengan memanfaatkan dataset karyawan dan alat analisis, proyek ini menghasilkan wawasan yang dapat membantu perusahaan mengidentifikasi risiko, merencanakan langkah mitigasi, dan meningkatkan kepuasan kerja karyawan.

---

## Permasalahan Bisnis
Perusahaan Jaya Jaya Maju menghadapi tingkat attrition yang tinggi, khususnya pada departemen tertentu. Hal ini memengaruhi stabilitas operasional perusahaan serta meningkatkan biaya rekrutmen dan pelatihan untuk menggantikan karyawan yang keluar. Beberapa masalah utama yang dihadapi adalah:

1. **Gaji Tidak Kompetitif**: Beberapa karyawan merasa gaji yang diterima tidak sebanding dengan kontribusi mereka.
2. **Kepuasan Kerja Rendah**: Lingkungan kerja dan dukungan manajerial tidak mendukung perkembangan karier karyawan.
3. **Beban Kerja Tinggi**: Terutama pada departemen dengan attrition rate tertinggi, seperti Produksi dan IT.

### Tujuan Bisnis
- Menurunkan tingkat attrition dengan memahami faktor-faktor penyebab utama.
- Memberikan rekomendasi strategis berbasis data untuk meningkatkan retensi karyawan.
- Mengurangi biaya rekrutmen dan pelatihan akibat pergantian karyawan.

---

## Cakupan Proyek
Proyek ini meliputi langkah-langkah berikut:

1. **Eksplorasi Data (Data Understanding)**:
   - Memahami distribusi data seperti gaji, kinerja, usia, masa kerja, dan tingkat kepuasan kerja.
   - Mengetahui hubungan antar variabel untuk mengidentifikasi faktor risiko.

2. **Analisis Data**:
   - Menggunakan teknik analisis statistik dan visualisasi untuk mengungkap pola attrition.
   - Menerapkan model prediktif untuk mengidentifikasi karyawan berisiko tinggi keluar.

3. **Dashboard Visualisasi**:
   - Membuat dashboard interaktif untuk menyampaikan insight kepada manajemen.

4. **Rekomendasi dan Strategi Mitigasi**:
   - Memberikan langkah-langkah konkret untuk meningkatkan kepuasan kerja dan retensi karyawan.

---

## Visualisasi Dashboard dan Insight

![mjuna-dashboard](https://github.com/user-attachments/assets/d635c91c-b326-41e4-a195-5f23a7e6fbe0)

### 1. **Rata-rata Gaji dan Kinerja Karyawan**
- **Visualisasi**: Scatter plot atau bar chart yang menggambarkan hubungan antara gaji bulanan dan kinerja karyawan.
- **Insight**:
  - Karyawan dengan gaji lebih rendah (di bawah rata-rata) cenderung memiliki tingkat *attrition* lebih tinggi.
  - Kelompok dengan kinerja tinggi tetapi gaji rendah memiliki risiko keluar yang paling besar.
- **Rekomendasi**:
  - Tinjau ulang struktur gaji, khususnya untuk karyawan dengan kinerja tinggi, agar sesuai dengan kontribusinya.

---

### 2. **Tingkat Kepuasan Kerja dan Pengaruhnya terhadap Attrition**
- **Visualisasi**: Histogram atau pie chart yang menunjukkan distribusi kepuasan kerja antara karyawan yang keluar dan bertahan.
- **Insight**:
  - Karyawan yang keluar cenderung memiliki tingkat kepuasan kerja rendah (skor < 3 dari 5).
  - Faktor yang memengaruhi kepuasan kerja mencakup kesempatan promosi, lingkungan kerja, dan dukungan manajerial.
- **Rekomendasi**:
  - Tingkatkan kebijakan kesejahteraan, pelatihan pengembangan diri, dan program penghargaan berbasis kinerja.

---

### 3. **Attrition Berdasarkan Departemen**
- **Visualisasi**: Bar chart yang menampilkan tingkat *attrition* di tiap departemen.
- **Insight**:
  - Departemen dengan tingkat *attrition* tertinggi adalah **Produksi** dan **IT**.
  - Masalah utama mencakup beban kerja tinggi dan minimnya pengakuan terhadap kontribusi.
- **Rekomendasi**:
  - Lakukan survei khusus pada departemen dengan *attrition* tinggi.
  - Terapkan kebijakan keseimbangan kerja-hidup untuk mengurangi tekanan kerja.

---

### 4. **Proyeksi Attrition per Bulan**
- **Visualisasi**: Line chart yang menunjukkan tren rata-rata karyawan keluar per bulan.
- **Insight**:
  - Ada pola musiman, seperti peningkatan setelah penilaian tahunan atau pembayaran bonus.
- **Rekomendasi**:
  - Jadwalkan inisiatif retensi seperti kenaikan gaji atau promosi sebelum periode *attrition* meningkat.
  - Tingkatkan komunikasi terkait rencana pengembangan karier.

---

### 5. **Distribusi Faktor Risiko Attrition**
- **Visualisasi**: Heatmap atau radar chart yang menunjukkan korelasi antara faktor risiko (gaji, kepuasan kerja, usia, masa kerja) dan tingkat *attrition*.
- **Insight**:
  - Karyawan dengan masa kerja < 2 tahun dan usia < 30 tahun memiliki risiko *attrition* tertinggi.
  - Gaji dan kepuasan kerja adalah dua faktor utama.
- **Rekomendasi**:
  - Kembangkan program orientasi dan mentoring untuk karyawan baru.
  - Tawarkan jalur karier dan peluang pengembangan untuk meningkatkan loyalitas karyawan muda.

---

### 6. **Proyeksi Karyawan Berisiko Tinggi**
- **Visualisasi**: Tabel interaktif yang menampilkan daftar karyawan berisiko tinggi untuk keluar berdasarkan model prediktif.
- **Insight**:
  - Model prediktif mengidentifikasi karyawan dengan skor risiko tinggi berdasarkan gaji, kepuasan kerja, dan lama bekerja.
- **Rekomendasi**:
  - Lakukan diskusi dengan karyawan berisiko tinggi untuk memahami kebutuhan mereka.
  - Berikan intervensi seperti kenaikan gaji, pelatihan, atau perubahan tanggung jawab kerja.

---

## Setup Environment

### 1. Membuat Virtual Environment
Gunakan *virtual environment* untuk mengisolasi dependensi proyek:

# Buat virtual environment
```python -m venv venv```

# Aktifkan virtual environment
```venv\Scripts\activate```

# Install Dependencies
``` pip install -r requirements.txt ```

## Akses Dashboard
- **Link Dashboard**: [Dashboard Jaya Jaya Maju - Analisis Attrition](#) *(Link akan tersedia setelah dashboard dipublikasikan di Metabase)*

---

## Kesimpulan
Dari analisis ini, beberapa faktor utama yang memengaruhi *attrition rate* di perusahaan Jaya Jaya Maju adalah:
1. **Gaji Tidak Kompetitif**: Karyawan dengan gaji rendah lebih rentan untuk keluar.
2. **Kepuasan Kerja Rendah**: Faktor lingkungan kerja dan dukungan manajerial memainkan peran besar.
3. **Kurangnya Dukungan dari Manajer**: Karyawan yang merasa tidak mendapat pengakuan dari atasan lebih cenderung untuk keluar.

---

## Rekomendasi Umum
1. **Penyesuaian Gaji**:
   - Lakukan analisis pasar untuk memastikan gaji kompetitif.
2. **Pengembangan Program Pelatihan**:
   - Tingkatkan program pelatihan untuk memperbaiki peluang karier.
3. **Dukungan Kesejahteraan Karyawan**:
   - Tambahkan kebijakan keseimbangan kerja-hidup dan fasilitas kesejahteraan.

Dengan menggunakan insight dari dashboard ini, manajemen dapat mengambil langkah strategis untuk meningkatkan retensi karyawan dan menciptakan lingkungan kerja yang lebih baik.
