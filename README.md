# Pj1: Sistem Persepsi Deteksi Panah dan Huruf/Angka dengan YOLO

**Mata kuliah:** RET503 · Minggu 4-7 (Praktik Terpadu & Pj1)
**Nama / NIM:** Jonathan Adis Pratama / 4222401018
**Anggota kelompok:** *(isi nama dan NIM anggota kelompok)*

---

## 1. Ringkasan

Projek ini membangun model deteksi objek YOLO (`yolov8n`) untuk mendeteksi dua kelompok objek pada gambar:

| ID kelas | Nama kelas | Isi |
|---|---|---|
| 0 | `panah` | panah biru ke kiri dan panah hijau ke kanan |
| 1 | `huruf_angka` | huruf A-Z dan angka 0-9 |

Tiga eksperimen dijalankan dengan pengaturan yang sama kecuali satu perubahan di tiap eksperimen. Model terbaik, **E3_tuning**, mencapai **mAP50 0.990 dan mAP50-95 0.893 pada data test**.

---

## 2. Lingkup Persepsi dan Target

- **Tugas persepsi:** mendeteksi dan mengklasifikasikan penanda (panah atau huruf/angka) beserta posisinya (bounding box) pada gambar kamera.
- **Keluaran model:** kelas, kotak (x, y, lebar, tinggi), dan skor kepercayaan per objek.
- **Metrik utama:** mAP50 dan mAP50-95 pada data validasi/test, serta waktu inferensi per gambar (ms).
- **Pengujian akhir:** data test dipakai **sekali saja**, hanya untuk model yang dipilih berdasarkan data validasi.

---

## 3. Dataset

### 3.1 Sumber

| Bagian | Sumber | Keterangan |
|---|---|---|
| `panah` | Dataset Roboflow `image-detection-qixfs` versi 12 (workspace `image-detection-obstn`, lisensi CC BY 4.0) | Foto asli. Hanya gambar yang memiliki bounding box yang dipakai; gambar berlabel kosong dibuang. Kelas asli `Blue Left Arrow` dan `Green Right Arrow` digabung menjadi satu kelas `panah`. |
| `huruf_angka` | **Dibuat sendiri (sintetis)** dengan skrip Python (PIL dan OpenCV) | Karakter A-Z dan 0-9 dengan variasi font, warna, ukuran, rotasi, perspektif, blur, kecerahan, dan latar. 1-3 karakter per gambar, sebagian di atas kartu gelap. Bounding box dihitung otomatis dari bentuk karakter. |

### 3.2 Jumlah data

| Split | Gambar | Gambar `panah` | Gambar `huruf_angka` | Objek `panah` | Objek `huruf_angka` |
|---|---|---|---|---|---|
| train | 732 | 366 | 366 | 367 | 569 |
| val | 162 | 81 | 81 | 81 | 124 |
| test | 100 | 50 | 50 | 50 | 70 |
| **Total** | **994** | 497 | 497 | 498 | 763 |

- Jumlah gambar per kelas seimbang di setiap split.
- Proporsi split: sekitar 74% train, 16% val, 10% test.
- Format label: YOLO (`class_id cx cy w h`, dinormalisasi 0-1).
- Sidik jari label dataset (dari skrip praktik): `0C0176781F10`.
- Pengecekan visual: sampel gambar dengan bounding box dan nama kelas disimpan di `cek_dataset.png`.

---

## 4. Pengaturan Eksperimen

Semua eksperimen memakai pengaturan yang sama, kecuali kolom "Perubahan".

| Parameter | Nilai |
|---|---|
| Model | `yolov8n` |
| Epoch | 50 |
| Batch | 16 |
| `imgsz` | 320 (E1, E2), 640 (E3) |
| Seed | 0 (`deterministic=True`) |
| Perangkat | GPU Tesla T4 (Google Colab) |
| Pustaka | Ultralytics 8.4.178, Python 3.13, PyTorch 2.11 |

| Eksperimen | Perubahan | Penjelasan |
|---|---|---|
| E1_pretrained | - (baseline) | Bobot pretrained COCO (transfer learning) |
| E2_scratch | bobot acak | Arsitektur sama, tanpa bobot pretrained |
| E3_tuning | `imgsz=640` | E1 dengan ukuran gambar masukan 640 |

---

## 5. Hasil

### 5.1 Tabel eksperimen (data validasi)

| Eksperimen | Perubahan | Dataset | mAP50 | mAP50-95 | ms/gambar | Kesimpulan |
|---|---|---|---|---|---|---|
| E1_pretrained | - (baseline) | v1 | 0.995 | 0.885 | 2.7 | Titik pembanding |
| E2_scratch | bobot acak | v1 | 0.988 | 0.788 | 2.0 | Tanpa transfer learning lebih rendah; manfaat transfer learning terbukti |
| E3_tuning | imgsz=640 | v1 | 0.995 | 0.903 | 5.8 | **Dipertahankan** (mAP50-95 tertinggi), dengan waktu inferensi sekitar 2 kali lebih lama |

Metrik tambahan:

| Eksperimen | Precision | Recall | Waktu latih (menit) |
|---|---|---|---|
| E1_pretrained | 0.995 | 1.000 | 5.2 |
| E2_scratch | 0.980 | 0.974 | 4.8 |
| E3_tuning | 0.999 | 0.996 | 9.0 |

`ms/gambar` adalah waktu inferensi model saja pada GPU T4 (belum termasuk pra- dan pascapemrosesan).

### 5.2 Model terpilih dan hasil test

- **Model terpilih:** `E3_tuning` (mAP50-95 validasi tertinggi, 0.903).
- **Hasil pada data test (satu kali pengujian):** mAP50 **0.990** · mAP50-95 **0.893**.

### 5.3 Grafik

Grafik loss dan metrik tiap eksperimen:

- `hasil_praktik/E1_pretrained/results.png`
- `hasil_praktik/E2_scratch/results.png`
- `hasil_praktik/E3_tuning/results.png`

---

## 6. Analisis

**E1 vs E2 (manfaat transfer learning).** E1 memulai training dari bobot pretrained dan sudah mencapai mAP50 sekitar 0.90 pada epoch pertama, sedangkan E2 dari bobot acak mulai dari nilai hampir nol dan baru mencapai mAP50 0.96 di sekitar epoch 23. Di akhir 50 epoch, E1 unggul 0.097 pada mAP50-95 (0.885 vs 0.788). Selisih mAP50 kecil (0.995 vs 0.988), artinya E2 sudah bisa menemukan objek dengan benar, tetapi kotaknya kurang presisi. Kurva E2 juga masih naik di epoch-epoch terakhir, sehingga E2 kemungkinan belum konvergen dan akan menyusul bila dilatih lebih lama. Kesimpulan: transfer learning mempercepat dan meningkatkan hasil pada data sekecil ini.

**E3 (tuning `imgsz=640`).** Menaikkan ukuran gambar dari 320 ke 640 menaikkan mAP50-95 dari 0.885 menjadi 0.903 (+0.018) tanpa mengubah mAP50 (0.995). Peningkatannya ada pada ketepatan kotak, bukan pada kemampuan menemukan objek. Harganya, waktu inferensi naik dari 2.7 ms menjadi 5.8 ms (sekitar 2.1 kali) dan waktu latih dari 5.2 menjadi 9.0 menit. Selisih +0.018 berasal dari satu kali training (satu seed), sehingga sebagian bisa jadi variasi acak. Perubahan ini **dipertahankan** karena E3 memiliki mAP50-95 tertinggi dan mengikuti aturan pemilihan model pada praktik. Bila sistem akhir harus berjalan pada perangkat dengan daya terbatas, E1 (`imgsz=320`) adalah alternatif yang hampir sama akurat dan lebih cepat.

**Hasil test.** mAP50-95 pada data test (0.893) hanya berbeda 0.010 dari data validasi (0.903), sehingga tidak ada tanda pemilihan model yang menyesuaikan diri berlebihan terhadap data validasi.

**Keterbatasan.**

1. **Data `huruf_angka` bersifat sintetis**, sedangkan data `panah` adalah foto asli. Model bisa memanfaatkan perbedaan gaya gambar (sintetis vs foto) sebagai petunjuk, sehingga nilai mAP yang mendekati 1.0 belum tentu mencerminkan kinerja pada foto huruf/angka yang sebenarnya.
2. Banyak foto panah berasal dari lokasi dan benda yang sama, sehingga gambar di train, val, dan test mungkin sangat mirip. Ini membuat metrik cenderung optimistis.
3. Seluruh karakter A-Z dan 0-9 digabung menjadi satu kelas, jadi model hanya membedakan "panah" dan "huruf/angka", bukan karakter tertentu, dan tidak membedakan arah panah.
4. Tiap eksperimen hanya dijalankan sekali (satu seed), sehingga selisih kecil antar eksperimen tidak dapat dianggap signifikan secara statistik.
5. Kecepatan hanya diukur pada GPU T4, belum pada perangkat target robot.

**Langkah perbaikan.** Menambahkan foto asli huruf/angka dari kamera robot, memisahkan split berdasarkan lokasi atau sesi pengambilan, menjalankan beberapa seed, dan mengukur kecepatan pada perangkat target.

---

## 7. Struktur File yang Dikumpulkan

```
hasil_praktik_4222401018.txt     # laporan tabel E1-E3 dan hasil test
hasil_praktik_4222401018.csv     # data yang sama dalam format CSV
hasil_praktik/
├── E1_pretrained/               # results.png, confusion matrix, bobot
├── E2_scratch/
└── E3_tuning/                   # results.png, confusion matrix, bobot
cek_dataset.png                  # contoh gambar dengan bounding box
README.md                        # dokumen ini
```

---

## 8. Cara Menjalankan Ulang

1. Buka Google Colab, pilih runtime GPU (T4).
2. Upload `dataset_gabungan_panah_hurufangka.zip` dan `YOLO.zip` ke Colab, lalu ekstrak.
3. Ubah struktur dataset menjadi `dataset_v1` (`images/{train,val,test}` dan `labels/{train,val,test}`) dan buat `data.yaml` dengan dua kelas.
4. Jalankan skrip praktik:

```bash
pip install ultralytics
python 06_latih_bandingkan.py --nim 4222401018 --nama "Jonathan Adis Pratama" \
    --data /content/dataset_v1/data.yaml --model yolov8n \
    --epochs 50 --imgsz 320 --ubah imgsz=640
```

Skrip ini melatih E1, E2, dan E3, memilih model terbaik berdasarkan mAP50-95 validasi, lalu menguji model itu sekali pada data test.

---

## 9. Sumber dan Atribusi

- Dataset panah: Roboflow Universe, `image-detection-qixfs` v12, lisensi CC BY 4.0.
- Model dan pustaka: Ultralytics YOLOv8.
- Dataset `huruf_angka` dibuat sendiri dengan skrip Python (PIL, OpenCV, NumPy) memakai font DejaVu.
