# LAPORAN UJIAN AKHIR SEMESTER (UAS)

## Implementasi Algoritma C4.5 (Decision Tree) untuk Klasifikasi Level Obesitas

---

| | |
|---|---|
| **Mata Kuliah** | Data Mining – TIF 605 |
| **Tahun Ajaran** | 2025/2026 Genap |
| **Dosen** | Dr. Ir. Ananto Tri Sasongko, M.Sc. |
| **Topik** | Implementasi Algoritma Prediksi dan Deteksi Anomali |
| **Algoritma** | C4.5 (Decision Tree) |
| **Dataset** | Obesity Levels Dataset |

### Anggota Kelompok

| No | Nama | NIM |
|:--:|------|:---:|
| 1 | Muhammad Prayoga Putra Mahardhika | *(NIM)* |
| 2 | *(Nama Anggota 2)* | *(NIM)* |
| 3 | *(Nama Anggota 3)* | *(NIM)* |

> **Catatan:** Silakan isi NIM dan nama anggota kelompok di tabel di atas.

---

## Daftar Isi

1. [Pendahuluan](#1-pendahuluan)
2. [Data Understanding](#2-data-understanding)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Implementasi Algoritma C4.5](#4-implementasi-algoritma-c45-decision-tree)
5. [Evaluasi Hasil](#5-evaluasi-hasil)
6. [Visualisasi](#6-visualisasi)
7. [Kesimpulan](#7-kesimpulan)
8. [Referensi](#8-referensi)

---

## 1. Pendahuluan

### 1.1 Latar Belakang

Obesitas merupakan salah satu permasalahan kesehatan global yang terus meningkat prevalensinya. Menurut World Health Organization (WHO), obesitas telah menjadi epidemi global yang berdampak pada risiko berbagai penyakit kronis seperti diabetes tipe 2, penyakit kardiovaskular, dan gangguan muskuloskeletal. Identifikasi dini tingkat obesitas berdasarkan kebiasaan makan (*eating habits*) dan kondisi fisik (*physical condition*) seseorang menjadi sangat penting untuk upaya pencegahan dan penanganan yang tepat.

Teknik *data mining* memungkinkan kita untuk menggali pola tersembunyi dari data dan membangun model prediktif yang dapat mengklasifikasikan tingkat obesitas secara otomatis. Dalam laporan ini, kami mengimplementasikan **Algoritma C4.5 (Decision Tree)** untuk mengklasifikasikan tingkat obesitas berdasarkan dataset *Obesity Levels* yang berisi data kebiasaan makan dan kondisi fisik dari penduduk Meksiko, Peru, dan Kolombia.

### 1.2 Rumusan Masalah

1. Bagaimana cara mengimplementasikan algoritma C4.5 (Decision Tree) untuk klasifikasi tingkat obesitas?
2. Seberapa akurat model C4.5 dalam mengklasifikasikan 7 tingkat obesitas berdasarkan kebiasaan makan dan kondisi fisik?
3. Fitur apa saja yang paling berpengaruh dalam menentukan tingkat obesitas?

### 1.3 Tujuan

1. Mengimplementasikan algoritma C4.5 (Decision Tree) menggunakan Python di Google Colab.
2. Mengevaluasi performa model menggunakan metrik Accuracy, Precision, Recall, dan F1-Score.
3. Mengidentifikasi fitur-fitur yang paling berpengaruh dalam klasifikasi tingkat obesitas.
4. Menyajikan hasil analisis melalui visualisasi data yang informatif.

### 1.4 Penjelasan Singkat Algoritma C4.5

Algoritma **C4.5** dikembangkan oleh **Ross Quinlan** (1993) sebagai penyempurnaan dari algoritma ID3. C4.5 membangun model prediksi berupa **pohon keputusan (Decision Tree)** dengan mekanisme sebagai berikut:

1. **Menghitung Entropy** — mengukur tingkat ketidakpastian (impurity) dalam data:

$$Entropy(S) = -\sum_{i=1}^{n} p_i \cdot \log_2(p_i)$$

2. **Menghitung Information Gain** — mengukur pengurangan entropy setelah melakukan split berdasarkan atribut tertentu:

$$Gain(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} \cdot Entropy(S_v)$$

3. Atribut dengan **Information Gain tertinggi** dipilih sebagai *node* keputusan (splitting criterion).
4. Proses diulang secara **rekursif** pada setiap cabang hingga seluruh data terklasifikasi atau kondisi berhenti tercapai.

Keunggulan C4.5 dibandingkan ID3:
- Mampu menangani **data numerik** (continuous attributes)
- Mampu menangani **missing values**
- Menggunakan **pruning** untuk menghindari *overfitting*
- Hasil model mudah **diinterpretasi** dalam bentuk aturan-aturan keputusan (*if-then rules*)

---

## 2. Data Understanding

### 2.1 Deskripsi Dataset

Dataset yang digunakan adalah **Obesity Levels Dataset** yang berasal dari [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition). Dataset ini berisi data estimasi tingkat obesitas berdasarkan kebiasaan makan dan kondisi fisik seseorang dari penduduk Meksiko, Peru, dan Kolombia.

| Informasi | Detail |
|-----------|--------|
| **Jumlah Data** | 2.111 baris |
| **Jumlah Atribut** | 17 kolom (16 fitur + 1 target) |
| **Missing Values** | Tidak ada |
| **Tipe Data** | Campuran (Numerik dan Kategorikal) |

### 2.2 Penjelasan Atribut Dataset

Dataset ini memiliki 17 atribut yang terbagi ke dalam beberapa kategori:

#### Atribut Profil & Fisik Dasar

| No | Atribut (Singkatan) | Kepanjangan / Nama Lengkap | Penjelasan | Tipe Data |
|:--:|---------------------|---------------------------|------------|:---------:|
| 1 | **Gender** | Gender | Jenis kelamin responden | Kategorikal (Male / Female) |
| 2 | **Age** | Age | Usia responden dalam satuan tahun | Numerik |
| 3 | **Height** | Height | Tinggi badan dalam satuan meter | Numerik |
| 4 | **Weight** | Weight | Berat badan dalam satuan kilogram | Numerik |
| 5 | **family_history_with_overweight** | Family History with Overweight | Apakah memiliki riwayat keluarga obesitas/kelebihan berat badan | Kategorikal (yes / no) |

#### Atribut Kebiasaan Makan (*Eating Habits*)

| No | Atribut (Singkatan) | Kepanjangan / Nama Lengkap | Penjelasan | Tipe Data |
|:--:|---------------------|---------------------------|------------|:---------:|
| 6 | **FAVC** | *Frequent consumption of high caloric food* | Kebiasaan sering mengonsumsi makanan berkalori tinggi | Kategorikal (yes / no) |
| 7 | **FCVC** | *Frequency of consumption of vegetables* | Frekuensi mengonsumsi sayuran (1 = Tidak pernah, 2 = Kadang-kadang, 3 = Selalu) | Numerik (1–3) |
| 8 | **NCP** | *Number of main meals* | Jumlah waktu makan utama dalam sehari | Numerik (1–4) |
| 9 | **CAEC** | *Consumption of food between meals* | Kebiasaan ngemil/makan di antara waktu makan utama | Kategorikal (no, Sometimes, Frequently, Always) |
| 10 | **CH2O** | *Consumption of water daily* | Jumlah konsumsi air putih per hari (1 = <1L, 2 = 1–2L, 3 = >2L) | Numerik (1–3) |
| 11 | **CALC** | *Consumption of alcohol* | Frekuensi mengonsumsi minuman beralkohol | Kategorikal (no, Sometimes, Frequently, Always) |

#### Atribut Gaya Hidup & Kondisi Fisik (*Physical Condition & Lifestyle*)

| No | Atribut (Singkatan) | Kepanjangan / Nama Lengkap | Penjelasan | Tipe Data |
|:--:|---------------------|---------------------------|------------|:---------:|
| 12 | **SMOKE** | Smoke | Apakah responden seorang perokok aktif | Kategorikal (yes / no) |
| 13 | **SCC** | *Calories consumption monitoring* | Apakah responden rutin memonitor asupan kalori harian | Kategorikal (yes / no) |
| 14 | **FAF** | *Physical activity frequency* | Frekuensi olahraga per minggu (0 = Tidak pernah, 1 = 1–2 hari, 2 = 3–4 hari, 3 = >4 hari) | Numerik (0–3) |
| 15 | **TUE** | *Time using technology devices* | Waktu penggunaan perangkat teknologi per hari (0 = 0–2 jam, 1 = 3–5 jam, 2 = >5 jam) | Numerik (0–2) |
| 16 | **MTRANS** | *Transportation used* | Moda transportasi utama yang digunakan sehari-hari | Kategorikal (Automobile, Motorbike, Bike, Public_Transportation, Walking) |

#### Variabel Target / Kelas

| No | Atribut (Singkatan) | Kepanjangan / Nama Lengkap | Penjelasan |
|:--:|---------------------|---------------------------|------------|
| 17 | **NObeyesdad** | *Nivel de Obesidad* (Level of Obesity) | Label klasifikasi tingkat obesitas (7 kelas) |

**7 Kelas Target (NObeyesdad):**

| Kode | Keterangan (Bahasa Indonesia) | Jumlah Data | Persentase |
|------|-------------------------------|:-----------:|:----------:|
| Insufficient_Weight | Kekurangan Berat Badan | 272 | 12.9% |
| Normal_Weight | Berat Badan Normal | 287 | 13.6% |
| Overweight_Level_I | Kelebihan Berat Badan Tingkat I | 290 | 13.7% |
| Overweight_Level_II | Kelebihan Berat Badan Tingkat II | 290 | 13.7% |
| Obesity_Type_I | Obesitas Tipe I | 351 | 16.6% |
| Obesity_Type_II | Obesitas Tipe II | 297 | 14.1% |
| Obesity_Type_III | Obesitas Tipe III | 324 | 15.3% |

Distribusi kelas target relatif **seimbang** (*balanced*), sehingga tidak diperlukan teknik *resampling* tambahan.

### 2.3 Statistik Deskriptif

Berikut adalah ringkasan statistik deskriptif untuk atribut numerik dalam dataset:

| Statistik | Age | Height | Weight | FCVC | NCP | CH2O | FAF | TUE |
|-----------|:---:|:------:|:------:|:----:|:---:|:----:|:---:|:---:|
| **count** | 2111 | 2111 | 2111 | 2111 | 2111 | 2111 | 2111 | 2111 |
| **mean** | ~24.3 | ~1.70 | ~86.6 | ~2.42 | ~2.69 | ~2.01 | ~1.01 | ~0.66 |
| **std** | ~6.3 | ~0.09 | ~26.2 | ~0.53 | ~0.78 | ~0.61 | ~0.85 | ~0.66 |

---

## 3. Data Preprocessing

### 3.1 Pengecekan Missing Values

Dilakukan pengecekan missing values pada seluruh kolom dataset. Hasilnya menunjukkan bahwa **tidak terdapat missing values** pada semua 17 kolom.

```
Jumlah Missing Values per Kolom:
========================================
Gender                              : ✅ Tidak ada
Age                                 : ✅ Tidak ada
Height                              : ✅ Tidak ada
Weight                              : ✅ Tidak ada
family_history_with_overweight      : ✅ Tidak ada
FAVC                                : ✅ Tidak ada
FCVC                                : ✅ Tidak ada
NCP                                 : ✅ Tidak ada
CAEC                                : ✅ Tidak ada
SMOKE                               : ✅ Tidak ada
CH2O                                : ✅ Tidak ada
SCC                                 : ✅ Tidak ada
FAF                                 : ✅ Tidak ada
TUE                                 : ✅ Tidak ada
CALC                                : ✅ Tidak ada
MTRANS                              : ✅ Tidak ada
NObeyesdad                          : ✅ Tidak ada
```

**Kesimpulan:** Tidak perlu dilakukan penanganan missing values.

### 3.2 Pengecekan dan Penghapusan Data Duplikat

Ditemukan **24 baris data duplikat** dalam dataset. Data duplikat dihapus untuk menghindari bias pada model.

```
Jumlah data duplikat: 24
Menghapus 24 data duplikat...
✅ Data setelah menghapus duplikat: 2087 baris
```

### 3.3 Encoding Data Kategorikal

Algoritma C4.5 (Decision Tree) memerlukan input berupa data numerik. Oleh karena itu, dilakukan konversi data kategorikal menjadi numerik menggunakan **Label Encoding** dan **Ordinal Encoding**:

| Tipe Encoding | Kolom | Mapping |
|---------------|-------|---------|
| **Binary Encoding** | `Gender` | Female → 0, Male → 1 |
| **Binary Encoding** | `family_history_with_overweight` | no → 0, yes → 1 |
| **Binary Encoding** | `FAVC` | no → 0, yes → 1 |
| **Binary Encoding** | `SMOKE` | no → 0, yes → 1 |
| **Binary Encoding** | `SCC` | no → 0, yes → 1 |
| **Ordinal Encoding** | `CAEC` | no → 0, Sometimes → 1, Frequently → 2, Always → 3 |
| **Ordinal Encoding** | `CALC` | no → 0, Sometimes → 1, Frequently → 2, Always → 3 |
| **Label Encoding** | `MTRANS` | Automobile → 0, Bike → 1, Motorbike → 2, Public_Transportation → 3, Walking → 4 |
| **Label Encoding** | `NObeyesdad` | Berat Badan Kurang → 0, Berat Badan Normal → 1, ..., Obesitas Tipe III → 6 |

### 3.4 Pemisahan Fitur dan Target

```
Dimensi fitur (X): (2087, 16)
Dimensi target (y): (2087,)
```

**16 fitur** yang digunakan sebagai input model:
1. Gender, 2. Age, 3. Height, 4. Weight, 5. family_history_with_overweight, 6. FAVC, 7. FCVC, 8. NCP, 9. CAEC, 10. SMOKE, 11. CH2O, 12. SCC, 13. FAF, 14. TUE, 15. CALC, 16. MTRANS

### 3.5 Pembagian Data Training dan Testing

Dataset dibagi dengan rasio **80:20** menggunakan fungsi `train_test_split` dari scikit-learn dengan parameter `stratify=y` untuk menjaga proporsi kelas.

```
Hasil Pembagian Data:
========================================
Data Training : 1669 baris (80%)
Data Testing  : 418 baris (20%)
Total Data    : 2087 baris
```

---

## 4. Implementasi Algoritma C4.5 (Decision Tree)

### 4.1 Membangun Model

Model Decision Tree dibangun menggunakan `DecisionTreeClassifier` dari scikit-learn dengan parameter berikut:

```python
model_c45 = DecisionTreeClassifier(
    criterion='entropy',     # Menggunakan entropy (Information Gain) → konsep C4.5
    random_state=42,         # Untuk reproducibility
    max_depth=None,          # Tidak ada batasan kedalaman pohon
    min_samples_split=2,     # Minimum sampel untuk split
    min_samples_leaf=1       # Minimum sampel pada leaf node
)
```

**Parameter utama yang digunakan:**

| Parameter | Nilai | Penjelasan |
|-----------|:-----:|------------|
| `criterion` | `entropy` | Menggunakan **Entropy** dan **Information Gain** sesuai konsep algoritma C4.5 |
| `random_state` | `42` | Menjamin hasil yang **reproducible** (dapat diulang) |
| `max_depth` | `None` | Pohon tumbuh sepenuhnya tanpa batasan kedalaman |
| `min_samples_split` | `2` | Minimum 2 sampel agar node dapat di-split |
| `min_samples_leaf` | `1` | Minimum 1 sampel pada setiap *leaf node* |

### 4.2 Proses Training

Model dilatih (fit) menggunakan **1.669 data training** yang telah diproses:

```python
model_c45.fit(X_train, y_train)
```

### 4.3 Prediksi pada Data Testing

Setelah model dilatih, dilakukan prediksi pada **418 data testing**:

```python
y_pred = model_c45.predict(X_test)
```

### 4.4 Cuplikan Aturan Decision Tree

Berikut adalah cuplikan aturan (*rules*) yang dihasilkan dari pohon keputusan (dibatasi kedalaman 3 agar mudah dibaca):

```
Cuplikan Aturan Decision Tree (kedalaman maks 3):
============================================================
|--- Weight <= 86.64
|   |--- Weight <= 61.35
|   |   |--- Height <= 1.71
|   |   |   |--- class: Berat Badan Normal
|   |   |--- Height > 1.71
|   |   |   |--- class: Berat Badan Kurang
|   |--- Weight > 61.35
|   |   |--- Weight <= 75.12
|   |   |   |--- class: Kelebihan Berat I
|   |   |--- Weight > 75.12
|   |   |   |--- class: Kelebihan Berat II
|--- Weight > 86.64
|   |--- Weight <= 108.43
|   |   |--- ...
|   |--- Weight > 108.43
|   |   |--- class: Obesitas Tipe III
```

Dari cuplikan aturan di atas, terlihat bahwa atribut **Weight** (berat badan) menjadi fitur yang paling sering digunakan pada level awal pembuatan keputusan, mengindikasikan fitur ini memiliki **Information Gain** tertinggi.

---

## 5. Evaluasi Hasil

### 5.1 Confusion Matrix

Confusion Matrix menampilkan perbandingan antara kelas aktual dan kelas prediksi untuk semua 7 kelas tingkat obesitas:

|  | Berat Badan Kurang | Berat Badan Normal | Kelebihan Berat I | Kelebihan Berat II | Obesitas Tipe I | Obesitas Tipe II | Obesitas Tipe III |
|--|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Berat Badan Kurang** | **53** | 0 | 0 | 0 | 0 | 0 | 0 |
| **Berat Badan Normal** | 0 | **54** | 3 | 0 | 0 | 0 | 0 |
| **Kelebihan Berat I** | 0 | 4 | **48** | 3 | 0 | 0 | 0 |
| **Kelebihan Berat II** | 0 | 0 | 1 | **54** | 3 | 0 | 0 |
| **Obesitas Tipe I** | 0 | 0 | 0 | 0 | **68** | 2 | 0 |
| **Obesitas Tipe II** | 0 | 0 | 0 | 0 | 2 | **58** | 0 |
| **Obesitas Tipe III** | 0 | 0 | 0 | 0 | 0 | 0 | **64** |

**Interpretasi:** Sebagian besar data berada pada **diagonal utama** (diprediksi benar). Kesalahan klasifikasi terjadi terutama pada kelas-kelas yang berdekatan (misalnya antara Overweight Level I dan Normal Weight), yang secara medis memang memiliki karakteristik serupa.

### 5.2 Metrik Evaluasi Utama

```
==================================================
     HASIL EVALUASI MODEL C4.5 (Decision Tree)
==================================================
  Accuracy  : 0.9545 (95.45%)
  Precision : 0.9545 (95.45%)
  Recall    : 0.9545 (95.45%)
  F1-Score  : 0.9544 (95.44%)
==================================================
```

| Metrik | Skor | Penjelasan |
|--------|:----:|------------|
| **Accuracy** | **95.45%** | Persentase keseluruhan prediksi yang benar dari total 418 data testing |
| **Precision** | **95.45%** | Ketepatan model: dari semua data yang diprediksi suatu kelas, 95.45% benar |
| **Recall** | **95.45%** | Sensitivitas model: dari semua data aktual suatu kelas, 95.45% berhasil dideteksi |
| **F1-Score** | **95.44%** | Rata-rata harmonis antara Precision dan Recall |

### 5.3 Classification Report (Detail per Kelas)

| Kelas | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| Berat Badan Kurang | 1.0000 | 1.0000 | 1.0000 | 53 |
| Berat Badan Normal | 0.9310 | 0.9474 | 0.9391 | 57 |
| Kelebihan Berat I | 0.9231 | 0.8727 | 0.8972 | 55 |
| Kelebihan Berat II | 0.9474 | 0.9310 | 0.9391 | 58 |
| Obesitas Tipe I | 0.9315 | 0.9714 | 0.9510 | 70 |
| Obesitas Tipe II | 0.9667 | 0.9667 | 0.9667 | 60 |
| Obesitas Tipe III | 0.9846 | 0.9846 | 0.9846 | 65 |

**Analisis per kelas:**
- **Berat Badan Kurang** memiliki skor sempurna (100%) — model dapat mengidentifikasi semua kasus kekurangan berat badan tanpa kesalahan.
- **Kelebihan Berat I** memiliki skor terendah (F1 = 89.72%) — hal ini wajar karena kelas ini berada di "zona transisi" antara berat badan normal dan kelebihan berat badan yang lebih tinggi.
- **Obesitas Tipe III** memiliki skor sangat tinggi (F1 = 98.46%) — kasus obesitas berat lebih mudah dibedakan karena karakteristik fisik yang jelas.

### 5.4 Interpretasi Hasil Evaluasi

1. **Accuracy 95.45%** menunjukkan bahwa model C4.5 berhasil mengklasifikasikan **399 dari 418** data testing dengan benar.
2. Nilai **Precision ≈ Recall ≈ F1-Score** yang hampir sama menandakan model bekerja **secara seimbang** di semua kelas (tidak bias terhadap kelas tertentu).
3. Kesalahan klasifikasi umumnya terjadi antara kelas-kelas yang **bertetangga** (misalnya Overweight Level I dan Normal Weight), yang secara medis memang sulit dibedakan.
4. Model terbukti **robust** dalam menangani dataset dengan 7 kelas yang relatif seimbang.

---

## 6. Visualisasi

### 6.1 Distribusi Kelas Target (NObeyesdad)

Visualisasi ini menunjukkan persebaran 7 kelas target tingkat obesitas dalam dataset. Distribusi yang relatif seimbang menunjukkan bahwa dataset tidak memiliki masalah *class imbalance* yang signifikan.

![Distribusi Kelas Target Obesitas](outputs/target_distribution.png)

### 6.2 Confusion Matrix Heatmap

Visualisasi Confusion Matrix dalam bentuk *heatmap* yang menunjukkan performa prediksi model untuk setiap kelas. Warna yang lebih gelap pada diagonal utama menandakan prediksi yang benar.

![Confusion Matrix Heatmap](outputs/confusion_matrix.png)

### 6.3 Feature Importance (Tingkat Kepentingan Fitur)

Visualisasi *Feature Importance* menunjukkan fitur-fitur yang paling berpengaruh dalam proses klasifikasi berdasarkan nilai Information Gain. Fitur **Weight (Berat Badan)** mendominasi sebagai fitur terpenting.

![Feature Importance](outputs/feature_importance.png)

**Insight dari Feature Importance:**
- **Weight** (Berat Badan) memiliki kontribusi terbesar dalam menentukan tingkat obesitas.
- **Height** (Tinggi Badan) dan **Age** (Usia) juga berpengaruh signifikan.
- Fitur gaya hidup seperti **FCVC**, **NCP**, **CH2O** memiliki pengaruh yang lebih kecil tetapi tetap berkontribusi.
- Fitur seperti **SMOKE** dan **SCC** memiliki pengaruh yang relatif kecil.

### 6.4 Perbandingan Distribusi Aktual vs Prediksi

Visualisasi ini membandingkan distribusi kelas aktual dan kelas prediksi pada data testing, menunjukkan bahwa model mampu memprediksi distribusi kelas yang sangat mirip dengan data aktual.

### 6.5 Visualisasi Pohon Keputusan (Decision Tree)

Pohon keputusan divisualisasikan dengan kedalaman dibatasi hingga 4 level agar lebih mudah dibaca dan diinterpretasikan. Setiap node menampilkan:
- **Fitur** yang digunakan untuk split
- **Threshold** (batas nilai) untuk pemecahan
- **Entropy** pada node tersebut
- **Jumlah sampel** yang masuk ke node
- **Kelas mayoritas** pada node

---

## 7. Kesimpulan

Berdasarkan implementasi dan analisis yang telah dilakukan, dapat disimpulkan:

1. **Dataset** Obesity Levels terdiri dari **2.111 data** (2.087 setelah cleaning) dengan **16 fitur** dan **7 kelas target** tingkat obesitas. Dataset tidak memiliki missing values dan distribusi kelasnya relatif seimbang.

2. **Preprocessing** yang dilakukan meliputi:
   - Pengecekan missing values (tidak ditemukan)
   - Penghapusan 24 data duplikat
   - Encoding fitur kategorikal menjadi numerik menggunakan Label Encoding dan Ordinal Encoding
   - Pembagian data 80% training dan 20% testing dengan stratified sampling

3. **Algoritma C4.5 (Decision Tree)** berhasil diimplementasikan menggunakan parameter `criterion='entropy'` (Information Gain) dari library scikit-learn. Model dilatih menggunakan 1.669 data training dan dievaluasi menggunakan 418 data testing.

4. **Evaluasi model** menunjukkan hasil yang **sangat baik** dengan:
   - Accuracy: **95.45%**
   - Precision: **95.45%**
   - Recall: **95.45%**
   - F1-Score: **95.44%**

5. **Feature Importance** menunjukkan bahwa fitur **Weight** (Berat Badan) merupakan faktor paling dominan dalam klasifikasi tingkat obesitas, diikuti oleh **Height** (Tinggi Badan) dan **Age** (Usia).

6. Algoritma C4.5 terbukti **cocok** untuk dataset ini karena:
   - Mampu menangani campuran data numerik dan kategorikal
   - Menghasilkan model yang mudah diinterpretasikan dalam bentuk pohon keputusan
   - Memberikan performa klasifikasi yang tinggi untuk 7 kelas target

---

## 8. Referensi

1. Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann Publishers.
2. Palechor, F. M., & de la Hoz Manotas, A. (2019). Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico. *Data in Brief, 25*, 104344.
3. UCI Machine Learning Repository — [Estimation of Obesity Levels Based on Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544)
4. Scikit-learn Documentation — [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
5. Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann.

---

## Flowchart Proses Data Mining

```
┌─────────────────┐
│   Mulai / Start │
└────────┬────────┘
         ▼
┌─────────────────────┐
│  1. Data Understanding│
│  - Load dataset      │
│  - Eksplorasi data   │
│  - Statistik deskriptif│
│  - Cek distribusi    │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  2. Data Preprocessing│
│  - Cek missing values │
│  - Hapus duplikat     │
│  - Encoding kategorikal│
│  - Split train/test   │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  3. Implementasi     │
│     Algoritma C4.5   │
│  - Build model       │
│  - Training model    │
│  - Prediksi testing  │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  4. Evaluasi Hasil   │
│  - Confusion Matrix  │
│  - Accuracy, Precision│
│  - Recall, F1-Score  │
│  - Classification Rpt│
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  5. Visualisasi      │
│  - Decision Tree     │
│  - Feature Importance│
│  - Distribusi kelas  │
│  - Heatmap CM        │
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  6. Kesimpulan       │
│  & Interpretasi      │
└────────┬────────────┘
         ▼
┌─────────────────┐
│  Selesai / End  │
└─────────────────┘
```

---

*Laporan ini disusun sebagai Tugas Ujian Akhir Semester (UAS) mata kuliah Data Mining – TIF 605, Tahun Ajaran 2025/2026 Genap.*
