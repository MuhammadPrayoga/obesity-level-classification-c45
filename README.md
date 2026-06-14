# 🎓 Klasifikasi Tingkat Obesitas dengan Algoritma C4.5 (Decision Tree)

> **Tugas UAS Data Mining (TIF 605)** — Tahun Ajaran 2025/2026 Genap  
> Dosen: Dr. Ir. Ananto Tri Sasongko, M.Sc.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=google-colab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📌 Deskripsi Proyek

Proyek ini mengimplementasikan **algoritma C4.5 (Decision Tree)** untuk mengklasifikasikan **tingkat obesitas** seseorang ke dalam **7 kelas** berdasarkan kebiasaan makan dan kondisi fisik. Algoritma C4.5 menggunakan konsep **Entropy** dan **Information Gain** untuk membangun pohon keputusan secara rekursif.

### Hasil Evaluasi Model

| Metrik | Skor |
|:------:|:----:|
| **Accuracy** | **97.37%** |
| **Precision** | **97.42%** |
| **Recall** | **97.37%** |
| **F1-Score** | **97.36%** |

---

## 📊 Dataset

**Obesity Levels Dataset** ([UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition))

Dataset berisi estimasi tingkat obesitas berdasarkan kebiasaan makan dan kondisi fisik dari penduduk Meksiko, Peru, dan Kolombia.

- **Jumlah data**: 2.111 baris (2.087 setelah menghapus duplikat)
- **Jumlah fitur**: 16 fitur + 1 variabel target
- **Variabel target**: `NObeyesdad` (7 kelas tingkat obesitas)

### Fitur Dataset

| No | Fitur | Tipe | Keterangan |
|:--:|-------|:----:|------------|
| 1 | `Gender` | Kategorikal | Jenis kelamin (Male/Female) |
| 2 | `Age` | Numerik | Usia (dalam tahun) |
| 3 | `Height` | Numerik | Tinggi badan (dalam meter) |
| 4 | `Weight` | Numerik | Berat badan (dalam kg) |
| 5 | `family_history_with_overweight` | Kategorikal | Riwayat keluarga overweight (yes/no) |
| 6 | `FAVC` | Kategorikal | Konsumsi makanan berkalori tinggi (yes/no) |
| 7 | `FCVC` | Numerik | Frekuensi konsumsi sayuran (1-3) |
| 8 | `NCP` | Numerik | Jumlah makan utama per hari (1-4) |
| 9 | `CAEC` | Kategorikal | Konsumsi makanan di antara waktu makan |
| 10 | `SMOKE` | Kategorikal | Kebiasaan merokok (yes/no) |
| 11 | `CH2O` | Numerik | Konsumsi air per hari (1-3 liter) |
| 12 | `SCC` | Kategorikal | Monitoring konsumsi kalori (yes/no) |
| 13 | `FAF` | Numerik | Frekuensi aktivitas fisik (0-3) |
| 14 | `TUE` | Numerik | Waktu penggunaan gadget (0-2 jam) |
| 15 | `CALC` | Kategorikal | Frekuensi konsumsi alkohol |
| 16 | `MTRANS` | Kategorikal | Moda transportasi utama |

### Kelas Target (`NObeyesdad`)

| No | Kelas | Keterangan |
|:--:|-------|------------|
| 1 | `Insufficient_Weight` | Berat badan kurang |
| 2 | `Normal_Weight` | Berat badan normal |
| 3 | `Overweight_Level_I` | Kelebihan berat badan tingkat I |
| 4 | `Overweight_Level_II` | Kelebihan berat badan tingkat II |
| 5 | `Obesity_Type_I` | Obesitas tipe I |
| 6 | `Obesity_Type_II` | Obesitas tipe II |
| 7 | `Obesity_Type_III` | Obesitas tipe III |

---

## 🧠 Algoritma C4.5 (Decision Tree)

Algoritma **C4.5** dikembangkan oleh **Ross Quinlan** sebagai penyempurnaan dari algoritma ID3. C4.5 membangun pohon keputusan menggunakan:

1. **Entropy** — mengukur tingkat ketidakpastian data:

$$Entropy(S) = -\sum_{i=1}^{n} p_i \cdot \log_2(p_i)$$

2. **Information Gain** — mengukur pengurangan entropy setelah split:

$$Gain(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} \cdot Entropy(S_v)$$

3. Atribut dengan **Information Gain tertinggi** dipilih sebagai node keputusan.
4. Proses diulang secara **rekursif** hingga seluruh data terklasifikasi.

Implementasi menggunakan `DecisionTreeClassifier(criterion='entropy')` dari scikit-learn.

---

## 🔄 Metodologi

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────┐    ┌────────────┐
│  Dataset │───▶│    Data      │───▶│    Data      │───▶│ Algoritma│───▶│ Evaluasi │───▶│ Visualisasi│
│ (2111    │    │Understanding │    │Preprocessing │    │   C4.5   │    │  Model   │    │   Hasil    │
│  data)   │    │              │    │              │    │          │    │          │    │            │
└──────────┘    └──────────────┘    └──────────────┘    └──────────┘    └──────────┘    └────────────┘
```

### Tahap-tahap:

1. **Data Understanding** — Eksplorasi dataset, analisis statistik deskriptif, distribusi kelas target
2. **Data Preprocessing** — Hapus 24 data duplikat, encoding fitur kategorikal (Label + Ordinal Encoding), split 80:20
3. **Implementasi C4.5** — Bangun model `DecisionTreeClassifier(criterion='entropy', random_state=42)`
4. **Evaluasi** — Confusion Matrix, Accuracy, Precision, Recall, F1-Score
5. **Visualisasi** — 5 grafik: Confusion Matrix Heatmap, Decision Tree, Feature Importance, Distribusi Target, Metrik per Kelas

---

## 📈 Hasil Analisis

### Evaluasi Model

```
                     precision    recall  f1-score   support

Insufficient_Weight     1.0000    1.0000    1.0000        53
      Normal_Weight     0.9492    0.9825    0.9655        57
     Obesity_Type_I     0.9710    0.9571    0.9640        70
    Obesity_Type_II     0.9836    1.0000    0.9917        60
   Obesity_Type_III     1.0000    0.9846    0.9922        65
 Overweight_Level_I     0.9804    0.9091    0.9434        55
Overweight_Level_II     0.9344    0.9828    0.9580        58

           accuracy                         0.9737       418
          macro avg     0.9741    0.9737    0.9736       418
       weighted avg     0.9742    0.9737    0.9736       418
```

### Top 5 Feature Importance

| Rank | Fitur | Importance |
|:----:|-------|:----------:|
| 1 | **Weight** (Berat Badan) | 0.5832 |
| 2 | **Height** (Tinggi Badan) | 0.2036 |
| 3 | **Gender** (Jenis Kelamin) | 0.1274 |
| 4 | **Age** (Usia) | 0.0298 |
| 5 | **CH2O** (Konsumsi Air) | 0.0218 |

> **Insight**: Berat badan (Weight) merupakan fitur paling dominan dengan kontribusi 58.3%, mengkonfirmasi bahwa faktor fisik utama adalah penentu terkuat klasifikasi obesitas.

---

## 📁 Struktur Proyek

```
obesity-level-classification-c45/
├── 📓 312310569_Muhammad Prayoga Putra Mahardhika_UAS_DataMining.ipynb   # Notebook utama
├── 📊 ObesityDataSet_raw_and_data_sinthetic.csv                         # Dataset
├── 🖼️ 312310569_Muhammad Prayoga Putra Mahardhika_UAS_DataMining_Poster.html  # Poster A3
├── 📄 README.md                                                          # Dokumentasi
├── 📄 LICENSE                                                            # Lisensi MIT
└── 📄 .gitignore                                                         # Git ignore rules
```

---

## 🚀 Cara Menjalankan

### Prasyarat

- Python 3.10+
- Google Colab (direkomendasikan) atau Jupyter Notebook

### Menjalankan di Google Colab

1. Buka [Google Colab](https://colab.google/)
2. Upload file notebook (`.ipynb`) dan dataset (`.csv`)
3. Jalankan semua sel secara berurutan

### Menjalankan Secara Lokal

```bash
# Clone repository
git clone https://github.com/<username>/obesity-level-classification-c45.git
cd obesity-level-classification-c45

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Jalankan notebook
jupyter notebook "312310569_Muhammad Prayoga Putra Mahardhika_UAS_DataMining.ipynb"
```

### Dependencies

| Library | Versi | Kegunaan |
|---------|:-----:|----------|
| pandas | ≥2.0 | Manipulasi data |
| numpy | ≥1.22 | Komputasi numerik |
| scikit-learn | ≥1.0 | Algoritma ML & evaluasi |
| matplotlib | ≥3.5 | Visualisasi grafik |
| seaborn | ≥0.12 | Visualisasi statistik |

---

## 📝 Poster Ilmiah

Poster A3 dalam format HTML dapat dibuka di browser dan dicetak sebagai PDF:

1. Buka file `312310569_Muhammad Prayoga Putra Mahardhika_UAS_DataMining_Poster.html` di browser
2. Tekan `Ctrl + P` → atur ukuran kertas **A3**, orientasi **Portrait**
3. Centang **Background graphics** → Save as PDF

---

## 👤 Informasi

| | |
|---|---|
| **Nama** | Muhammad Prayoga Putra Mahardhika |
| **NIM** | 312310569 |
| **Mata Kuliah** | Data Mining (TIF 605) |
| **Dosen** | Dr. Ir. Ananto Tri Sasongko, M.Sc. |
| **Tahun Ajaran** | 2025/2026 Genap |

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---

<div align="center">
  <sub>Made with ❤️ for UAS Data Mining 2025/2026</sub>
</div>
