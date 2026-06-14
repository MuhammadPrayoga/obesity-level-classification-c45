# 🎓 Klasifikasi Tingkat Obesitas dengan Algoritma C4.5 (Decision Tree)

> **Machine Learning Classification Portfolio Project**  
> Created by Muhammad Prayoga Putra Mahardhika

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter_Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📌 Deskripsi Proyek

Proyek portofolio ini mengimplementasikan **algoritma C4.5 (Decision Tree)** untuk mengklasifikasikan **tingkat obesitas** seseorang ke dalam **7 kelas** berdasarkan kebiasaan makan dan kondisi fisik. Algoritma C4.5 menggunakan konsep **Entropy** dan **Information Gain** untuk membangun pohon keputusan secara rekursif, menjadikannya salah satu model klasifikasi yang paling mudah diinterpretasikan (white-box model).

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

- **Jumlah data**: 2.111 baris (2.087 setelah proses data cleaning)
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

Implementasi proyek ini menggunakan `DecisionTreeClassifier(criterion='entropy')` dari *scikit-learn* untuk mencerminkan mekanisme seleksi atribut dari algoritma C4.5.

---

## 🔄 Metodologi

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────┐    ┌────────────┐
│  Dataset │───▶│    Data      │───▶│    Data      │───▶│ Algoritma│───▶│ Evaluasi │───▶│ Visualisasi│
│ (2111    │    │Understanding │    │Preprocessing │    │   C4.5   │    │  Model   │    │   Hasil    │
│  data)   │    │              │    │              │    │          │    │          │    │            │
└──────────┘    └──────────────┘    └──────────────┘    └──────────┘    └──────────┘    └────────────┘
```

### Tahapan:

1. **Data Understanding** — Eksplorasi dataset, analisis statistik deskriptif, visualisasi distribusi kelas target.
2. **Data Preprocessing** — Data cleaning (menghapus duplikat), feature encoding (Label Encoding & Ordinal Encoding), dan splitting dataset (80% Train, 20% Test dengan parameter `stratify=y`).
3. **Model Implementation** — Membangun model Decision Tree menggunakan parameter optimal untuk reproducibility.
4. **Model Evaluation** — Mengukur performa menggunakan metrik Confusion Matrix, Accuracy, Precision, Recall, dan F1-Score.
5. **Data Visualization** — Membangun grafik interaktif (Heatmap, Feature Importance bar chart, dll).

---

## 📈 Hasil Analisis

### Evaluasi Model (Classification Report)

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

| Rank | Fitur | Importance Score |
|:----:|-------|:----------------:|
| 1 | **Weight** (Berat Badan) | 0.5832 |
| 2 | **Height** (Tinggi Badan) | 0.2036 |
| 3 | **Gender** (Jenis Kelamin) | 0.1274 |
| 4 | **Age** (Usia) | 0.0298 |
| 5 | **CH2O** (Konsumsi Air) | 0.0218 |

> **Business Insight**: Fitur berat badan (Weight) memiliki tingkat kepentingan lebih dari 58% dalam membuat keputusan klasifikasi. Hal ini mengkonfirmasi secara data-driven bahwa faktor fisik dasar merupakan penentu terkuat tingkat obesitas dibandingkan dengan kebiasaan sekunder seperti waktu penggunaan gadget (TUE).

---

## 📁 Struktur Repository

```
obesity-level-classification-c45/
├── 📓 Obesity_Level_Classification_C45.ipynb            # Notebook utama berisi seluruh kode
├── 📊 ObesityDataSet_raw_and_data_sinthetic.csv         # Dataset
├── 🖼️ Obesity_Level_Classification_Poster.html          # Visualisasi poster presentasi proyek
├── 📄 README.md                                         # Dokumentasi proyek
├── 📄 LICENSE                                           # Lisensi MIT
└── 📄 .gitignore                                        # Git ignore rules
```

---

## 🚀 Cara Menjalankan Project

### Prasyarat

- Python 3.10+
- Jupyter Notebook / JupyterLab

### Menjalankan Secara Lokal

```bash
# Clone repository
git clone https://github.com/MuhammadPrayoga/obesity-level-classification-c45.git
cd obesity-level-classification-c45

# Install dependencies yang dibutuhkan
pip install pandas numpy scikit-learn matplotlib seaborn

# Buka dan jalankan notebook
jupyter notebook "Obesity_Level_Classification_C45.ipynb"
```

### Teknologi Utama yang Digunakan

| Library | Kegunaan Utama |
|---------|----------------|
| `pandas` | Manipulasi dan eksplorasi data tabular |
| `numpy` | Operasi aljabar linear dan komputasi numerik |
| `scikit-learn` | Implementasi algoritma Decision Tree dan kalkulasi metrik evaluasi |
| `matplotlib` & `seaborn` | Visualisasi statistik dan grafis (Heatmaps, Bar charts) |

---

## 👤 Author

**Muhammad Prayoga Putra Mahardhika**
- GitHub: [@MuhammadPrayoga](https://github.com/MuhammadPrayoga)

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).
