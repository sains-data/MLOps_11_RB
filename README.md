# Klasifikasi Motif Batik dengan CNN (MobileNetV2) & MLflow

Proyek ini bertujuan untuk melakukan **klasifikasi gambar motif batik** menggunakan arsitektur **MobileNetV2** sebagai _feature extractor_ dan lapisan tambahan untuk prediksi multi-kelas. Semua eksperimen, parameter, dan metrik dicatat menggunakan **MLflow** untuk memudahkan pelacakan dan reproduksi.

---

## ✅ Fitur Utama
* **Model**: MobileNetV2 (pretrained ImageNet) dengan lapisan GlobalAveragePooling, Dropout, dan Dense untuk klasifikasi.
* **Pelacakan Eksperimen**: MLflow digunakan untuk mencatat parameter (learning rate, jumlah epoch), metrik per epoch (akurasi dan loss), serta menyimpan model sebagai artifact.
* **Pipeline Data**: Menggunakan generator berbasis `ImageDataGenerator` untuk normalisasi dan manajemen batch dataset.
* **Evaluasi Lengkap**: Termasuk confusion matrix, classification report, dan pengukuran waktu inferensi rata-rata per sampel.
* **Penyimpanan Model**: Format `.keras` dan _weights_ `.h5` untuk kemudahan _deployment_ dan _checkpointing_.

---

## 📂 Dataset
Dataset tersedia di Google Drive: **(https://www.kaggle.com/datasets/buyungsaloka/motif-batik-dataset)**

Struktur dataset:

```

archive/
├─ train/
│  ├─ kelas_1/
│  ├─ kelas_2/
│  └─ ...
├─ val/
│  ├─ kelas_1/
│  ├─ kelas_2/
│  └─ ...
└─ test/
├─ kelas_1/
├─ kelas_2/
└─ ...

```

## 📑 Struktur Direktori

```
.
├─ README.md
├─ Kode/
│  └─ klasifikasi_batik.ipynb
├─ src/
│  ├─ data.py           # utilitas data & eksplorasi
│  ├─ model.py          # arsitektur & compile
│  ├─ train.py          # training + MLflow logging
│  ├─ eval.py           # evaluasi & visualisasi
│  └─ predict.py        # inferensi gambar
├─ requirements.txt
└─ artifacts/
   ├─ best_model_MobileNetV2.h5
   ├─ model_batik_mobilenetv2.keras
   └─ mlruns/

```

---

## 🔧 Persyaratan
* Python 3.8+
* TensorFlow/Keras
* scikit-learn
* pandas, numpy, matplotlib, seaborn
* mlflow

## 👥 Kredit Tim
Kelompok 11 — MLOps RB :
* Amalia Melani Putri
* Azizah Kusumah Putri
* Fayyaza Aqila Syafitri Achjar
* Nabiilah Putri Karnaia
