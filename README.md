# Klasifikasi Motif Batik dengan CNN (MobileNetV2) & MLflow

Proyek ini bertujuan untuk melakukan **klasifikasi gambar motif batik** menggunakan arsitektur **MobileNetV2** sebagai _feature extractor_ dan lapisan tambahan untuk prediksi multi-kelas. Semua eksperimen, parameter, dan metrik dicatat menggunakan **MLflow** untuk memudahkan pelacakan dan reproduksi.

🌐 **Demo Aplikasi (Hugging Face Spaces):**  
👉 https://huggingface.co/spaces/Nabiilah-Putri/Klasifikasi-Motif-Batik

---

## 🎯 Tujuan Proyek
1. Membangun model deep learning untuk klasifikasi motif batik otomatis
2. Melestarikan warisan budaya melalui teknologi AI
3. Membuat sistem yang mudah diakses via web interface
4. Dokumentasi reproducible dengan MLflow

---

## 🏷️ Batik Classifications
Model dapat mengklasifikasikan 14 motif batik tradisional Indonesia:

| No | Nama Motif | Asal Daerah | Karakteristik |
|----|------------|-------------|---------------|
| 1 | Barong | Bali | Motif mitologis berupa singa dengan mahkota |
| 2 | Celup | Berbagai daerah | Teknik pewarnaan dengan pencelupan |
| 3 | Cendrawasih | Papua | Burung cendrawasih dengan warna cerah |
| 4 | Ceplok | Jawa Tengah | Motif geometris berulang seperti bunga |
| 5 | Dayak | Kalimantan | Pola etnik dengan simbol budaya Dayak |
| 6 | Insang | Kalimantan | Motif seperti insang ikan dengan pola lengkung |
| 7 | Kawung | Jawa | Pola geometris bulatan seperti buah kawung |
| 8 | Lontara | Sulawesi Selatan | Berbasis aksara Lontara dengan pola simetris |
| 9 | Mataketeran | Sumbawa | Mata ketengan dengan pola detail rumit |
| 10 | Megamendung | Cirebon | Awan berwarna cerah dengan gradasi warna |
| 11 | Ondel-ondel | Betawi | Boneka besar Betami dengan wajah khas |
| 12 | Parang | Jawa Tengah | Motif diagonal seperti pedang (parang) |
| 13 | Pring | Yogyakarta | Motif bambu dengan pola vertikal |
| 14 | Rumah Minang | Sumatera Barat | Pola arsitektur rumah gadang dengan atap khas |

---

## ✅ Fitur Utama
* **Model**: MobileNetV2 (pretrained ImageNet) dengan lapisan GlobalAveragePooling, Dropout, dan Dense untuk klasifikasi.
* **Pelacakan Eksperimen**: MLflow digunakan untuk mencatat parameter (learning rate, jumlah epoch), metrik per epoch (akurasi dan loss), serta menyimpan model sebagai artifact.
* **Pipeline Data**: Menggunakan generator berbasis `ImageDataGenerator` untuk normalisasi dan manajemen batch dataset.
* **Evaluasi Lengkap**: Termasuk confusion matrix, classification report, dan pengukuran waktu inferensi rata-rata per sampel.
* **Penyimpanan Model**: Format `.keras` dan _weights_ `.h5` untuk kemudahan _deployment_ dan _checkpointing_.

---

## 🏗️ Arsitektur Sistem

```
1. Data Collection
   ↓
2. Preprocessing (resize 224x224, normalization, augmentation)
   ↓
3. Data Splitting (Train 80%, Validation 10%, Test 10%)
   ↓
4. Model Building (MobileNetV2 base + custom layers)
   ↓
5. Model Training (with MLflow tracking)
   ↓
6. Model Evaluation & Validation
   ↓
7. Model Export (.h5 format)
   ↓
8. Deployment (Hugging Face Spaces)
```

---

## 📑 Struktur Direktori

```
.
├─ README.md
├─ Kode/
│  └─ klasifikasi_batik.ipynb
├─ src/
│  ├─ app.py
├─ requirements.txt
└─ artifacts/
   ├─ best_model_MobileNetV2.h5
   ├─ model_batik_mobilenetv2.keras
   └─ mlruns/

```

---

## 📂 Dataset
Dataset tersedia di Google Drive: **(https://www.kaggle.com/datasets/buyungsaloka/motif-batik-dataset)**

### **Karakteristik Dataset:**
- **Total Gambar**: ~1,400 gambar (14 kelas × ~100 gambar/kelas)
- **Format**: JPG/PNG
- **Resolusi**: Bervariasi (di-resize ke 224×224 untuk training)

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

---

## 🚀 Menjalankan Aplikasi di Hugging Face   

#### Simpan model dalam format Keras
```
model_MobileNetV2.save("model_batik_mobilenetv2.keras")
```

#### Simpan hanya weights
```
FILENAME = "model_batik_mobilenetv2.weights.h5"
model_MobileNetV2.save_weights(FILENAME)
```

#### Instalasi pustaka
```
!pip install huggingface-hub -q
```

#### Login ke Hugging Face Hub
```
from huggingface_hub import notebook_login
notebook_login()  # Masukkan Access Token Hugging Face
```

#### Tentukan nama repository
```
REPO_ID = "Nabiilah-Putri/Batik_Classification"
MODEL_FILENAME = "model_batik_mobilenetv2.keras"
CLASSES_FILENAME = "class_names.json"
```

#### Simpan model
```
model_MobileNetV2.save(MODEL_FILENAME)
print(f"Model disimpan sebagai: {MODEL_FILENAME}")
```

#### Simpan daftar kelas ke JSON
```
class_indices = test_generator.class_indices
index_to_class = {v: k for k, v in class_indices.items()}
with open(CLASSES_FILENAME, "w") as f:
    json.dump(index_to_class, f)
print(f"Daftar kelas disimpan sebagai: {CLASSES_FILENAME}")
```

#### Buat Repository baru (jika belum ada)
```
create_repo(REPO_ID, exist_ok=True, repo_type="model")
api = HfApi()
```

#### Upload file model
```
api.upload_file(
    path_or_fileobj=MODEL_FILENAME,
    path_in_repo=MODEL_FILENAME,
    repo_id=REPO_ID,
    repo_type="model",
)
print(f"File {MODEL_FILENAME} berhasil diunggah.")
```

#### Upload file class names
```
api.upload_file(
    path_or_fileobj=CLASSES_FILENAME,
    path_in_repo=CLASSES_FILENAME,
    repo_id=REPO_ID,
    repo_type="model",
)
print(f"File {CLASSES_FILENAME} berhasil diunggah.")
```

#### menulis requirements.txt
```
%%writefile requirements.txt
tensorflow
gradio
numpy
pillow
huggingface-hub
```
---

## Hugging Face
Link: https://huggingface.co/spaces/Nabiilah-Putri/Klasifikasi-Motif-Batik

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
