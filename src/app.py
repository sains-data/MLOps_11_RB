%%writefile app.py

import tensorflow as tf
import numpy as np
import json
import gradio as gr
from tensorflow.keras.preprocessing import image
from huggingface_hub import hf_hub_download
from PIL import Image

# Konfigurasi
MODEL_REPO_ID = "Nabiilah-Putri/Batik_Classification"
WEIGHTS_FILENAME = "model_batik_mobilenetv2.weights.h5"
CLASSES_FILENAME = "class_names.json"

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 14

# Deskripsi Jenis Batik
BATIK_DESCRIPTIONS = {
    "lontara": (
        "*Asal*: Sulawesi Selatan\n"
        "*Ciri*: Motif terinspirasi dari aksara Lontara dengan garis-garis geometris khas.\n"
        "*Makna*: Pengetahuan, identitas budaya, dan kearifan lokal.\n"
        "*Konteks Pakai*: Busana adat dan kegiatan budaya."
    ),
    "metaketeran": (
        "*Asal*: Bali\n"
        "*Ciri*: Pola geometris berulang dengan susunan simetris dan ritmis.\n"
        "*Makna*: Keseimbangan, keteraturan, dan harmoni.\n"
        "*Konteks Pakai*: Kain adat dan busana tradisional Bali."
    ),
    "megamendung": (
        "*Asal*: Cirebon\n"
        "*Ciri*: Motif awan berlapis dengan garis tebal dan gradasi warna khas.\n"
        "*Makna*: Kesabaran, ketenangan, dan kebijaksanaan.\n"
        "*Konteks Pakai*: Kemeja, gaun, dan kerajinan batik."
    ),
    "ondel-ondel": (
        "*Asal*: Jakarta (Betawi)\n"
        "*Ciri*: Motif figur ondel-ondel dan ornamen budaya Betawi.\n"
        "*Makna*: Identitas budaya, perlindungan, dan keterbukaan.\n"
        "*Konteks Pakai*: Acara budaya dan busana kasual."
    ),
    "parang": (
        "*Asal*: Jawa (Yogyakarta dan Surakarta)\n"
        "*Ciri*: Pola diagonal berulang menyerupai parang atau ombak.\n"
        "*Makna*: Keteguhan, keberanian, dan pantang menyerah.\n"
        "*Konteks Pakai*: Busana adat dan acara resmi."
    ),
    "pring": (
        "*Asal*: Jawa\n"
        "*Ciri*: Motif bambu (pring) dengan susunan vertikal atau diagonal.\n"
        "*Makna*: Kesederhanaan, keteguhan, dan kebermanfaatan.\n"
        "*Konteks Pakai*: Busana harian dan semi-formal."
    ),
    "rumah-minang": (
        "*Asal*: Sumatera Barat\n"
        "*Ciri*: Motif rumah gadang dengan atap gonjong yang khas.\n"
        "*Makna*: Kebersamaan, adat, dan kekuatan keluarga.\n"
        "*Konteks Pakai*: Busana adat Minangkabau."
    ),
    "celup": (
        "*Asal*: Nusantara\n"
        "*Ciri*: Motif sederhana hasil teknik celup atau ikat warna.\n"
        "*Makna*: Kesederhanaan dan kealamian.\n"
        "*Konteks Pakai*: Busana santai dan kain tradisional."
    ),
    "cendrawasih": (
        "*Asal*: Papua\n"
        "*Ciri*: Ornamen burung cendrawasih dengan bentuk anggun dan dekoratif.\n"
        "*Makna*: Keindahan, kemuliaan, dan kemakmuran.\n"
        "*Konteks Pakai*: Busana pesta dan seni batik."
    ),
    "ceplok": (
        "*Asal*: Jawa\n"
        "*Ciri*: Pola geometris berulang seperti lingkaran atau kotak.\n"
        "*Makna*: Keteraturan, keseimbangan, dan keteguhan.\n"
        "*Konteks Pakai*: Busana formal dan kain tradisional."
    ),
    "dayak": (
        "*Asal*: Kalimantan\n"
        "*Ciri*: Motif etnik Dayak seperti aso dan ornamen alam.\n"
        "*Makna*: Hubungan manusia dengan alam dan leluhur.\n"
        "*Konteks Pakai*: Busana etnik dan kegiatan budaya."
    ),
    "insang": (
        "*Asal*: Kalimantan Barat\n"
        "*Ciri*: Motif menyerupai insang ikan dengan garis simetris.\n"
        "*Makna*: Kehidupan, keberlanjutan, dan keseimbangan alam.\n"
        "*Konteks Pakai*: Busana adat dan resmi daerah."
    ),
    "kawung": (
        "*Asal*: Jawa\n"
        "*Ciri*: Motif elips menyerupai buah aren yang tersusun simetris.\n"
        "*Makna*: Kesucian, pengendalian diri, dan keseimbangan.\n"
        "*Konteks Pakai*: Busana formal maupun non-formal."
    ),
    "barong": (
        "*Asal*: Bali\n"
        "*Ciri*: Motif Barong dengan bentuk ekspresif dan dekoratif.\n"
        "*Makna*: Perlindungan, kekuatan, dan keseimbangan antara baik dan buruk.\n"
        "*Konteks Pakai*: Busana adat dan pertunjukan budaya."
    ),
}

# Bangun ulang arsitektur model (sama dengan saat training)
def build_model(num_classes):
    # Menggunakan MobileNetV2 base
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    return model

# Load model dan class names dari Hugging Face Hub
try:
    weights_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=WEIGHTS_FILENAME
    )

    class_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=CLASSES_FILENAME
    )

    MODEL = build_model(NUM_CLASSES)
    MODEL.load_weights(weights_path)

    with open(class_path, "r") as f:
        class_map = json.load(f)

    CLASS_NAMES = {int(k): v for k, v in class_map.items()}
    LABEL_NAMES = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]

    MODEL_READY = True

except Exception as e:
    print("GAGAL MEMUAT MODEL:", e)
    MODEL_READY = False
    LABEL_NAMES = ["Error"]

# Fungsi Prediksi
def classify_image(input_image: Image.Image):
    if not MODEL_READY:
        return {"Model gagal dimuat": 1.0}, "❌ Model tidak tersedia"

    # Preprocessing
    img = input_image.convert("RGB").resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    processed_img = img_array / 255.0

    # Expand Dimensi (Batch Size)
    img_batch = np.expand_dims(processed_img, axis=0)

    # Prediksi
    preds = MODEL.predict(img_batch, verbose=0)[0]

    results = {
        LABEL_NAMES[i]: float(preds[i])
        for i in range(len(preds))
    }
    top_class = max(results, key=results.get)
    confidence = results[top_class] * 100

    # Ambil deskripsi berdasarkan kelas teratas
    description = BATIK_DESCRIPTIONS.get(
        top_class,
        "Deskripsi belum tersedia untuk motif ini. Silakan perbarui kamus BATIK_DESCRIPTIONS."
    )

    # Ambil Top-5 untuk Markdown
    top5 = sorted(results.items(), key=lambda x: x[1], reverse=True)[:5]
    top5_md = "\n".join([f"- **{k}**: {v*100:.2f}%" for k, v in top5])

    summary = (
        f"### 🧵 Hasil Prediksi\n"
        f"Motif batik terdeteksi: *{top_class.upper()}*  \n"
        f"Tingkat kepercayaan: *{confidence:.2f}%*\n\n"
        f"### 📚 Penjelasan Motif\n"
        f"{description}\n\n"
        f"### 🔝 Top-5 Probabilitas\n"
        f"{top5_md}"
    )

    return results, summary
    
# INTERFACE GRADIO
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🇮🇩 Klasifikasi Motif Batik Indonesia
    ### Menggunakan MobileNetV2 (Transfer Learning)
    Aplikasi ini mengklasifikasikan **14 motif batik Indonesia**
    menggunakan model **MobileNetV2** dengan lapisan klasifikasi 256 neuron.
    **Langkah penggunaan:**
    1. Unggah gambar motif batik
    2. Klik **Submit**
    3. Lihat hasil prediksi, tingkat kepercayaan, dan penjelasan motif
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="pil",
                label="📤 Unggah Gambar Batik"
            )
            submit_btn = gr.Button("🔍 Submit", variant="primary")

        with gr.Column():
            label_output = gr.Label(
                num_top_classes=5,
                label="📊 Probabilitas Kelas"
            )
            summary_output = gr.Markdown()

    submit_btn.click(
        fn=classify_image,
        inputs=image_input,
        outputs=[label_output, summary_output]
    )
    
    gr.Markdown("""
    ---
    **Informasi Model**
    - Arsitektur: MobileNetV2
    - Input Size: 224 × 224
    - Jumlah Kelas: 14
    - Framework: TensorFlow & Keras
    - Deployment: Hugging Face Spaces (Gradio)
    """)

# JALANKAN APLIKASI
demo.launch()
