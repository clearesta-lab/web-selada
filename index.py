import json
import os
import sys
 
import cv2
import numpy as np
 
# ===================== KONFIGURASI =====================
IMG_SIZE = (128, 128)
BATCH = 16
EPOCHS = 30
DATASET_DIR = "dataset"
MODEL_PATH = "selada_cnn.keras"
CLASS_NAMES = ["masa_pertumbuhan", "mendekati_panen", "siap_panen"]
CLASS_LABELS = {
    "masa_pertumbuhan": "Masa Pertumbuhan",
    "mendekati_panen": "Mendekati Panen",
    "siap_panen": "Siap Panen",
}
PX_PER_CM = 40      # kalibrasi: berapa piksel = 1 cm (sama dengan CNN_SCALE di HTML)
T1_CM, T2_CM = 6, 9  # ambang lebar daun: >=6 cm mendekati panen, >=9 cm siap panen
 
 
# ===================== MODEL CNN =====================
def build_model(num_classes=3):
    from tensorflow.keras import layers, models
 
    model = models.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),
        layers.Rescaling(1.0 / 255),
        # augmentasi (aktif hanya saat training)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.15),
 
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
 
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
 
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
 
        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
 
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
 
 
def train():
    import tensorflow as tf
 
    if not os.path.isdir(DATASET_DIR):
        sys.exit(f"Folder '{DATASET_DIR}' tidak ditemukan. Buat dulu sesuai struktur di atas.")
 
    common = dict(
        directory=DATASET_DIR, labels="inferred", label_mode="int",
        class_names=CLASS_NAMES, image_size=IMG_SIZE, batch_size=BATCH,
        validation_split=0.2, seed=42,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(subset="training", **common)
    val_ds = tf.keras.utils.image_dataset_from_directory(subset="validation", **common)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
 
    model = build_model(len(CLASS_NAMES))
    model.summary()
 
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    model.save(MODEL_PATH)
    print(f"Model tersimpan: {MODEL_PATH}")
 
 
_model = None
 
 
def get_model():
    global _model
    if _model is None:
        import tensorflow as tf
        if not os.path.exists(MODEL_PATH):
            return None
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model
 
 
def cnn_predict(bgr):
    """Prediksi kematangan dengan CNN. Return dict persen per kelas, atau None jika model belum dilatih."""
    model = get_model()
    if model is None:
        return None
    rgb = cv2.cvtColor(cv2.resize(bgr, IMG_SIZE), cv2.COLOR_BGR2RGB)
    p = model.predict(rgb[None].astype("float32"), verbose=0)[0] * 100
    return {CLASS_LABELS[c]: round(float(v), 1) for c, v in zip(CLASS_NAMES, p)}
 
 
# ===================== SEGMENTASI & LEBAR DAUN =====================
def segment(bgr):
    """Segmentasi warna di ruang HSV (OpenCV: H 0-179, S/V 0-255)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
 
    green = (h >= 35) & (h <= 85) & (s >= 56) & (v >= 51)
    yellow = (h >= 19) & (h < 35) & (s >= 89) & (v >= 115)
    brown = (h >= 5) & (h < 19) & (s >= 77) & (v >= 38) & (v <= 166)
    dark = v < 56
    red = (s > 128) & (v > 77) & ((h < 5) | (h > 85))
    return green, yellow, brown, dark, red
 
 
def find_holes(green):
    """Area non-hijau yang tertutup daun (tidak menyentuh tepi gambar) = lubang."""
    nongreen = (~green).astype(np.uint8)
    n, labels = cv2.connectedComponents(nongreen, connectivity=4)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    enclosed = (nongreen == 1) & ~np.isin(labels, border)
    return enclosed
 
 
def measure_leaf_width(green):
    """Lebar daun (px) = rentang kolom hijau persentil 1-99, seperti di HTML."""
    cols = np.where(green.sum(axis=0) >= 3)[0]
    if len(cols) < 5:
        return None
    x1 = cols[int(len(cols) * 0.01)]
    x2 = cols[int(np.ceil(len(cols) * 0.99)) - 1]
    return float(x2 - x1)
 
 
def is_lettuce(bgr, green, red):
    """Validasi sederhana: apakah gambar cukup mengandung warna hijau daun."""
    n = green.size
    g_ratio = green.sum() / n
    if g_ratio < 0.03:
        return f"Daun hijau hampir tidak terdeteksi ({g_ratio * 100:.1f}% area gambar)."
    hue_mean = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 0][green].mean() * 2
    if hue_mean < 75 or hue_mean > 150:
        return f"Warna hijau tidak sesuai selada (hue rata-rata {hue_mean:.0f} derajat)."
    if red.sum() / n >= 0.35:
        return "Gambar didominasi warna non-tanaman."
    return None
 
 
def phase_by_width(w_cm):
    if w_cm >= T2_CM:
        return "Siap Panen"
    if w_cm >= T1_CM:
        return "Mendekati Panen"
    return "Masa Pertumbuhan"
 
 
# ===================== DETEKSI HAMA & PENYAKIT =====================
def diagnose(bgr, green, yellow, brown, dark, red):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    zone = cv2.dilate(green.astype(np.uint8), k).astype(bool)
    enclosed = find_holes(green)
    in_plant = zone | enclosed
 
    Gc = int(green.sum())
    Yc = int((yellow & in_plant & ~green).sum())
    Bc = int((brown & in_plant & ~green & ~yellow).sum())
    Dc = int((dark & in_plant & ~green & ~yellow & ~brown).sum())
    Hc = int((enclosed & ~green & ~yellow & ~brown & ~dark).sum())
    A = max(1, Gc + Yc + Bc + Dc + Hc)
    sh = {"g": Gc / A * 100, "y": Yc / A * 100, "b": Bc / A * 100,
          "d": Dc / A * 100, "h": Hc / A * 100}
 
    def lvl(v):
        return "Berat" if v >= 30 else "Sedang" if v >= 15 else "Ringan"
 
    out = []
    if sh["y"] >= 8:
        out.append({"name": "Klorosis (daun menguning)", "persen": round(sh["y"], 1), "level": lvl(sh["y"]),
                    "saran": "Cek pH (5.5-6.5) dan EC larutan nutrisi, sesuaikan dosis nutrisi A/B."})
    if sh["b"] >= 8:
        out.append({"name": "Tip burn / nekrosis tepi", "persen": round(sh["b"], 1), "level": lvl(sh["b"]),
                    "saran": "Turunkan suhu, perbaiki sirkulasi udara, pastikan suplai kalsium dan cek EC."})
    if sh["d"] >= 6:
        out.append({"name": "Bercak gelap (jamur/bakteri)", "persen": round(sh["d"], 1), "level": lvl(sh["d"]),
                    "saran": "Kurangi kelembapan dan percikan air ke daun, buang daun terinfeksi."})
    if sh["h"] >= 5:
        out.append({"name": "Lubang / bagian daun hilang", "persen": round(sh["h"], 1), "level": lvl(sh["h"]),
                    "saran": "Periksa sisi bawah daun dan media tanam, buang hama manual, pasang jaring/perangkap."})
    return out, round(sh["g"], 1)
 
 
# ===================== PIPELINE UTAMA =====================
def analyze(bgr):
    green, yellow, brown, dark, red = segment(bgr)
 
    reason = is_lettuce(bgr, green, red)
    if reason is None and measure_leaf_width(green) is None:
        reason = "Bentuk daun tidak terdeteksi."
    if reason:
        return {"selada": False, "alasan": reason}
 
    w_px = measure_leaf_width(green)
    w_cm = w_px / PX_PER_CM
 
    cnn = cnn_predict(bgr)  # prediksi CNN (None jika belum ada model)
    if cnn is None:
        # fallback: probabilitas dari lebar daun (sama seperti HTML)
        sig = lambda x: 1 / (1 + np.exp(-x))
        g, r = sig((T1_CM - w_cm) / 0.8), sig((w_cm - T2_CM) / 0.8)
        n = max(0.001, 1 - g - r)
        s = g + n + r
        cnn = {"Masa Pertumbuhan": round(g / s * 100, 1),
               "Mendekati Panen": round(n / s * 100, 1),
               "Siap Panen": round(r / s * 100, 1)}
        sumber = "fallback lebar daun (model CNN belum dilatih)"
    else:
        sumber = "CNN"
 
    fase = max(cnn, key=cnn.get)
    items, sehat = diagnose(bgr, green, yellow, brown, dark, red)
 
    return {
        "selada": True,
        "lebar_cm": round(w_cm, 2),
        "lebar_px": round(w_px),
        "fase_by_lebar": phase_by_width(w_cm),
        "fase": fase,
        "keyakinan": cnn[fase],
        "probabilitas": cnn,
        "sumber_prediksi": sumber,
        "area_sehat_persen": sehat,
        "hama_penyakit": items,
        "status_kesehatan": "Sehat" if not items else "Ada gejala",
    }
 
 
def read_image_bytes(data):
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("File bukan gambar valid")
    # resize ke lebar 480 seperti di HTML supaya hasil konsisten
    sc = 480 / img.shape[1]
    return cv2.resize(img, (480, max(1, round(img.shape[0] * sc))))
 
 
# ===================== API FLASK =====================
def serve():
    from flask import Flask, jsonify, request
    from flask_cors import CORS
 
    app = Flask(__name__)
    CORS(app)  # izinkan HTML (file:// atau server lain) memanggil API ini
 
    @app.get("/health")
    def health():
        return jsonify({"ok": True, "model": os.path.exists(MODEL_PATH)})
 
    @app.post("/predict")
    def predict():
        f = request.files.get("file")
        if f is None:
            return jsonify({"error": "kirim gambar di field 'file'"}), 400
        try:
            img = read_image_bytes(f.read())
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(analyze(img))
 
    app.run(host="0.0.0.0", port=5000)
 
 
# ===================== CLI =====================
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "train":
        train()
    elif cmd == "predict" and len(sys.argv) > 2:
        with open(sys.argv[2], "rb") as fh:
            print(json.dumps(analyze(read_image_bytes(fh.read())), indent=2, ensure_ascii=False))
    elif cmd == "serve":
        serve()
    else:
        print(__doc__)