import json
import os
import sys
import time
import urllib.request
 
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
T1_CM, T2_CM = 6, 9

# Firebase Realtime Database (shared project)
FIREBASE_DB_URL = "https://selada-a9428-default-rtdb.asia-southeast1.firebasedatabase.app"  # ambang lebar daun: >=6 cm mendekati panen, >=9 cm siap panen
 
# ===================== FIREBASE =====================
FIREBASE_NODE = "-P56DmqHVTCP5FIhZ7R5"
FIREBASE_BASE_URL = f"{FIREBASE_DB_URL}/{FIREBASE_NODE}"

def firebase_put(path, payload):
    """Simpan data ke Firebase Realtime Database via REST PUT."""
    url = f"{FIREBASE_BASE_URL}/{path.strip('/')} .json".replace(" ", "")
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="PUT", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=8) as response:
        return json.loads(response.read().decode("utf-8"))

def firebase_post(path, payload):
    """Tambah record baru dengan ID otomatis Firebase."""
    url = f"{FIREBASE_BASE_URL}/{path.strip('/')} .json".replace(" ", "")
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=8) as response:
        return json.loads(response.read().decode("utf-8"))

def save_cnn_firebase(hasil, filename=None):
    """Simpan hasil CNN ke cnn/terakhir dan cnn/riwayat."""
    payload = dict(hasil)
    payload["filename"] = filename or "-"
    payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    firebase_put("cnn/terakhir", payload)
    return firebase_post("cnn/riwayat", payload)


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
