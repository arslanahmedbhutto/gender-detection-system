"""Shared helpers for loading face images, preprocessing them and running the model."""
import os
from pathlib import Path

import cv2
import numpy as np
from sklearn import preprocessing

IMG_SIZE = 128
CLASS_NAMES = ["Female", "Male"]  # label 0 = women, label 1 = men
LABELS = {"women": 0, "men": 1}

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("GENDER_DATA_DIR", REPO_ROOT / "data" / "dataset1"))
MODEL_PATH = REPO_ROOT / "model" / "gender_model.keras"


def read_image(image_path):
    """Read an image as grayscale and resize it to IMG_SIZE x IMG_SIZE."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))


def load_split(split, data_dir=DATA_DIR):
    """Load one dataset split (train / valid / test) as (images, labels)."""
    images, labels = [], []
    for folder, label in LABELS.items():
        for image_file in sorted((Path(data_dir) / split / folder).iterdir()):
            images.append(read_image(image_file))
            labels.append(label)
    return np.array(images), np.array(labels)


def preprocess(images):
    """Flatten each image and L2-normalise it. Used for training and prediction alike."""
    images = np.asarray(images, dtype="float32").reshape(len(images), IMG_SIZE * IMG_SIZE)
    return preprocessing.normalize(images)


def predict_image(model, image_path):
    """Return (class name, confidence) for a single image file."""
    probs = model.predict(preprocess([read_image(image_path)]), verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx])
