"""Predict the gender for one or more face images from the command line.

Usage:
    python predict.py path/to/face1.jpg [path/to/face2.jpg ...]
"""
import sys

from tensorflow import keras

from src.gender_detection import MODEL_PATH, predict_image


def main(paths):
    if not paths:
        print(__doc__)
        return 1
    model = keras.models.load_model(MODEL_PATH)
    for path in paths:
        label, confidence = predict_image(model, path)
        print(f"{path}: {label} ({confidence:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
