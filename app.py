"""Desktop app: pick a face image and see the predicted gender.

Usage:
    python app.py
"""
import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk
from tensorflow import keras

from src.gender_detection import MODEL_PATH, predict_image

PREVIEW_SIZE = (220, 220)


class GenderApp:
    def __init__(self, root):
        self.model = keras.models.load_model(MODEL_PATH)
        self.root = root
        root.title("Gender Detection System")
        root.configure(bg="#f4f6f8", padx=24, pady=20)

        tk.Label(root, text="Gender Detection System", font=("Segoe UI", 18, "bold"),
                 bg="#f4f6f8", fg="#1f2933").pack()
        tk.Label(root, text="Load a face photo to classify it", font=("Segoe UI", 10),
                 bg="#f4f6f8", fg="#52606d").pack(pady=(0, 12))

        self.preview = tk.Label(root, bg="#d9e2ec", width=30, height=14)
        self.preview.pack(pady=6)

        self.result = tk.Label(root, text="Prediction: -", font=("Segoe UI", 16, "bold"),
                               bg="#f4f6f8", fg="#1f2933")
        self.result.pack(pady=(12, 2))
        self.confidence = tk.Label(root, text="", font=("Segoe UI", 11), bg="#f4f6f8", fg="#52606d")
        self.confidence.pack()

        tk.Button(root, text="Load and Classify Image", font=("Segoe UI", 11), bg="#2563eb", fg="white",
                  activebackground="#1d4ed8", relief="flat", padx=14, pady=6,
                  command=self.choose_image).pack(pady=16)

    def choose_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.classify(path)

    def classify(self, path):
        label, confidence = predict_image(self.model, path)
        photo = ImageTk.PhotoImage(Image.open(path).convert("RGB").resize(PREVIEW_SIZE))
        self.preview.config(image=photo, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1])
        self.preview.image = photo
        colour = "#2563eb" if label == "Male" else "#db2777"
        self.result.config(text=f"Prediction: {label}", fg=colour)
        self.confidence.config(text=f"Confidence: {confidence:.1%}")


if __name__ == "__main__":
    root = tk.Tk()
    GenderApp(root)
    root.mainloop()
