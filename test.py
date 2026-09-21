import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMAGE_SIZE = (150, 150)
MODEL_PATH = "model.h5"
DATA_DIR = "data"


def load_class_labels(data_dir: str) -> list:
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")
    return sorted(os.listdir(data_dir))


def pre_process_image(image_path: str) -> np.ndarray:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

    model = load_model(MODEL_PATH)
    class_labels = load_class_labels(DATA_DIR)

    test_images = ["test.png", "test2.png"]
    predictions = {}

    for img_path in test_images:
        processed_img = pre_process_image(img_path)
        pred = model.predict(processed_img, verbose=0)
        predicted_class = class_labels[np.argmax(pred[0])]
        predictions[img_path] = predicted_class

    print("Predicted Class:")
    print("Test 1:", predictions["test.png"])
    print("Test 2:", predictions["test2.png"])


if __name__ == "__main__":
    main()