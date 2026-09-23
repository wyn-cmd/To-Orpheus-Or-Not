# Image classification test script for evaluating saved Keras models against sample images.

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMAGE_SIZE = (150, 150)
MODEL_PATH = "model.h5"
DATA_DIR = "data"
TEST_IMAGES = ["test.png", "test2.png"]


# Load class labels alphabetically from the training data directory.
def load_class_labels(data_dir: str) -> list:
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")
    
    labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    if not labels:
        # Fallback to all files/folders if no subdirectories are found.
        labels = sorted(os.listdir(data_dir))
    return labels


# Load and preprocess a single image for model inference.
def pre_process_image(image_path: str) -> np.ndarray:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


# Run predictions on hardcoded test images and print the results.
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

    model = load_model(MODEL_PATH)
    class_labels = load_class_labels(DATA_DIR)

    print("Predicted Class:")
    for img_path in TEST_IMAGES:
        processed_img = pre_process_image(img_path)
        pred = model.predict(processed_img, verbose=0)
        predicted_class = class_labels[np.argmax(pred[0])]
        print(f"{img_path}: {predicted_class}")


if __name__ == "__main__":
    main()