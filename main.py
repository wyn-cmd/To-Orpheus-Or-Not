# Version 1.1

# Image classification training script for the To-Orpheus project.

import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load images and corresponding labels from the dataset directory structure.
def load_images(directory):
    images = []
    labels = []
    
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if not os.path.isdir(folder_path):
            continue
            
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            
            if img is not None:
                img = cv2.resize(img, (150, 150))
                img = img / 255.0
                images.append(img)
                labels.append(folder)
                
    return np.array(images), np.array(labels)

# Configuration constants
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
DATA_DIR = 'data'

# Load and preprocess dataset
x, y = load_images(DATA_DIR)

le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Build the convolutional neural network model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model using the training splits
model.fit(x_train, y_train, epochs=10, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))

# Evaluate performance on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Save the trained model weights and architecture to disk
model.save('model.h5')