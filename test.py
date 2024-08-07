import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# image path
image_path = "test.png"


model = load_model("model.h5")

# function to pre-process the image for prediction
def pre_process_image(img):
  img = cv2.imread(img)
  img = cv2.resize(img, (150, 150))
  img = img.astype("float32") / 255.0
  img = np.expand_dims(img, axis=0)
  return img

# Pre-process the image
image = pre_process_image(image_path)
image2 = pre_process_image('test2.png')

# prediction if image is Orpheus or not
prediction = model.predict(image)
prediction2 = model.predict(image2)

# Get the Not-Orpheus and Orpheus labels
class_labels = os.listdir("data")

# display predicted class
predicted_class = class_labels[np.argmax(prediction[0])]
predicted_class2 = class_labels[np.argmax(prediction2[0])]
print("Predicted Class:\nTest 1:", predicted_class)
print('Test 2:', predicted_class2)