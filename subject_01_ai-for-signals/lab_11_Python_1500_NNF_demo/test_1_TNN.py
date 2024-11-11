import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

# Подготовка правильных ответов
model = tf.keras.models.load_model("plant_diagnosis_3D.keras")

path = "image_for_test/1.png"

image = tf.keras.preprocessing.image.load_img(path, target_size=(4, 6))
x = tf.keras.preprocessing.image.img_to_array(image)
classes = model.predict(np.array([x]))

print("Вероятность принадлежности тестового изображения к одному из классов:")
print(classes)

print("Номер класса, предсказанного нейросетью:")
print(np.argmax(classes) + 1)
