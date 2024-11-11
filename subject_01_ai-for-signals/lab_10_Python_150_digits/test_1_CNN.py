import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np

# Подготовка правильных ответов
model = tf.keras.models.load_model("digit_diagnosis.keras")

path = "image_for_test/digit6024.png"

image = tf.keras.preprocessing.image.load_img(path, target_size=(28, 28))
x = tf.keras.preprocessing.image.img_to_array(image)
classes = model.predict(np.array([x]))

print("Вероятность принадлежности тестового изображения к одному из классов:")
print(classes)

print("Номер класса, предсказанного нейросетью:")
print(np.argmax(classes) + 1)

classes = model.predict(np.array([x]))

# Вывод элемента 3 тензора classes нулевого ранга (т.е. вектора)
# Поскольку элементы в тензоре нумеруются начиная с нуля, фактически элемент будет 4
print("Максимальная вероятность принадлежности:")
print(np.max(classes))

# Задание порога распознавания
level = 0.85

print("Порог распознавания = ", level)
if np.max(classes) < level:
    print("Ошибка распознавания!")
