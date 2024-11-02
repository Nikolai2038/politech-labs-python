import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow.keras import utils

#Подготовка правильных ответов

model = tf.keras.models.load_model('digit_diagnosis.h5')

path = 'image_for_test/digit6024.png'
#path = 'image_for_test/digit2101.png'

image = tf.keras.preprocessing.image.load_img(path, target_size=(28, 28))
x = tf.keras.preprocessing.image.img_to_array(image)
classes = model.predict(np.array([x]))
print(classes) #Выведет набор вероятностей принадлежности тестового изображения к каждому классу
print('номер класса, предсказанного нейросетью')
print(np.argmax(classes)+1)
#print('вероятность принадлежности тестового изображения к одному из классов')
classes = model.predict(np.array([x]))
#вывод элемента 3 тензора classes нулевого ранга (т.е. вектора)
#поскольку элементы в тензоре нумеруются начиная с нуля, фактически элемент будет 4
print('максимальная вероятность принадлежности')
print(np.max(classes))
level=0.85 #задание порога распознавания
print('порог распознавания=',level)
if np.max(classes)<level:
	print('ошибка распознавания!')