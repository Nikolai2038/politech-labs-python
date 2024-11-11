# Программа обученной нейронной сети выводит вероятности всех возможных вариантов результатов классификации (отнесения к нескольким классам).
# Предусмотрена возможность задания порога распознавания level.
# Если все полученные вероятности меньше level, выводится сообщение "ошибка распознавания".
# Файлы для тестирования:
# - digit6102 - изображение числа 6 - правильное распознавание
# - digit6045 - изображение числа 6 - ошибка распознавания

from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.models.sequential import Sequential
from keras.src.layers.convolutional.conv2d import Conv2D
from keras.src.layers.pooling.max_pooling2d import MaxPooling2D
from keras.src.layers.activations.activation import Activation
from keras.src.layers.regularization.dropout import Dropout
from keras.src.layers.reshaping.flatten import Flatten
from keras.src.layers.core.dense import Dense

# Каталог с данными для обучения
train_dir = "train_dir"

# Каталог с данными для тестирования
test_dir = "test_dir"

# Каталог с данными для тестирования
val_dir = "val_dir"

# Размеры изображения
img_width, img_height = 28, 28

# Размерность тензора на основе изображения для входных данных в нейронную сеть
input_shape = (img_width, img_height, 3)

# TODO: В цикле проверить все значения
# Количество эпох: 50, 75, 100 или 125
epochs = 50

# TODO: В цикле проверить все значения
# Размер мини-выборки: 20, 50 или 100
batch_size = 20

# TODO: В цикле проверить все значения
# Количество нейронов во входном слое: 500, 700, 900 или 1200
input_neurons = 500

# TODO: В цикле проверить все значения
# Количество нейронов в скрытом слое: 500, 700, 900 или 1200
hidden_neurons = 500

# Количество изображений для обучения
nb_train_samples = 100

# Количество изображений для проверки
nb_validation_samples = 30

# Количество изображений для тестирования
nb_test_samples = 20

model = Sequential()

# Входной слой
model.add(Conv2D(input_neurons, (3, 3), input_shape=input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Скрытый слой
model.add(Flatten())
model.add(Dense(hidden_neurons))
model.add(Activation("relu"))

model.add(Dropout(0.25))

# Выходной слой
model.add(Dense(9))
model.add(Activation("softmax"))

model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="sparse",
)

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="sparse",
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="sparse",
)

model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size,
    epochs=epochs,
    shuffle=True,
)
model.save("digit_diagnosis.keras")

scores = model.evaluate(test_generator)
print("Score: ", scores)

# TODO: Вывести время, затраченное на обучение - в секундах с долями
# ...

# TODO: Вывести количество эпох, размер мини-выборки и количество нейронов во входном и скрытом слоях, при которых самый высокий Score
# ...
