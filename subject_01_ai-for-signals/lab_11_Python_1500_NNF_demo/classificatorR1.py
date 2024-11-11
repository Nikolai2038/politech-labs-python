# Программа нейронной сети прямого распространения для распознавания 8-ми болезней пшеницы по 3D-цифровым описаниям изображений листьев.
# Количество цифровых компонентов - 6 (R,G,B,RG,RB,GB).
# Количество параметров Харалика - 4.
# Количество цифровых описаний для обучения - 1500.

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

# Каталог с данными для обучения
train_dir = "train_dir"

# Каталог с данными для тестирования
test_dir = "test_dir"

# Каталог с данными для тестирования
val_dir = "val_dir"

# Размеры изображения
img_width, img_height = 4, 6

# Размерность тензора на основе изображения для входных данных в нейронную сеть
input_shape = (img_width, img_height, 3)

# TODO: В цикле проверить все значения
# Количество эпох: 50, 75, 100 или 125
epochs = 50

# TODO: В цикле проверить все значения
# Размер мини-выборки: 20, 50 или 100
batch_size = 20

# Количество нейронов во входном слое
input_neurons = 500

# Количество нейронов в скрытом слое
hidden_neurons = 500

# Количество изображений для обучения
nb_train_samples = 1000

# Количество изображений для проверки
nb_validation_samples = 300

# Количество изображений для тестирования
nb_test_samples = 200

model = Sequential()

# Входной слой
model.add(Dense(input_neurons, input_shape=input_shape))
model.add(Activation("relu"))

# Скрытый слой
model.add(Flatten())
model.add(Dense(hidden_neurons))
model.add(Activation("relu"))

model.add(Dropout(0.25))

# Выходной слой
model.add(Dense(8))
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
model.save("plant_diagnosis_3D.keras")

scores = model.evaluate(test_generator)
print("Score: ", scores)

# TODO: Вывести время, затраченное на обучение - в секундах с долями
# ...

# TODO: Вывести количество эпох и размер мини-выборки, при которых самый высокий Score
# ...
