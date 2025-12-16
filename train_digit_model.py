import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

(x_mnist, y_mnist), (x_mnist_test, y_mnist_test) = mnist.load_data()

x_mnist = x_mnist.reshape(-1, 28, 28, 1) / 255.0
x_mnist_test = x_mnist_test.reshape(-1, 28, 28, 1) / 255.0

y_mnist = tf.keras.utils.to_categorical(y_mnist, 10)
y_mnist_test = tf.keras.utils.to_categorical(y_mnist_test, 10)

DATASET_PATH = r"C:\Users\aarav_qoo\PycharmProjects\pythonProject1\Sudoku_using_Opencv\Digit_Detection_Model\digit_updated"
IMG_SIZE = 28

x_custom = []
y_custom = []


for label in range(10):
    folder_path = os.path.join(DATASET_PATH, str(label))
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        if np.mean(img) > 127:
            img = cv2.bitwise_not(img)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        x_custom.append(img)
        y_custom.append(label)


x_custom = np.array(x_custom).reshape(-1, 28, 28, 1)
y_custom = tf.keras.utils.to_categorical(y_custom, 10)

x_custom_train, x_custom_test, y_custom_train, y_custom_test = train_test_split(
    x_custom, y_custom, test_size=0.2, random_state=42)

x_train = np.concatenate((x_mnist, x_custom_train), axis=0)
y_train = np.concatenate((y_mnist, y_custom_train), axis=0)

indices = np.arange(len(x_train))
np.random.shuffle(indices)

x_train = x_train[indices]
y_train = y_train[indices]

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)


model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=15,
    validation_data=(x_mnist_test, y_mnist_test)
)

loss_mnist, acc_mnist = model.evaluate(x_mnist_test, y_mnist_test)
loss_custom, acc_custom = model.evaluate(x_custom_test, y_custom_test)

print("MNIST accuracy:", acc_mnist)
print("Custom dataset accuracy:", acc_custom)


model.save("digit_model.h5")



