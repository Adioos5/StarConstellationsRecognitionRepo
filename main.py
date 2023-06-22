import os
import cv2
import numpy as np
from keras.optimizers import Adamax, SGD
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Przygotowanie danych i wczytanie zestawu treningowego, walidacyjnego i testowego
# Dane wejściowe mogą być np. mapami astronomicznymi, obrazami lub danymi dotyczącymi położenia gwiazd

from keras.layers import Layer
from keras.layers import Conv2D, concatenate

class MultipleKernelConv2D(Layer):
    def __init__(self, num_kernels, kernel_sizes, activation='relu', **kwargs):
        super(MultipleKernelConv2D, self).__init__(**kwargs)
        self.num_kernels = num_kernels
        self.kernel_sizes = kernel_sizes
        self.activation = activation

    def build(self, input_shape):
        input_channels = input_shape[-1]
        self.kernels = []
        for kernel_size in self.kernel_sizes:
            kernel = self.add_weight(name='kernel_{}'.format(kernel_size),
                                     shape=(kernel_size[0], kernel_size[1], input_channels, self.num_kernels),
                                     initializer='glorot_uniform',
                                     trainable=True)
            self.kernels.append(kernel)

    def call(self, inputs):
        outputs = []
        for kernel in self.kernels:
            conv = Conv2D(self.num_kernels, kernel, activation=self.activation, padding='same')(inputs)
            outputs.append(conv)
        merged = concatenate(outputs, axis=-1)
        return merged

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.num_kernels * len(self.kernel_sizes),)


def loadDataset(dirPath, labelsPath):
    # Definiowanie ścieżki do folderu zawierającego obrazy konstelacji
    image_folder = dirPath
    file = open(labelsPath, 'r')
    Lines = file.readlines()

    # Przygotowanie list na obrazy i etykiety
    images = []
    labels = []
    i = -1
    # Przechodzenie przez pliki w folderze obrazów
    for filename in os.listdir(image_folder):
        i+=1
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Wczytanie obrazu i przekształcenie go do odpowiedniego rozmiaru lub formatu
            image = cv2.imread(os.path.join(image_folder, filename))
            image = cv2.resize(image, (32, 32))  # Dopasowanie do ustalonego rozmiaru
            value = np.sqrt(((image.shape[0] / 2.0) ** 2.0) + ((image.shape[1] / 2.0) ** 2.0))
            polar_image = cv2.linearPolar(image, (image.shape[0] / 2, image.shape[1] / 2), value, cv2.WARP_FILL_OUTLIERS)
            polar_image = polar_image.astype(np.uint8)
            # Dodanie obrazu do listy obrazów
            images.append(polar_image)
            # Dodanie odpowiedniej etykiety na podstawie nazwy pliku lub innych informacji
            label = Lines[i]  # Funkcja, która przypisuje etykietę
            labels.append(label)

    return (np.array(images), np.array(labels))

(train_images, train_labels) = loadDataset("data/train2", "data/train_classes.txt")
(validation_images, validation_labels) = loadDataset("data/validation2", "data/validation_classes.txt")
(test_images, test_labels) = loadDataset("data/test2", "data/test_classes.txt")

# Kodowanie one-hot etykiet
num_classes = 16
train_labels = to_categorical(train_labels, num_classes)
validation_labels = to_categorical(validation_labels, num_classes)

# Inicjalizacja modelu
model = Sequential()

# Dodanie warstw konwolucyjnych
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Spłaszczenie danych
model.add(Flatten())

# Dodanie warstw w pełni połączonych
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Normalizacja obrazów (zakres pikseli [0, 255] do [0, 1])
train_images = train_images.astype('float32') / 255
validation_images = validation_images.astype('float32') / 255

# Trenowanie modelu
epochs = 30
batch_size = 64

history = model.fit(train_images, train_labels, validation_split=0.30, batch_size=batch_size, epochs=epochs,
          validation_data=(validation_images, validation_labels))

# Ocena modelu na zestawie testowym

# Normalizacja obrazów (zakres pikseli [0, 255] do [0, 1])
test_images = test_images.astype('float32') / 255
# Kodowanie one-hot etykiet
test_labels = to_categorical(test_labels, num_classes)

# Ocena modelu na danych testowych
loss, accuracy = model.evaluate(test_images, test_labels)

print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot the training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

# sample = test_images[0].reshape((1,32,32,3))
# predict_x = model.predict(sample)
#
# print(predict_x)