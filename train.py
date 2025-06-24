import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# 1. Carregar os dados MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalizar e ajustar formato
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# 3. Codificar as labels (one-hot encoding)
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 4. Construir o modelo
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 5. Compilar
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Treinar
model.fit(x_train, y_train_cat, epochs=5, batch_size=64, validation_split=0.1)

# 7. Avaliar
loss, acc = model.evaluate(x_test, y_test_cat)
print(f"\nAcurácia no teste: {acc:.4f}")

# 8. Salvar o modelo
model.save('mnist_number_recognition_keras.h5')

# 9. Visualizar algumas previsões
predictions = model.predict(x_test)

for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Real: {y_test[i]}, Predito: {np.argmax(predictions[i])}")
    plt.axis('off')
    plt.show()
