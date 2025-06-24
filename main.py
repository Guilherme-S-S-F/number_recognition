from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

# Função de preparação da imagem
def prepare_image(path):
    # Abrir imagem e converter para escala de cinza
    img = Image.open(path).convert('L')

    # Inverter cores: MNIST usa dígito branco em fundo preto
    img = ImageOps.invert(img)

    # Redimensionar para 28x28
    img = img.resize((28, 28))

    # Converter para array, normalizar e ajustar dimensões
    img_arr = np.array(img).astype('float32') / 255.0
    img_arr = np.expand_dims(img_arr, axis=-1)  # Adiciona canal (28, 28, 1)
    img_arr = np.expand_dims(img_arr, axis=0)   # Adiciona batch (1, 28, 28, 1)

    return img_arr, img

# Carregar o modelo previamente treinado
model = load_model('mnist_number_recognition_keras.h5')

# Caminho da imagem a ser testada
image_path = 'numero_teste.png'

# Preparar a imagem
img_input, img_processed = prepare_image(image_path)

# Fazer predição
prediction = model.predict(img_input)
predicted_number = np.argmax(prediction)

# Mostrar resultado
print(f"O número reconhecido é: {predicted_number}")

# Visualizar a imagem usada na predição
plt.imshow(img_processed, cmap='gray')
plt.title(f"Predição: {predicted_number}")
plt.axis('off')
plt.show()
