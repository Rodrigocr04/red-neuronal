# Haciendolo desde 0

import numpy as np # para operaciones con matrices

# definir las funciones de activacion y sus derivadas
# Sigmoide y su derivada, para la capa de salida 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoid(a):
    return a * (1 - a)

# ReLU y su derivada, para capas ocultas
def relu(x):
    return np.maximum(0, x)

def derivada_relu(a):
    return np.where(a > 0, 1, 0)

# Prueba con los datos de excel
# Entrada X (20 datos, latitud y longitud)
X = np.array([
    [-2.848,   5.975], [-10.429,  -8.024], [ -1.284,   2.872], [  4.758,   5.955],
    [ -3.938,  -5.78 ], [  8.525,  17.866], [-13.715,   4.789], [ -3.19 ,  13.309],
    [  3.227,  18.344], [ -5.014, -17.507], [  0.491,   1.897], [ -0.54 ,   1.12 ],
    [ -2.032,   1.092], [ -1.476,  -1.606], [ -0.084,   3.386], [ -1.618,   1.16 ],
    [  2.238,  -1.671], [ -2.102,   1.924], [ -1.059,   4.297], [  2.642,   2.981]
])

# Etiquetas Y (respuestas esperadas): primeras 10 son 0, siguientes 10 son 1
Y = np.array([0]*10 + [1]*10).reshape(20, 1)

# Estructura de capas, Entrada(2) -> Oculta1(4) -> Oculta2(8) -> Salida(1)
# Inicializacion de pesos y sesgos
W1 = np.full((2, 4), 0.75)
b1 = np.full((1, 4), 0.25)

W2 = np.full((4, 8), 0.75)
b2 = np.full((1, 8), 0.25)

W3 = np.full((8, 1), 0.75)
b3 = np.full((1, 1), 0.25)

lr = 0.01  # Tasa de aprendizaje

# Propagacion hacia adelante
# Capa 1 (Entrada -> Capa Oculta 1)
z1 = np.dot(X, W1) + b1
a1 = relu(z1)

# Capa 2 (Capa Oculta 1 -> Capa Oculta 2)
z2 = np.dot(a1, W2) + b2
a2 = relu(z2)

# Capa 3 (Capa Oculta 2 -> Capa de Salida)
z3 = np.dot(a2, W3) + b3
a3 = sigmoid(z3)  # prediccion final 

print("Predicciones a3 (Epoca 0):")
print(np.round(a3, 4))

# Propagacion hacia atras
# Delta de la capa de salida
delta3 = (a3 - Y) * derivada_sigmoid(a3)

# Delta de la capa oculta 2
# Se propaga el delta de la capa 3 usando W3 traspuesto y la derivada de ReLU
delta2 = np.dot(delta3, W3.T) * derivada_relu(a2)

# Delta de la capa oculta 1
delta1 = np.dot(delta2, W2.T) * derivada_relu(a1)

# Actualizar los parametros de la capa 3
b3 = b3 - np.mean(delta3, axis=0, keepdims=True) * lr
W3 = W3 - np.dot(a2.T, delta3) * lr

# igual en las demas capas
b2 = b2 - np.mean(delta2, axis=0, keepdims=True) * lr
W2 = W2 - np.dot(a1.T, delta2) * lr

b1 = b1 - np.mean(delta1, axis=0, keepdims=True) * lr
W1 = W1 - np.dot(X.T, delta1) * lr

print("Nuevos pesos W3 (despues de Epoca 0):")
print(np.round(W3, 8))

epochs = 2  # cantidad de epocas

for epoch in range(epochs):
    # Propagacion hacia adelante
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)

    z3 = np.dot(a2, W3) + b3
    a3 = sigmoid(z3)

    # Calculo del loss (error cuadratico medio)
    loss = np.mean((a3 - Y) ** 2)

    # Propagacion hacia atras
    delta3 = (a3 - Y) * derivada_sigmoid(a3)
    delta2 = np.dot(delta3, W3.T) * derivada_relu(a2)
    delta1 = np.dot(delta2, W2.T) * derivada_relu(a1)

    # Actualizar los parametros
    b3 -= np.mean(delta3, axis=0, keepdims=True) * lr
    W3 -= np.dot(a2.T, delta3) * lr

    b2 -= np.mean(delta2, axis=0, keepdims=True) * lr
    W2 -= np.dot(a1.T, delta2) * lr

    b1 -= np.mean(delta1, axis=0, keepdims=True) * lr
    W1 -= np.dot(X.T, delta1) * lr

    if (epoch + 1) % 1 == 0:
        print(f"Epoca {epoch + 1} - Loss: {loss:.6f}")