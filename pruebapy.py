# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 21:44:27 2024

@author: user
"""

from flask import Flask, request, render_template
from pca import PCA_SCRATH
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
import numpy as np



mnist = fetch_openml('mnist_784')
mnist.target = mnist.target.astype(int)

X = mnist.data[(mnist.target == 0) | (mnist.target == 8)]
y = mnist.target[(mnist.target == 0) | (mnist.target == 8)]
print(X.shape)
print(type(X))
pca = PCA_SCRATH(n_components=2)

pca.fit(X)
X_transformed = pca.transform(X)

model = LogisticRegression(max_iter=3000)
model.fit(X_transformed, y)

# Verificar si hay al menos una imagen disponible
if X.shape[0] == 0:
    print("No hay imágenes disponibles para los dígitos 0 y 8.")
else:
    # Seleccionar una imagen al azar
    random_index = np.random.choice(X.shape[0])

    # Obtener la imagen y su etiqueta correspondiente
    random_image = X[random_index]
    random_label = y[random_index]

    # Imprimir la forma y tipo de la imagen seleccionada al azar
    print("Forma de la imagen seleccionada al azar:", random_image.shape)
    print("Tipo de la imagen seleccionada al azar:", type(random_image))
    print("Etiqueta de la imagen seleccionada al azar:", random_label)

    # Resto de tu código
    pca = PCA_SCRATH(n_components=2)
    pca.fit(X)
    X_transformed = pca.transform(X)

    model = LogisticRegression(max_iter=3000)
    model.fit(X_transformed, y)

    # Predecir la etiqueta de la imagen seleccionada al azar
    random_image_transformed = pca.transform(random_image.reshape(1, -1))
    prediction = model.predict(random_image_transformed)

    # Imprimir la etiqueta predicha
    print("Etiqueta predicha para la imagen seleccionada al azar:", prediction[0])