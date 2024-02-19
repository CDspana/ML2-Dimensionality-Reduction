# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 20:13:10 2024

@author: user
"""

# app.py
from flask import Flask, request, render_template
from pca import PCA_SCRATH
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
import numpy as np

app = Flask(__name__)

mnist = fetch_openml('mnist_784')
# Convertir las etiquetas a enteros
mnist.target = mnist.target.astype(int)

# Seleccionar solo las imágenes correspondientes a los dígitos 0 y 8
X = mnist.data[(mnist.target == 0) | (mnist.target == 8)]
y = mnist.target[(mnist.target == 0) | (mnist.target == 8)]

# Aplicar PCA
pca = PCA_SCRATH(n_components=2)


# fit the data
pca.fit(X)

# transform the data using the PCA object
X_transformed = pca.transform(X)

# Crear un modelo simple para propósitos de demostración
model = LogisticRegression(max_iter=3000)
model.fit(X_transformed, y)



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Procesa el registro recibido y realiza la clasificación
        # Aquí deberías implementar la lógica de procesamiento del registro

    
        
        # Crear una imagen de ejemplo (asegúrate de que tenga la forma correcta)
        example_image = X.iloc[3].values.reshape(1, -1)

        # Aplicar PCA a la imagen de ejemplo
        example_image_transformed = pca.transform(example_image)

        # Realizar una predicción con el modelo
        predicted_class = model.predict(example_image_transformed)
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print(predicted_class)

        return render_template('index.html', predicted_class=predicted_class)

    return render_template('index.html', predicted_class=None)

if __name__ == '__main__':
    app.run(debug=False)
    
    print("Servidor ejecutándose en http://127.0.0.1:5000/")

