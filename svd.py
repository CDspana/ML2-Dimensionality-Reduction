# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 19:58:42 2024

@author: user
"""

# SVD from scrath
from PIL import Image
import numpy as np

class SVD_SCRATH:
    def __init__(self, image_path):
        # Cargar la imagen y convertirla a escala de grises
        self.image = Image.open(image_path).convert('L')
        self.matrix = np.array(self.image)

    def svd(self):
        # Realizar la SVD
        U, S, VT = np.linalg.svd(self.matrix, full_matrices=False)
        return U, S, VT

    def reconstruct_image(self, U, S, VT, k):
        # Reconstruir la imagen usando k valores singulares
        reconstructed_matrix = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]
        reconstructed_image = Image.fromarray(reconstructed_matrix.astype(np.uint8))
        return reconstructed_image

    def calculate_metrics(self, original, approx):
        mse_value = np.mean((original - approx) ** 2)
        psnr_value = 10 * np.log10((255 ** 2) / mse_value)
        return mse_value, psnr_value