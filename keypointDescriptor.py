import numpy as np
import math
import sys
import cv2
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as sig
import time
from scaleSpace import *
from keypointDetection import *
from timeDecorator import timeit




def orientationPointCle(point_cle,L,sigma_list):
    # QUESTION : Quel sigma on choisi? La borne inférieure ou la borne supérieur? (car les points clés sont repérés
    # par rapport à la DoG et là on retourne sur L
    y, x, s = point_cle # Pour le test
    R = 4  # Rayon de la zone de voisinage
    n = 2 * R + 1
    L_grady, L_gradx = gradient(L[:, :, s])
    G = gaussian_filter(n, 1.5 * sigma_list[s]).flatten()

    points_cles_orientes = np.empty((0, 4))
    # Calcul des matrices de d'amplitude et d'orientation pour chaque point du voisinage et conversion en vecteur
    mat_m = np.sqrt((2 * L_gradx[y - R:y + R + 1, x - R:x + R + 1]) ** 2 + (
    2 * L_grady[y - R:y + R + 1, x - R:x + R + 1]) ** 2).flatten()
    mat_theta = np.arctan2(L_grady[y - R:y + R + 1, x - R:x + R + 1],
                           L_gradx[y - R:y + R + 1, x - R:x + R + 1]).flatten() + np.pi

    inervalles = np.linspace(0, 2 * np.pi, 37)  # 37 éléments donc 36 intervalles
    hist = np.zeros(36)
    for i in range(0, 36):
        bool1 = mat_theta > inervalles[i]
        bool2 = mat_theta < inervalles[i + 1]  # Obligé de créer des variables, triste langage de programmation..
        # Numéros des pixels situés dans l'intervalle en question
        pixels_inervalle = np.nonzero(bool1 & bool2)
        # Pondération par la fenêtre gaussienne et normalisation par l'amplitude
        hist[i] = sum(G[pixels_inervalle] / mat_m[pixels_inervalle])

    orientations_ind = np.nonzero(hist > 0.8 * np.max(hist))
    orientations = inervalles[orientations_ind] + (
                                                  2 * np.pi / 36) / 2  # On prend la valeur médiane de l'intervalle (pas de 2pi/36)
    for orientation in orientations:  # Peut être moyen de le faire plus court..
        points_cles_orientes = np.vstack((points_cles_orientes, [y, x, s, orientation]))

    return points_cles_orientes
