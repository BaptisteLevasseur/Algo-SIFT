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



def orientationPointsCles(points_cles_list,L, sigma_list):
    # Prend la liste des points clés ainsi que les images blurred (et les sigma correspondant)
    # Renvoie la liste des points clés assignés à une ou plusieurs orientations
    points_cles_orientes_list = np.empty((0, 4))
    # Parcourt des différents points clés et assignatiton d'orientation pour chacun
    for i in range(0, np.size(points_cles_list, 0)):
        point_cle = points_cles_list[i]
        points_cles_orientes = orientationPointCle(point_cle, L, sigma_list)
        points_cles_orientes_list = np.vstack((points_cles_orientes_list, points_cles_orientes))
    return points_cles_orientes_list

def orientationPointCle(point_cle,L,sigma_list):
    # QUESTION : Quel sigma on choisi? La borne inférieure ou la borne supérieur? (car les points clés sont repérés
    # par rapport à la DoG et là on retourne sur L
    y, x, s = point_cle # Pour le test
    R = 4  # Rayon de la zone de voisinage
    n = 2 * R + 1 # Taille du masque
    L_grady, L_gradx = gradient(L[:, :, s])
    # On travaille avec des vecteurs pour tous les pixels -> flatten()
    G = gaussian_filter(n, 1.5 * sigma_list[s]).flatten() # Filtre gaussien

    points_cles_orientes = np.empty((0, 4))
    # Calcul des matrices de d'amplitude et d'orientation pour chaque point du voisinage et conversion en vecteur
    mat_m = np.sqrt((2 * L_gradx[y - R:y + R + 1, x - R:x + R + 1]) ** 2 + (
    2 * L_grady[y - R:y + R + 1, x - R:x + R + 1]) ** 2).flatten()
    mat_theta = np.arctan2(L_grady[y - R:y + R + 1, x - R:x + R + 1],
                           L_gradx[y - R:y + R + 1, x - R:x + R + 1]).flatten() + np.pi
    n_bins=36
    inervalles = np.linspace(0, 2 * np.pi, n_bins+1)  # 37 éléments donc 36 intervalles
    hist = np.zeros(n_bins)
    for i in range(0, n_bins):
        bool1 = mat_theta > inervalles[i]
        bool2 = mat_theta < inervalles[i + 1]  # Obligé de créer des variables, triste langage de programmation..
        # Numéros des pixels situés dans l'intervalle en question
        pixels_inervalle = np.nonzero(bool1 & bool2)
        # Pondération par la fenêtre gaussienne et normalisation par l'amplitude
        hist[i] = sum(G[pixels_inervalle] / mat_m[pixels_inervalle])

    orientations_ind = np.nonzero(hist > 0.8 * np.max(hist))
    orientations = inervalles[orientations_ind] + (
                                                  2 * np.pi / n_bins) / 2  # On prend la valeur médiane de l'intervalle (pas de 2pi/36)
    for orientation in orientations:  # Peut être moyen de le faire plus court..
        points_cles_orientes = np.vstack((points_cles_orientes, [y, x, s, orientation]))

    return points_cles_orientes


def rotationGradient(point_cle,L,n_pixel):
    # Applique la rotation sur les deux gradients l'image blurred L pour le point clé considéré
    y, x, s, theta = point_cle
    s = s.astype(int)
    L = L[:, :, s]
    # Rotation des gradients par rapport au point x,y dans la région de n_pixel
    L_grady, L_gradx = gradient(L)
    # Gradients avec rotations
    L_gradx_region = np.zeros((n_pixel, n_pixel))
    L_grady_region = np.zeros((n_pixel, n_pixel))
    for i in range(0, n_pixel):
        for j in range(0, n_pixel):
            i_ref = (i - n_pixel / 2 - 1)  # coordonnée i recentrée
            j_ref = (j - n_pixel / 2 - 1)  # coordonnée j recentrée
            # Matrice de rotation
            mat_rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            # Coordonnées des points de la rotation
            i_rot, j_rot = np.dot(mat_rot, np.array([i_ref, j_ref]))
            y_rot, x_rot = np.floor(np.array([i_rot, j_rot]) + np.array([y, x])).astype(int)
            # Enregistrement des régions de rotation
            L_gradx_region[i, j] = L_gradx[y_rot, x_rot]
            L_grady_region[i, j] = L_grady[y_rot, x_rot]

    return L_gradx_region,L_grady_region

def descripteurPointCle(point_cle,L,n_pixel):
    y, x, s, theta = point_cle
    s = s.astype(int)
    L = L[:, :, s]
    descripteur = np.array([y, x])
    ## Calcul de l'histogramme sur chaque sous-région (zone)

    # Filtre gaussien appliqué à la région
    G = gaussian_filter(n_pixel, 1.5 * sigma_list[s])  # Filtre gaussien
    # Matrices de magnitude et de rotation
    mat_m = np.sqrt(2 * L_gradx_region ** 2 + (2 * L_grady_region) ** 2)
    mat_theta = np.arctan2(L_grady_region, L_gradx_region) + np.pi

    # Pondération de chaque pixel (double pondération)
    ponderation = G / mat_m  # Pondération de chaque pixel

    # Compteur de chaque zone (row major order)
    zone_count = 0
    # Application de l'histogramme

    for i in range(0, n_zone):
        for j in range(0, n_zone):
            # Définition des intervalles pour l'histogramme
            intervalles = np.linspace(0, 2 * np.pi, n_bins + 1)
            hist = np.zeros(n_bins)
            # Limites de la zone
            zone_y = slice(n_pixel_zone * i, n_pixel_zone * (i + 1))
            zone_x = slice(n_pixel_zone * j, n_pixel_zone * (j + 1))
            mat_theta_zone = mat_theta[zone_y, zone_x].flatten()
            ponderation_zone = ponderation[zone_y, zone_x].flatten()
            # Remplissage de chacun des intervalles
            # TODO : Plafonnement des valeurs
            for k in range(0, n_bins):
                bool1 = mat_theta_zone > intervalles[k]
                bool2 = mat_theta_zone < intervalles[
                    k + 1]  # Obligé de créer des variables, triste langage de programmation..
                # Numéros des pixels situés dans l'intervalle en question
                pixels_intervalle = np.nonzero(bool1 & bool2)
                # Pondération par la fenêtre gaussienne et normalisation par l'amplitude
                hist[k] = sum(ponderation_zone[pixels_intervalle])
            descripteur = np.concatenate([descripteur, hist])
    return descripteur
