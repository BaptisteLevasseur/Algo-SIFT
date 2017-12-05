import numpy as np
import math
import sys
import cv2
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as sig
import time
from keypointDetection import *
from scaleSpace import *
from timeDecorator import timeit
from keypointDescriptor import *
from keypointDetection import *
from scaleSpace import *

image_initiale = mpimg.imread("lena.jpg")[:, :, 1]
image_initiale = image_initiale / 255

points_cles_list=np.array([[ 72,291,1],
 [116,418,1],
 [178,289,1],
 [190,274,1],
 [228,136,1],
 [252,273,1],
 [259,143,1],
 [264,109,1],
 [270,104,1],
 [272,117,1],
 [276, 94,1],
 [295,165,1],
 [307, 93,1],
 [309,172,1],
 [313,205,1],
 [329,148,1],
 [332,175,1],
 [344,148,1],
 [349,310,1],
 [356,304,1],
 [368,114,1],
 [392,129,1],
 [396,135,1],
 [397, 67,1],
 [401,116,1],
 [403, 98,1],
 [405, 93,1],
 [407,348,1],
 [420,163,1],
 [426,177,1],
 [430,171,1],
 [445,168,1],
 [458,173,1],
 [498,136,1],
 [ 60,457,2],
 [ 79,293,2],
 [ 93,311,2],
 [105,324,2],
 [114,322,2],
 [114,386,2],
 [120,379,2],
 [126,371,2],
 [128,126,2],
 [151,411,2],
 [176,401,2],
 [187,390,2],
 [203,383,2],
 [210,373,2],
 [238,226,2],
 [240,366,2],
 [262,282,2],
 [267,337,2],
 [272,128,2],
 [278,102,2],
 [295,172,2],
 [311,185,2],
 [330,141,2],
 [352,159,2],
 [371,157,2],
 [379,121,2],
 [386,313,2],
 [396,321,2],
 [397,323,2],
 [399,327,2],
 [403,445,2],
 [409,477,2],
 [422,170,2],
 [425,354,2],
 [428,183,2],
 [433,103,2],
 [434,151,2],
 [440,172,2],
 [465,373,2],
 [476,423,2],
 [483,378,2],
 [490,380,2],
 [494,381,2],
 [ 61,449,3],
 [177,244,3],
 [177,388,3],
 [187,365,3],
 [242,341,3],
 [248,136,3],
 [259,268,3],
 [264,249,3],
 [264,324,3],
 [267,275,3],
 [269,337,3],
 [279,120,3],
 [312,373,3],
 [316,316,3],
 [320,384,3],
 [322,177,3],
 [337,170,3],
 [350,160,3],
 [462,153,3],
 [476,415,3],
 [476,439,3],
 [486,426,3]])
nb_octave = 0
s=3

image = image_initiale[0::nb_octave+1, 0::nb_octave+1]
L_list, sigma_list = pyramideDeGaussiennes(image, s, nb_octave)

points_cles_orientes_list=orientationPointsCles(points_cles_list,L_list, sigma_list)


i=10
# rotationGradient(L_list,points_cles_orientes_list[i],16)
y,x,s,theta=points_cles_orientes_list[i]
s=s.astype(int)
L=L_list[:,:,s]
n_zone=4 #Donc 4x4 zones
n_pixel_zone=4 #Donc 4x4 pixel par zone
n_bins = 8
n_pixel=n_zone*n_pixel_zone #


descripteur=np.array([y,x])
descripteurs_list=np.empty((0, n_zone**2*n_bins+2))

# Rotation des gradients par rapport au point x,y dans la région de n_pixel
L_grady, L_gradx = gradient(L)
# Gradients avec rotations
L_gradx_region=np.zeros((n_pixel, n_pixel))
L_grady_region=np.zeros((n_pixel, n_pixel))
for i in range(0,n_pixel):
 for j in range(0,n_pixel):
  i_ref=(i-n_pixel/2-1) # coordonnée i recentrée
  j_ref=(j-n_pixel/2-1) # coordonnée j recentrée
  # Matrice de rotation
  mat_rot=np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
  # Coordonnées des points de la rotation
  i_rot,j_rot=np.dot(mat_rot,np.array([i_ref,j_ref]))
  y_rot,x_rot=np.floor(np.array([i_rot,j_rot])+np.array([y,x])).astype(int)
  # Enregistrement des régions de rotation
  L_gradx_region[i,j]=L_gradx[y_rot,x_rot]
  L_grady_region[i, j] = L_grady[y_rot, x_rot]

## Calcul de l'histogramme sur chaque sous-région (zone)

# Filtre gaussien appliqué à la région
G = gaussian_filter(n_pixel, 1.5 * sigma_list[s]) # Filtre gaussien
# Matrices de magnitude et de rotation
mat_m = np.sqrt(2 * L_gradx_region ** 2 + (2 * L_grady_region) ** 2)
mat_theta = np.arctan2(L_grady_region,L_gradx_region) + np.pi

# Pondération de chaque pixel (double pondération)
ponderation=G/mat_m # Pondération de chaque pixel

# Compteur de chaque zone (row major order)
zone_count = 0
# Application de l'histogramme

for i in range(0, n_zone):
 for j in range(0, n_zone):
  #Définition des intervalles pour l'histogramme
  intervalles = np.linspace(0, 2 * np.pi, n_bins + 1)
  hist = np.zeros(n_bins)
  # Limites de la zone
  zone_y=slice(n_pixel_zone*i,n_pixel_zone*(i+1))
  zone_x=slice(n_pixel_zone*j,n_pixel_zone*(j+1))
  mat_theta_zone=mat_theta[zone_y,zone_x].flatten()
  ponderation_zone=ponderation[zone_y,zone_x].flatten()
  # Remplissage de chacun des intervalles
  # TODO : Plafonnement des valeurs
  for k in range(0, n_bins):
   bool1 = mat_theta_zone > intervalles[k]
   bool2 = mat_theta_zone < intervalles[k + 1]  # Obligé de créer des variables, triste langage de programmation..
   # Numéros des pixels situés dans l'intervalle en question
   pixels_intervalle = np.nonzero(bool1 & bool2)
   # Pondération par la fenêtre gaussienne et normalisation par l'amplitude
   hist[k] = sum(ponderation_zone[pixels_intervalle])
  descripteur=np.concatenate([descripteur, hist])

descripteurs_list=np.vstack((descripteurs_list,descripteur))




#
# plt.imshow(descripteur_region_gradx,cmap='gray')
# plt.show()
#


