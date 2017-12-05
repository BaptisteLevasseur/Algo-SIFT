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

n_zone=4 #Donc 4x4 zones
n_pixel_zone=4 #Donc 4x4 pixel par zone
n_bins = 8

descripteurs_list=np.empty((0, n_zone**2*n_bins+2))
for i in range(0,np.size(points_cles_orientes_list,0)):
    point_cle=points_cles_orientes_list[i]
    L_grady_region,L_gradx_region=rotationGradient(point_cle,L_list,16)
    descripteur=descripteurPointCle(point_cle,L_list,sigma_list,L_grady_region,L_gradx_region,n_pixel_zone,n_zone,n_bins)
    descripteurs_list=np.vstack((descripteurs_list,descripteur))


np.savetxt('test1.txt', descripteurs_list)

#
# plt.imshow(descripteur_region_gradx,cmap='gray')
# plt.show()
#


