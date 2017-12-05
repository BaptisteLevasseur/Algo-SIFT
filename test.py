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

points_cles_list=np.array([[  2,  2 , 1],
 [  2,508 , 1],
 [ 72,291 , 1],
 [116,418 , 1],
 [178,289 , 1],
 [190,274 , 1],
 [228,136 , 1],
 [252,273 , 1],
 [259,143 , 1],
 [264,109 , 1],
 [270,104 , 1],
 [272,117 , 1],
 [276, 94 , 1],
 [295,165 , 1],
 [307, 93 , 1],
 [309,172 , 1],
 [313,205 , 1],
 [329,148 , 1],
 [332,175 , 1],
 [344,148 , 1],
 [349,310 , 1],
 [356,304 , 1],
 [368,114 , 1],
 [392,129 , 1],
 [396,135 , 1],
 [397, 67 , 1],
 [401,116 , 1],
 [403, 98 , 1],
 [405, 93 , 1],
 [407,348 , 1],
 [420,163 , 1],
 [426,177 , 1],
 [430,171 , 1],
 [445,168 , 1],
 [458,173 , 1],
 [498,136 , 1],
 [504,133 , 1],
 [509,169 , 1],
 [509,379 , 1],
 [509,410 , 1],
 [509,509 , 1],
 [ 60,457 , 2],
 [ 79,293 , 2],
 [ 80,509 , 2],
 [ 93,311 , 2],
 [105,324 , 2],
 [114,322 , 2],
 [114,386 , 2],
 [120,379 , 2],
 [126,371 , 2],
 [128,126 , 2],
 [151,411 , 2],
 [176,401 , 2],
 [187,390 , 2],
 [203,383 , 2],
 [210,373 , 2],
 [238,226 , 2],
 [240,366 , 2],
 [262,282 , 2],
 [267,337 , 2],
 [272,128 , 2],
 [278,102 , 2],
 [295,172 , 2],
 [311,185 , 2],
 [330,141 , 2],
 [352,159 , 2],
 [371,157 , 2],
 [377,508 , 2],
 [379,121 , 2],
 [386,313 , 2],
 [396,321 , 2],
 [397,323 , 2],
 [399,327 , 2],
 [401,  2 , 2],
 [403,445 , 2],
 [409,477 , 2],
 [422,170 , 2],
 [425,354 , 2],
 [428,183 , 2],
 [433,103 , 2],
 [434,151 , 2],
 [440,172 , 2],
 [465,373 , 2],
 [476,423 , 2],
 [483,378 , 2],
 [490,380 , 2],
 [494,381 , 2],
 [509, 56 , 2],
 [  3,416 , 3],
 [ 61,449 , 3],
 [177,244 , 3],
 [177,388 , 3],
 [187,365 , 3],
 [242,341 , 3],
 [248,136 , 3],
 [259,268 , 3],
 [264,249 , 3],
 [264,324 , 3],
 [267,275 , 3],
 [269,337 , 3],
 [279,120 , 3],
 [312,373 , 3],
 [316,316 , 3],
 [320,384 , 3],
 [322,177 , 3],
 [337,170 , 3],
 [350,160 , 3],
 [462,153 , 3],
 [476,415 , 3],
 [476,439 , 3],
 [486,426 , 3],
 [508, 20 , 3],
 [508,475 , 3]])
nb_octave = 0
s=3

image = image_initiale[0::nb_octave+1, 0::nb_octave+1]
L, sigma_list = pyramideDeGaussiennes(image, s, nb_octave)

# points_cles_orientes_list = np.empty((0, 4))
# for i in range(0,np.size(points_cles_list)):
#  point_cle=points_cles_list[i] # Pour le test
#  points_cles_orientes=orientationPointCle(point_cle,L,sigma_list)
#  points_cles_orientes_list=np.concatenate(points_cles_orientes_list,points_cles_orientes)
#
# print(points_cles_orientes_list)

print(orientationPointCle(points_cles_list[2],L,sigma_list))
print(points_cles_list[0])

