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



def orientationPointsCles(L,extrema_list):
    # QUESTION : Quel sigma on choisi? La borne inférieure ou la borne supérieur? (car les points clés sont repérés
    # par rapport à la DoG et là on retourne sur L
    nb_points=np.size(extrema_list, 0)
    points_cles_liste = np.empty((0, 4))
    for i in range(0,nb_points):
        [y,x,s] = extrema_list[i, :]
        m = np.sqrt((L[y+1,x,s]-L[y-1,x,s])**2+(L[y,x+1,s]-L[y,x-1,s])**2)
        theta = np.arctan((L[y+1,x,s]-L[y-1,x,s])/(L[y,x+1,s]-L[y,x-1,s]))
        points_cles_liste = np.vstack((points_cles_liste, [y, x, m,theta]))
    return points_cles_liste
