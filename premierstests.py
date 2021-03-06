import numpy as np
import math
import sys
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as sig
import time
from keypointDetection import *
from scaleSpace import *
from timeDecorator import timeit
from keypointDescriptor import *
import matchingPoints




# full pipeline
def getDescriptors(image_name):
    print("Computing descriptors of "+image_name+"...")
    t1=time.time()
    #On réalise les calculs sur le canal rouge de l'image
    image_initiale = mpimg.imread(image_name)[:, :, 1]
    image = image_initiale/255

    nb_octave = 4
    s=3

    print("Différence de Gaussiennes")
    DoG_list, sigma_list = differenceDeGaussiennes(image, s, nb_octave)


    #extrema / points clés
    r_courb_principale=10
    seuil_contraste=0.03
    extrema_final_list = []
    for octave in range(nb_octave):
        print("Detection des points-clés dans l'octave " + str(octave))
        t = time.time()
        DoG = DoG_list[octave]
        extrema = detectionPointsCles(DoG, sigma_list, seuil_contraste, r_courb_principale, octave)
        extrema_final_list.append(extrema)
        t2 = time.time() - t
        print("Calcul effectué en {0:.2f} secondes".format(t2))


    #descripteurs
    n_zone = 4  # Donc 4x4 zones
    n_pixel_zone = 4  # Donc 4x4 pixel par zone
    n_bins = 8

    final_descriptor_list = np.empty((0, n_zone ** 2 * n_bins + 2))
    print("Computing Descriptors...")
    for octave in range(nb_octave):
        L_list, sigma_list = pyramideDeGaussiennes(image, s, octave)
        points_cles_orientes_list = orientationPointsCles(extrema_final_list[octave], L_list, sigma_list)

        descripteurs_list = np.empty((0, n_zone ** 2 * n_bins + 2))
        for i in range(0, np.size(points_cles_orientes_list, 0)):
            point_cle = points_cles_orientes_list[i]
            L_grady_region, L_gradx_region = rotationGradient(point_cle, L_list, 16)
            descripteur = descripteurPointCle(point_cle, L_list, sigma_list, L_grady_region, L_gradx_region, n_pixel_zone,
                                          n_zone, n_bins)
            #on remet les coordonnées du descripteur à l'échelle de l'image de base
            descripteur[0:2] = descripteur[0:2]*(2**octave)
            descripteurs_list = np.vstack((descripteurs_list, descripteur))
        final_descriptor_list = np.vstack((final_descriptor_list, descripteurs_list))
    return final_descriptor_list


if __name__ == "__main__":
    image1 = "Redgauche.jpg"
    image2 = "Reddroite.jpg"

    # à utiliser (True) si on a déjà les descripteurs
    loadDesc = True

    d1 = None
    d2 = None
    if loadDesc:
        print("Loading descriptors...")
        try:
            d1 = np.loadtxt('desc_'+image1+'.txt')
            d2 = np.loadtxt('desc_'+image2+'.txt')
        except FileNotFoundError:
            print("Files not found. Calculating descriptors...")
            d1 = getDescriptors(image1)
            d2 = getDescriptors(image2)
            np.savetxt('desc_' + image1 + '.txt', d1)
            np.savetxt('desc_' + image2 + '.txt', d2)
    else:
        d1 = getDescriptors(image1)
        d2 = getDescriptors(image2)
        print("Saving descriptors on disk for next time !")
        np.savetxt('desc_' + image1 + '.txt', d1)
        np.savetxt('desc_' + image2 + '.txt', d2)
    print('Entering final pipeline...')
    matchingPoints.final_pipeline(d1, d2, image1, image2)
