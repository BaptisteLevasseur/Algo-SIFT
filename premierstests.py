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



#permet d'afficher une image; l'image se fermera en appuyant sur une touche du clavier
def displayImage(image, title = 'image'):
    ratio = image.shape[1]/image.shape[0]
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', int(800*ratio), 800 )
    cv2.imshow("image", image)
    cv2.waitKey(0)


def castToGrayScale(image):
    image_gray = np.zeros(image.shape[0:2])
    image_gray[:] = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    return image_gray


# full pipeline
def main():
    t1=time.time()
    image_initiale = mpimg.imread("lena.jpg")[:, :, 1]
    image=image_initiale/255

    nb_octave = 4
    s=3



    # Plot la pyramide de gaussienne

    print("Pyramide de Gaussiennes")
    L,sigma_list = pyramideDeGaussiennes(image, s, 0)
    # f,axarr = plt.subplots(2,3)
    # axarr[0,0].imshow(L[:,:,0],cmap='gray')
    # axarr[0,1].imshow(L[:,:,1],cmap='gray')
    # axarr[0,2].imshow(L[:,:,2],cmap='gray')
    # axarr[1,0].imshow(L[:,:,3],cmap='gray')
    # axarr[1,1].imshow(L[:,:,4],cmap='gray')
    # axarr[1,2].imshow(L[:,:,5],cmap='gray')

    # Plot la différence de gaussienne
    print("Différence de Gaussiennes")
    DoG_list, sigma_list = differenceDeGaussiennes(image, s, nb_octave)
    # f, axarr = plt.subplots(2, 3)
    # axarr[0, 0].imshow(DoG[:, :, 0],cmap='gray')
    # axarr[0, 1].imshow(DoG[:, :, 1],cmap='gray')
    # axarr[0, 2].imshow(DoG[:, :, 2],cmap='gray')
    # axarr[1, 0].imshow(DoG[:, :, 3],cmap='gray')
    # axarr[1, 1].imshow(DoG[:, :, 4],cmap='gray')
    # plt.show()


    r=10
    seuil_contraste=0.03
    n, m = np.shape(image)

    extrema_final_list = []

    for DoG in DoG_list:
        t = time.time()
        print("Détection d'extrema et ")
        print("Elimination des points situés sur les bords de l'image")
        extrema= suppressionBordsImage(detectionExtrema(DoG), m, n, 8)
        print("Elimination des faibles contrastes")
        extrema_contraste=detectionContraste(DoG,extrema,seuil_contraste)
        print("Elimination des arêtes")
        extrema_bords=detectionEdges(DoG, r, extrema_contraste)
        extrema_final_list.append(extrema_bords)
        t2 = time.time() - t
        print("{0:.2f} secondes".format(t2))

    n_zone = 4  # Donc 4x4 zones
    n_pixel_zone = 4  # Donc 4x4 pixel par zone
    n_bins = 8

    final_descriptor_list = []

    for octave in range(nb_octave):
        L_list, sigma_list = pyramideDeGaussiennes(image, s, octave)
        points_cles_orientes_list = orientationPointsCles(extrema_final_list[octave], L_list, sigma_list)

        descripteurs_list = np.empty((0, n_zone ** 2 * n_bins + 2))
        for i in range(0, np.size(points_cles_orientes_list, 0)):
            point_cle = points_cles_orientes_list[i]
            L_grady_region, L_gradx_region = rotationGradient(point_cle, L_list, 16)
            descripteur = descripteurPointCle(point_cle, L_list, sigma_list, L_grady_region, L_gradx_region, n_pixel_zone,
                                          n_zone, n_bins)
            descripteurs_list = np.vstack((descripteurs_list, descripteur))
            final_descriptor_list.append(descripteurs_list)
    pass

if __name__ == "__main__":
    main()