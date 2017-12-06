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
def getDescriptors(image_name):
    t1=time.time()
    image_initiale = mpimg.imread(image_name)[:, :, 1]
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


    r_courb_principale=10
    seuil_contraste=0.03
    n, m = np.shape(image)

    extrema_final_list = []

    for octave in range(nb_octave):
        t = time.time()
        DoG = DoG_list[octave]
        extrema = detectionPointsCles(DoG, sigma_list, seuil_contraste, r_courb_principale, octave)
        extrema_final_list.append(extrema)
        t2 = time.time() - t
        print("{0:.2f} secondes".format(t2))

    n_zone = 4  # Donc 4x4 zones
    n_pixel_zone = 4  # Donc 4x4 pixel par zone
    n_bins = 8

    final_descriptor_list = np.empty((0, n_zone ** 2 * n_bins + 2))

    for octave in range(nb_octave):
        L_list, sigma_list = pyramideDeGaussiennes(image, s, octave)
        points_cles_orientes_list = orientationPointsCles(extrema_final_list[octave], L_list, sigma_list)

        descripteurs_list = np.empty((0, n_zone ** 2 * n_bins + 2))
        for i in range(0, np.size(points_cles_orientes_list, 0)):
            point_cle = points_cles_orientes_list[i]
            L_grady_region, L_gradx_region = rotationGradient(point_cle, L_list, 16)
            descripteur = descripteurPointCle(point_cle, L_list, sigma_list, L_grady_region, L_gradx_region, n_pixel_zone,
                                          n_zone, n_bins)
            descripteur[0:2] = descripteur[0:2]*(2**octave)
            descripteurs_list = np.vstack((descripteurs_list, descripteur))
        final_descriptor_list = np.vstack((final_descriptor_list, descripteurs_list))
    pass



if __name__ == "__main__":
    image1 = "gaucheReduit.jpg"
    image2 = "droiteReduit.jpg"
    d1 = getDescriptors(image1)
    d2 = getDescriptors(image2)

    matchingPoints.final_pipeline(d1,d2,image1,image2)
