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




#2.1 Construction de la différence de gaussiennes





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



#2.2 Détection des points clés
def detectionPointsCles(DoG, sigma, seuil_contraste, r_courb_principale, resolution_octave):
    # Pourquoi a t'on besoin de sigma?
    extrema = detectionExtrema(DoG)
    extrema_contraste = detectionContraste(DoG, extrema,seuil_contraste)
    extrema_bords = detectionBords(DoG, r_courb_principale, extrema_contraste)
    extrema_bords[:,0:2] = extrema_bords[:,0:2]*resolution_octave #Compense le downscaling pour les afficher sur l'image finale
    return extrema_bords,sigma



#2.3 Descripteurs


#wrapper for convolution
def conv(image1,mask):
    return scipy.signal.convolve2d(image1,mask)


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
    # filename = 'droite.jpg']
    # image = cv2.imread(filename)
    # image = image/255
    # print("size of loaded picture: " + str(image.shape))
    # #grayscale conversion :
    # image_gray = castToGrayScale(image)

    # displayImage(image)
    # displayImage(image_gray, 'grayscale')

    image_initiale = mpimg.imread("lena.jpg")[:, :, 1]
    image_initiale=image_initiale/255

    nb_octave = 1
    s=3

    image = image_initiale[0::nb_octave, 0::nb_octave]


    # Plot la pyramide de gaussienne

    print("Pyramide de Gaussiennes")
    L,sigma_list = pyramideDeGaussiennes(image, s, nb_octave)
    # f,axarr = plt.subplots(2,3)
    # axarr[0,0].imshow(L[:,:,0],cmap='gray')
    # axarr[0,1].imshow(L[:,:,1],cmap='gray')
    # axarr[0,2].imshow(L[:,:,2],cmap='gray')
    # axarr[1,0].imshow(L[:,:,3],cmap='gray')
    # axarr[1,1].imshow(L[:,:,4],cmap='gray')
    # axarr[1,2].imshow(L[:,:,5],cmap='gray')

    # Plot la différence de gaussienne
    print("Différence de Gaussiennes")
    DoG, sigma_list = differenceDeGaussiennes(image, s, nb_octave)
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
    #
    print("Détection d'extrema")
    extrema= detectionExtrema(DoG)
    print("Elimination des faibles contrastes")
    extrema_contraste=detectionContraste(DoG,extrema,seuil_contraste)
    print("Elimination des bords")
    extrema_bords=detectionBords(DoG, r, extrema_contraste)
    # #

    t2=time.time()
    print(t2-t1)

    y=extrema[:,0]
    x=extrema[:,1]
    y_contraste =extrema_contraste[:, 0]
    x_contraste =extrema_contraste[:, 1]
    y_bords =extrema_bords[:, 0]
    x_bords =extrema_bords[:, 1]
    # #
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(DoG[:,:,1], cmap='gray')
    axarr[1].imshow(DoG[:,:,1], cmap='gray')
    axarr[2].imshow(DoG[:,:,1], cmap='gray')
    #
    axarr[0].scatter(x,y) #Ou (y,x)?
    axarr[1].scatter(x_contraste,y_contraste)  # Ou (y,x)?
    axarr[2].scatter(x_bords,y_bords)  # Ou (y,x)?
    #
    #
    # # f, axarr = plt.subplots(3,3)
    # # axarr[0,0].imshow(image, cmap='gray')
    # # axarr[0,1].imshow(image, cmap='gray')
    # # axarr[0,2].imshow(image, cmap='gray')
    # # axarr[1, 0].imshow(image, cmap='gray')
    # # axarr[1, 1].imshow(image, cmap='gray')
    # # axarr[1, 2].imshow(image, cmap='gray')
    # # axarr[2, 0].imshow(image, cmap='gray')
    # # axarr[2, 1].imshow(image, cmap='gray')
    # #
    # # axarr[0,0].scatter(x, y)  # Ou (y,x)?
    # # axarr[0,1].scatter(n-x, n-y)  # Ou (y,x)?
    # # axarr[0, 2].scatter(n-x, y)  # Ou (y,x)?
    # # axarr[1, 0].scatter(x, n-y)  # Ou (y,x)?
    # # axarr[1, 1].scatter(y, x)  # Ou (y,x)?
    # # axarr[1, 2].scatter(n-y, n-x)  # Ou (y,x)?
    # # axarr[2, 0].scatter(n-y, x)  # Ou (y,x)?
    # # axarr[2, 1].scatter(y, n-x)  # Ou (y,x)?
    # # print(compteurExtrema(image_initiale,s,nb_octave,r,seuil_contraste))
    # print(orientationPointsCles(L,extrema_bords))
    plt.show()



if __name__ == "__main__":
    main()
