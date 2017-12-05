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
    extrema= suppressionBordsImage(detectionExtrema(DoG),m, n, 8)
    print("Elimination des faibles contrastes")
    extrema_contraste=detectionContraste(DoG,extrema,seuil_contraste)
    print("Elimination des bords")
    extrema_bords=detectionEdges(DoG, r, extrema_contraste)
    # #
    print(extrema_bords)
    t2=time.time()
    print(t2-t1)

    y=extrema[:,0]
    x=extrema[:,1]
    y_contraste =extrema_contraste[:, 0]
    x_contraste =extrema_contraste[:, 1]
    y_bords =extrema_bords[:, 0]
    x_bords =extrema_bords[:, 1]
    #
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(DoG[:,:,1], cmap='gray')
    axarr[1].imshow(DoG[:,:,1], cmap='gray')
    axarr[2].imshow(DoG[:,:,1], cmap='gray')
    #
    axarr[0].scatter(x,y)
    axarr[1].scatter(x_contraste,y_contraste)
    axarr[2].scatter(x_bords,y_bords)

    plt.show()



if __name__ == "__main__":
    main()