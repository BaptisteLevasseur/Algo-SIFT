import numpy as np
import math
import sys
import cv2
import scipy.signal




#2.1 Construction de la différence de gaussiennes
def differenceDeGaussiennes(image_initiale, s, nb_octave):
    listeDesMatrices = []
    vectDesSigma = 0;
    pass
    return(listeDesMatrices,vectDesSigma)


#2.2 Détection des points clés
def detectionPointsCles(DoG, sigma, seuil_contraste, r_courb_principale, resolution_octave):
    pass



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
    filename = 'droite.jpg'
    image = cv2.imread(filename)
    image = image/255
    print("size of loaded picture: " + str(image.shape))
    #grayscale conversion :
    image_gray = castToGrayScale(image)

    displayImage(image)
    displayImage(image_gray, 'grayscale')
    #test d'un petit filtre moyenneur pour le plaisir
    mask = 1/25*np.array([[1.,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]])
    image_moyennee =conv(image_gray,mask)

    displayImage(image_moyennee,"moyenne")
    sigma = 1
    nb_octaves = 3
    difDeGauss = differenceDeGaussiennes(image_gray,sigma,nb_octaves)


if __name__ == "__main__":
    main()
