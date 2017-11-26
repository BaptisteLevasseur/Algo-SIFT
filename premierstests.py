import numpy as np
import math
import sys
# import cv2
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as sig




#2.1 Construction de la différence de gaussiennes

def gaussian_filter(n,sigma):
    if sigma == 0:
        G = np.zeros((n, n))
        centre = int((n-1)/2)
        G[centre, centre] = 1
    else:
        x = np.linspace(-(n - 1) / 2, (n - 1) / 2, n)
        xv, yv = np.meshgrid(x, x)
        G = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(xv ** 2 + yv ** 2) / (2 * sigma ** 2))
        G = G / sum(sum(G))
    return G

def pyramideDeGaussiennes(image_initiale,s,nb_octave):
    n = 5
    k = 2 ** (1 / s)
    sigma = 1.6

    image_octave = image_initiale[0:-1:nb_octave, 0:-1:nb_octave]
    n_im, m_im = np.shape(image_octave)
    n_im += n - 1 # Je sais pas trop pourquoi, encore ces fucking indices!
    m_im += n - 1
    gaussienne_list = np.zeros((n_im, m_im, s + 3))
    sigma_list=np.zeros(s+3)
    for j in range(0, s + 3):
        G = gaussian_filter(n, (k ** j) * sigma)
        sigma_list[j]=k ** j * sigma
        gaussienne_list[:, :, j] = sig.convolve2d(image_octave, G)
    return (gaussienne_list, sigma_list)

def differenceDeGaussiennes(image_initiale,s,nb_octave):
    DoG_list=0
    sigma_list=0

    gaussienne_list, sigma_list = pyramideDeGaussiennes(image_initiale, s, nb_octave)

    n_im, m_im, s_im = np.shape(gaussienne_list)
    DoG_list = np.zeros((n_im, m_im, s_im - 1))
    for i in range(0, s_im - 1):
        DoG_list[:, :, i] = gaussienne_list[:, :, i + 1] - gaussienne_list[:, :, i]

    return (DoG_list,sigma_list);

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
    # filename = 'droite.jpg']
    # image = cv2.imread(filename)
    # image = image/255
    # print("size of loaded picture: " + str(image.shape))
    # #grayscale conversion :
    # image_gray = castToGrayScale(image)

    # displayImage(image)
    # displayImage(image_gray, 'grayscale')

    image = mpimg.imread("lena.jpg")[:, :, 1]

    nb_octave = 2
    s=3

    image_conv = pyramideDeGaussiennes(image, s, nb_octave)


    # f,axarr = plt.subplots(2,3)
    # axarr[0,0].imshow(image_conv[:,:,0])
    # axarr[0,1].imshow(image_conv[:,:,1])
    # axarr[0,2].imshow(image_conv[:,:,2])
    # axarr[1,0].imshow(image_conv[:,:,3])
    # axarr[1,1].imshow(image_conv[:,:,4])
    # axarr[1,2].imshow(image_conv[:,:,5])

    L, sigma_list = differenceDeGaussiennes(image, s, nb_octave)

    f, axarr = plt.subplots(2, 3)
    axarr[0, 0].imshow(L[:, :, 0])
    axarr[0, 1].imshow(L[:, :, 1])
    axarr[0, 2].imshow(L[:, :, 2])
    axarr[1, 0].imshow(L[:, :, 3])
    axarr[1, 1].imshow(L[:, :, 4])

    plt.show()


if __name__ == "__main__":
    main()
