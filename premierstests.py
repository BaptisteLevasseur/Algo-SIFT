import numpy as np
import math
import sys
import cv2
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as sig
import time




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
    n = 21 # Taille du masque
    k = 2 ** (1 / s)
    n_pad = int((n - 1) / 2)
    sigma = 1.6 # Choix le sigma initial
    n_im, m_im = np.shape(image_initiale)
    image_octave = image_initiale[0:n_im:nb_octave, 0:m_im:nb_octave]
    n_im, m_im = np.shape(image_octave)
    n_im_pad = n_im + n - 1 # Je sais pas trop pourquoi, encore ces fucking indices!
    m_im_pad = m_im + n - 1
    gaussienne_list=np.zeros((n_im, m_im, s + 3))
    gaussienne_list_pad = np.zeros((n_im_pad, m_im_pad, s + 3))
    sigma_list=np.zeros(s+3)
    for j in range(0, s + 3):
        G = gaussian_filter(n, (k ** j) * sigma)
        sigma_list[j]=k ** j * sigma
        gaussienne_list_pad[:, :, j] = sig.convolve2d(image_octave, G)
    gaussienne_list = gaussienne_list_pad[n_pad:n_im_pad - n_pad, n_pad:m_im_pad - n_pad,:]
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

def gradient(image):
    n,m = np.shape(image)
    gradx=np.zeros((n,m))
    grady = np.zeros((n, m))

    gradx[:,1:-1] = (image[:,2:m]-image[:,0:m-2])/2
    gradx[:,-1] = image[:, -1] - image[:, -2]
    gradx[:, 0] = image[:, 1] - image[:, 0]

    grady[1:-1,:] = (image[2:n, :] - image[0:n - 2, :]) / 2
    grady[-1,:] = image[-1,:] - image[-2,:]
    grady[0,:] = image[1,:] - image[0,:]
    return [grady,gradx]

def hessienne(image):
    # AXES??? Pour l'instant le x est vers la droite et le y vers le bas
    Dy, Dx = gradient(image)
    Dyy, Dyx = gradient(Dy)
    Dxy, Dxx = gradient(Dx)
    return [[Dxx,Dxy],[Dyx,Dyy]]

def detectionExtrema(DoG):
    # Pour l'instant on se contente de regarder à l'intérieur du cube de l'octave
    # Il faudra gérer les effets au niveau des bords du cube
    # Renvoyé pour les indices (i,j,s) = (y,x,s)
    n,m, nb_sigma= np.shape(DoG)
    extrema_list = np.empty((0, 3), int)
    for s in range(1,nb_sigma-1):
        for y in range(1,n-1):
            for x in range(1,m-1):
                # Si le maximum au centre des 24 pixels
                maxi=np.argmax(DoG[y - 1:y + 2, x - 1:x + 2, s - 1:s + 2])==13
                mini=np.argmin(DoG[y - 1:y + 2, x - 1:x + 2, s - 1:s + 2])==13
                if maxi or mini: # or maxi (sur une image grayscale de détection de contour, les bords sont en noir => minimums)
                    extrema_list = np.vstack((extrema_list, [y,x,s]))

    print(np.size(extrema_list,0))
    return extrema_list

def detectionContraste(DoG,extrema_list,seuil_contraste):
    # Il faudra rajouter l'interpolation (je connais pas la théorie sur les dérivées vectorielles)
    list_size = np.size(extrema_list, 0)
    contraste = np.ones(list_size, dtype=bool)

    for i in range(0, list_size):
        x = extrema_list[i, :] # Vecteur x = [y,x,s]
        if abs(DoG[tuple(x)]) < seuil_contraste:
            contraste[i] = False

    extrema_contraste_list = extrema_list[contraste]
    print(np.size(extrema_contraste_list, 0))
    return extrema_contraste_list

def detectionBords(DoG,r,extrema_list):
    list_size = np.size(extrema_list, 0)
    bord = np.ones(list_size, dtype=bool)

    y = extrema_list[:, 0]
    x = extrema_list[:, 1]
    s = extrema_list[:, 2]

    for i in range(0, list_size):
        D = DoG[:, :, s[i]]
        [[Dxx, Dxy], [Dyx, Dyy]] = hessienne(D)
        TrH = Dxx[y[i], x[i]] + Dyy[y[i], x[i]]
        DetH = Dxx[y[i], x[i]] * Dyy[y[i], x[i]] - (Dxy[y[i], x[i]]) ** 2
        if TrH ** 2 / DetH >= (r + 1) ** 2 / r:
            bord[i] = False
    extrema_bords_list = extrema_list[bord]
    print(np.size(extrema_bords_list, 0))
    return extrema_bords_list

def compteurExtrema(image_initiale,s,nb_octave,r,seuil_contraste):
    DoG, sigma_list = differenceDeGaussiennes(image_initiale, s, nb_octave)
    extrema= detectionExtrema(DoG)
    extrema_contraste=detectionContraste(DoG,extrema,seuil_contraste)
    extrema_bords=detectionBords(DoG, r, extrema_contraste)
    n_extrema=np.size(extrema,0)
    n_faible_contraste = n_extrema-np.size(extrema_contraste, 0)
    n_points_arrete=n_extrema-n_faible_contraste-np.size(extrema_bords,0)
    return n_extrema,n_faible_contraste,n_points_arrete


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
    seuil_contraste=1
    n, m = np.shape(image)
    #
    print("Détection d'extrema")
    extrema= detectionExtrema(DoG)
    print("Elimination des faibles contrastes")
    extrema_contraste=detectionContraste(DoG,extrema,seuil_contraste)
    # print("Elimination des bords")
    # extrema_bords=detectionBords(DoG, r, extrema_contraste)
    # #

    t2=time.time()
    print(t2-t1)

    y=extrema[:,0]
    x=extrema[:,1]
    y_contraste =extrema_contraste[:, 0]
    x_contraste =extrema_contraste[:, 1]
    # y_bords =extrema_bords[:, 0]
    # x_bords =extrema_bords[:, 1]
    # #
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(DoG[:,:,1], cmap='gray')
    axarr[1].imshow(DoG[:,:,1], cmap='gray')
    axarr[2].imshow(DoG[:,:,1], cmap='gray')
    #
    axarr[0].scatter(x,y) #Ou (y,x)?
    axarr[1].scatter(x_contraste,y_contraste)  # Ou (y,x)?
    # axarr[2].scatter(y_bords,x_bords)  # Ou (y,x)?
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
