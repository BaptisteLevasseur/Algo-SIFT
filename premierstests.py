import numpy as np
import math
import sys
import cv2
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
    n = 10 # Taille du masque
    k = 2 ** (1 / s)
    sigma = 1.6 # Choix le sigma initial

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

def gradient(image):
    n,m = np.shape(image)
    gradx=np.zeros((n,m))
    grady = np.zeros((n, m))

    gradx[:,1:-1] = (image[:,2:m]-image[:,0:m-2])/2
    gradx[:,-1] = image[:, -1] - image[:, -2]
    gradx[:, 0] = image[:, 1] - image[:, 0]

    grady[1:-1,:] = (image[2:n, :] - image[0:m - 2, :]) / 2
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
    #
    n,m, nb_sigma= np.shape(DoG)
    extrema_list = np.empty((0, 3), int)
    for s in range(1,nb_sigma-1):
        for y in range(1,n-1):
            for x in range(1,m-1):
                if np.argmax(DoG[y - 1:y + 2, x - 1:x + 2, s - 1:s + 2])==13:
                    extrema_list = np.vstack((extrema_list, [y,x,s]))
    return extrema_list

def detectionContraste(DoG,extrema_list):
    # Il faudra rajouter l'interpolation (je connais pas la théorie sur les dérivées vectorielles)
    list_size = np.size(extrema_list, 0)
    contraste = np.ones(list_size, dtype=bool)

    for i in range(0, list_size):
        x = extrema_list[i, :]
        if abs(DoG[tuple(x)]) < 0.03:
            contraste[i] = False

    extrema_contraste_list = extrema_list[contraste]
    return extrema_contraste_list

def detectionBords(DoG,r,extrema_list):
    list_size = np.size(extrema_list, 0)
    bord = np.ones(list_size, dtype=bool)

    x = extrema_list[:, 0]
    y = extrema_list[:, 1]
    s = extrema_list[:, 2]

    for i in range(0, list_size):
        D = DoG[:, :, s[i]]
        [[Dxx, Dxy], [Dyx, Dyy]] = hessienne(D)
        TrH = Dxx[y[i], x[i]] + Dyy[y[i], x[i]]
        DetH = Dxx[y[i], x[i]] * Dyy[y[i], x[i]] - (Dxy[y[i], x[i]]) ** 2
        if TrH ** 2 / DetH >= (r + 1) ** 2 / r:
            bord[i] = False
    extrema_bords_list = extrema_list[bord]
    return extrema_bords_list

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

    nb_octave = 1
    s=3


    # Plot la pyramide de gaussienne

    image_conv = pyramideDeGaussiennes(image, s, nb_octave)
    # f,axarr = plt.subplots(2,3)
    # axarr[0,0].imshow(image_conv[:,:,0])
    # axarr[0,1].imshow(image_conv[:,:,1])
    # axarr[0,2].imshow(image_conv[:,:,2])
    # axarr[1,0].imshow(image_conv[:,:,3])
    # axarr[1,1].imshow(image_conv[:,:,4])
    # axarr[1,2].imshow(image_conv[:,:,5])

    # Plot la différence de gaussienne
    DoG, sigma_list = differenceDeGaussiennes(image, s, nb_octave)
    # f, axarr = plt.subplots(2, 3)
    # axarr[0, 0].imshow(DoG[:, :, 0],cmap='gray')
    # axarr[0, 1].imshow(DoG[:, :, 1],cmap='gray')
    # axarr[0, 2].imshow(DoG[:, :, 2],cmap='gray')
    # axarr[1, 0].imshow(DoG[:, :, 3],cmap='gray')
    # axarr[1, 1].imshow(DoG[:, :, 4],cmap='gray')
    # plt.show()
    r=10
    n, m = np.shape(image)
    extrema= detectionExtrema(DoG)
    extrema_contraste=detectionContraste(DoG,extrema)
    extrema_bords=detectionBords(DoG, r, extrema_contraste)
    x=m-extrema[:,0]
    y=n-extrema[:,1]
    x_contraste =m- extrema_contraste[:, 0]
    y_contraste =n- extrema_contraste[:, 1]
    x_bords =m- extrema_bords[:, 0]
    y_bords =n- extrema_bords[:, 1]

    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(image, cmap='gray')
    axarr[1].imshow(image, cmap='gray')
    axarr[2].imshow(image, cmap='gray')

    axarr[0].scatter(x,y) #Ou (y,x)?
    axarr[1].scatter(x_contraste, y_contraste)  # Ou (y,x)?
    axarr[2].scatter(x_bords, y_bords)  # Ou (y,x)?
    plt.show()



if __name__ == "__main__":
    main()
