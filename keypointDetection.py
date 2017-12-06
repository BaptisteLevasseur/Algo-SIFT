import numpy as np
from basicOperations import *


def detectionExtrema(DoG):
    # Pour l'instant on se contente de regarder à l'intérieur du cube de l'octave
    # Il faudra gérer les effets au niveau des bords du cube
    # Renvoyé pour les indices (i,j,s) = (y,x,s)
    n,m, nb_sigma= np.shape(DoG)
    extrema_list = np.empty((0, 3), int)
    #TODO: Optimiser la fonction
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

def detectionEdges(DoG,r,extrema_list):
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

def compteurExtrema(image_initiale,s,no_octave,r,seuil_contraste):
    DoG, sigma_list = differenceDeGaussiennes(image_initiale, s, no_octave)
    extrema= detectionExtrema(DoG)
    extrema_contraste = detectionContraste(DoG, extrema, seuil_contraste)
    extrema_bords = detectionBords(DoG, r, extrema_contraste)
    n_extrema=np.size(extrema, 0)
    n_faible_contraste = n_extrema-np.size(extrema_contraste, 0)
    n_points_arrete=n_extrema-n_faible_contraste-np.size(extrema_bords,0)
    return n_extrema, n_faible_contraste, n_points_arrete


#a pour but de supprimer les points clés trop près du bord pour pouvoir ensuite calculer sans soucie
#le descriptorSize décrit le "rayon" du descripteur (usuellement 8 ou 16 pixels)
def suppressionBordsImage(extrema, xSize, ySize, descriptorSize):
    l =  np.empty((0, 3), int)

    xmin = descriptorSize
    xmax = xSize - descriptorSize
    ymin = descriptorSize
    ymax = ySize - descriptorSize
    for e in extrema:
        if e[0] > xmin and e[0] < xmax and e[1] > ymin and e[1] < ymax:
            l = np.vstack((l, e))
    return l


#2.2 Détection des points clés
#on note qu'on utilise pas sigma
def detectionPointsCles(DoG, sigma, seuil_contraste, r_courb_principale, resolution_octave):
    # Pourquoi a t-on besoin de sigma? Bonne question
    extrema = detectionExtrema(DoG)
    extrema_contraste = detectionContraste(DoG, extrema,seuil_contraste)
    extrema_bords = detectionEdges(DoG, r_courb_principale, extrema_contraste)
    xsize,ysize = DoG[resolution_octave].shape[0:2]
    extrema_bords_final = suppressionBordsImage(extrema_bords, xsize,ysize, 8)
    extrema_bords[:,0:2] = extrema_bords[:,0:2]*resolution_octave #Compense le downscaling pour les afficher sur l'image finale
    return extrema_bords