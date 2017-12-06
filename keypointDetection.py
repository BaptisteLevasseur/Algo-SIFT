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

def detectionContraste(DoG,extrema_list,seuil_contraste,D_grad,D_H):
    # Il faudra rajouter l'interpolation (je connais pas la théorie sur les dérivées vectorielles)
    list_size = np.size(extrema_list, 0)
    contraste = np.ones(list_size, dtype=bool)

    for i in range(0, list_size):
        x = extrema_list[i, :] # Vecteur x = y,x,s

        ########################
        #TODO : Valeurs des offsets trop élevé

        #Calcul des dérivées premières et secondaires au point X
        grady, gradx, grads = D_grad
        D1 = [grady[tuple(x)],gradx[tuple(x)],grads[tuple(x)]]
        [Dyy, Dyx, Dys], [Dxy, Dxx, Dxs], [Dsy, Dsx, Dss] = D_H
        D2=[Dyy[tuple(x)],Dyx[tuple(x)],Dys[tuple(x)]],\
           [Dxy[tuple(x)],Dxx[tuple(x)],Dxs[tuple(x)]],\
           [Dsy[tuple(x)],Dsx[tuple(x)],Dss[tuple(x)]]

        # Calcul de l'offset estimé
        # Limite à revoir, le déterminant trop faible donne des trop grandes valeurs d'offset
        if np.linalg.det(D2) > 10**(-10):
            y_e,x_e,s_e=-np.dot(np.linalg.inv(D2),D1)
            if abs(y_e)>0.5:
                y_e=np.sign(y_e)*(1-y_e)
                x[0]+=np.sign(y_e)
            if abs(x_e)>0.5:
                x_e=np.sign(x_e)*(1-x_e)
                x[1]+=np.sign(x_e)
            if abs(s_e)>0.5:
                s_e=np.sign(s_e)*(1-s_e)
                x[2] += np.sign(s_e)
            x_est=[y_e,x_e,s_e]
        else:
            x_est=[0, 0, 0]
        # Contraste interpolé
        D=DoG[tuple(x)]+1/2*np.dot(np.transpose(D1),x_est)
        ########################


        #Remplacer DoG[tuple(x)] par D si ça marche
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

    D_grad=gradient(DoG)
    D_H = hessienne(DoG)

    extrema = detectionExtrema(DoG)
    xsize, ysize = DoG.shape[0:2]
    # pour avoir des valeurs valides après rotation (dans la partie descripteur),
    # on enlève les points à 8*sqrt(2) du bord, soit à moins de 12
    extrema_bords = suppressionBordsImage(extrema, xsize, ysize, 12)

    extrema_contraste = detectionContraste(DoG, extrema_bords,seuil_contraste,D_grad,D_H)
    extrema_edges = detectionEdges(DoG, r_courb_principale, extrema_contraste)

    #extrema_bords[:,0:2] = extrema_bords[:,0:2]*resolution_octave #Compense le downscaling pour les afficher sur l'image finale
    return extrema_edges