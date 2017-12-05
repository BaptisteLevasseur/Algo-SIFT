import numpy as np
import scipy.signal as sig
from basicOperations import *



#attention : les octaves sont numérotées à partir de zéro
def pyramideDeGaussiennes(image_initiale,s,no_octave):
    n = 21 # TODO:  Taille du masque
    k = 2 ** (1 / s)
    n_pad = int((n - 1) / 2)
    sigma = 1.6 # Choix le sigma initial
    n_im, m_im = np.shape(image_initiale)
    image_octave = image_initiale[0:n_im:2**no_octave, 0:m_im:2**no_octave]
    n_im, m_im = np.shape(image_octave)
    n_im_pad = n_im + n - 1  # (équivalent à n_im + 2*n_pad)
    m_im_pad = m_im + n - 1
    gaussienne_list_pad = np.zeros((n_im_pad, m_im_pad, s + 3))
    sigma_list=np.zeros(s+3)
    for j in range(0, s + 3):
        G = gaussian_filter(n, (k ** j) * sigma)
        sigma_list[j]=k ** j * sigma
        gaussienne_list_pad[:, :, j] = sig.convolve2d(image_octave, G)
    gaussienne_list = gaussienne_list_pad[n_pad:n_im_pad - n_pad, n_pad:m_im_pad - n_pad,:] #TODO: peut être un décalage
    return gaussienne_list, sigma_list


#attention : on renvoie une liste de sigma unique pour toutes les octaves
def differenceDeGaussiennes(image_initiale,s,nb_octave):
    DoG_all_octave = []
    for i in range(nb_octave):
        gaussienne_list, sigma_list = pyramideDeGaussiennes(image_initiale, s, i)
        n_im, m_im, s_im = np.shape(gaussienne_list)
        DoG_list = np.zeros((n_im, m_im, s_im - 1))
        for j in range(0, s_im - 1):
            DoG_list[:, :, j] = gaussienne_list[:, :, j + 1] - gaussienne_list[:, :, j]
        DoG_all_octave.append(DoG_list)
    return DoG_list, sigma_list
