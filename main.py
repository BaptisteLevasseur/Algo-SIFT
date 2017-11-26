# coding: utf8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as sig


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


def differenceDeGaussiennes(image_initiale,s,nb_octave):
    DoG_list=0
    sigma_list=0

    gaussienne_list, sigma_list = pyramideDeGaussiennes(image, s, nb_octave)

    n_im, m_im, s_im = np.shape(gaussienne_list)
    DoG_list = np.zeros((n_im, m_im, s_im - 1))
    for i in range(0, s_im - 1):
        DoG_list[:, :, i] = gaussienne_list[:, :, i + 1] - gaussienne_list[:, :, i]

    return (DoG_list,sigma_list);

def pyramideDeGaussiennes(image_initiale,s,nb_octave):
    n = 5
    k = 2 ** (1 / s)
    sigma = 1.6

    image_octave = image[0:-1:nb_octave, 0:-1:nb_octave]
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

# image=mpimg.imread("theArtist.png")[:,:,1]
image=mpimg.imread("lena.jpg")[:,:,1]



# Super fonction codée à la main
# image_conv=convolution(image,M_g)

nb_octave=1
s=3

image_conv=pyramideDeGaussiennes(image,3,1)


# f,axarr = plt.subplots(2,3)
# axarr[0,0].imshow(image_conv[:,:,0])
# axarr[0,1].imshow(image_conv[:,:,1])
# axarr[0,2].imshow(image_conv[:,:,2])
# axarr[1,0].imshow(image_conv[:,:,3])
# axarr[1,1].imshow(image_conv[:,:,4])
# axarr[1,2].imshow(image_conv[:,:,5])


L,sigma_list = differenceDeGaussiennes(image,s,nb_octave)

f,axarr = plt.subplots(2,3)
axarr[0,0].imshow(L[:,:,0])
axarr[0,1].imshow(L[:,:,1])
axarr[0,2].imshow(L[:,:,2])
axarr[1,0].imshow(L[:,:,3])
axarr[1,1].imshow(L[:,:,4])

plt.show()



