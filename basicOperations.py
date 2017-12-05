import numpy as np


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
    Dy, Dx = gradient(image)
    Dyy, Dyx = gradient(Dy)
    Dxy, Dxx = gradient(Dx)
    return [[Dxx,Dxy],[Dyx,Dyy]]


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
