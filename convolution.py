# coding: utf8

import numpy as np

def convolution(image,M):

    n_M=int(np.size(M,1))
    n_pad=int((n_M-1)/2)
    n,m=np.int_(np.shape(image))

    # Padding de l'image

    image_pad=np.concatenate((np.zeros((n_pad, m)), image, np.zeros((n_pad, m))),0)
    image_pad=np.concatenate((np.zeros((n+n_M-1,n_pad)),image_pad,np.zeros((n+n_M-1,n_pad))),1)

    n_ipad, m_ipad=np.int_(np.shape(image_pad))

    # Convolution de l'image
    image_conv=np.zeros((n_ipad,m_ipad))
    for i in range (n_pad,n_ipad-n_pad):
        for j in range(n_pad, m_ipad - n_pad):
            image_conv[i][j]=np.sum(image_pad[i - n_pad:i + n_pad + 1, j - n_pad:j + n_pad + 1] * M)

    # Depadding de l'image

    image_conv=image_conv[n_pad:n_ipad-n_pad,n_pad:m_ipad-n_pad]
    return image_conv;