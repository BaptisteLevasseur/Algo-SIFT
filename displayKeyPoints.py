import numpy as np
import math
import sys

import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.collections as cl
import scipy.signal as sig
import time
from keypointDetection import *
from scaleSpace import *
from timeDecorator import timeit



def displayDescriptor(descriptorList, image):
    fig, ax = plt.subplots()

    ax.imshow(image,cmap='gray')

    for i in range(descriptorList.shape[0]):
        x = descriptorList[i,0]
        y = descriptorList[i,1]
        mag = descriptorList[i,2]
        theta = descriptorList[i,3]

        circle = plt.Circle((x, y), mag, color='y', fill=False)
        x2 = np.cos(theta)*mag + x
        y2 = np.sin(theta)*mag + y
        line = plt.Line2D([x, x2], [y, y2], color='b')

        ax.add_artist(circle)
        ax.add_line(line)

    plt.show()

descriptorList = np.array([[10, 10, 200, 0], [100, 200, 120, 1], [400, 400, 40, 2], [200, 20, 100, 0]])
image_initiale = mpimg.imread("lena.jpg")[:, :, 1]
image_initiale = image_initiale / 255
displayDescriptor(descriptorList, image_initiale)





