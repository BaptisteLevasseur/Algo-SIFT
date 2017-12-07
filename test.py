import numpy as np
import math
import sys
import cv2
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as sig
import time
from keypointDetection import *
from scaleSpace import *
from timeDecorator import timeit
from keypointDescriptor import *
from keypointDetection import *
from scaleSpace import *





def question1_1(image,s):
    L,sigma_list = pyramideDeGaussiennes(image, s, 0)
    # Plot la pyramide de gaussienne

    plt.figure(1)
    for i in range(0, s + 2):
     plt.subplot(231 + i)
     plt.imshow(L[:, :, i], cmap='gray')
     plt.title(sigma_list[i])
    plt.draw()


def question1_2(image,nb_octave,s):
 # Plot la pyramide de gaussienne
 DoG_list, sigma_list = differenceDeGaussiennes(image, s, nb_octave)
 DoG=DoG_list[0]

 plt.figure(2)
 for i in range(0,s+2):
  plt.subplot(231+i)
  plt.imshow(DoG[:,:,i],cmap='gray')
  plt.title(sigma_list[i])
 plt.draw()


def question2_2(image,nb_octave,s):
 r_courb_principale = 10
 seuil_contraste = 0.03
 DoG_list, sigma_list = differenceDeGaussiennes(image, s, nb_octave)
 compteur=np.zeros((nb_octave,4))
 for octave in range(0,nb_octave):
  compteur[octave,:]=compteurExtrema(DoG_list,octave,r_courb_principale,seuil_contraste)
 print("Compteurs des points cles")
 print(compteur)

 n_final=compteur[:,0]-np.sum(compteur[:,1::],1)

 plt.figure(3)
 plt.plot(range(0,nb_octave),n_final,'+-')
 plt.title("Evolution du nombre de points clés en fonction de la résolution de l'octave")
 plt.ylabel("Nombre de points clés conservés")
 plt.xlabel("Résolution de l'octave")
 plt.draw()

 def question2_3(image, nb_octave, s):
  r_courb_principale = 10
  seuil_contraste = 0.03
  DoG_list, sigma_list = differenceDeGaussiennes(image, s, nb_octave)


  # # compteur = np.zeros((nb_octave, 4))
  # print("Compteurs des points cles")
  # print(compteur)
  #
  # n_final = compteur[:, 0] - np.sum(compteur[:, 1::], 1)
  #
  # plt.figure(3)
  # plt.plot(range(0, nb_octave), n_final, '+-')
  # plt.title("Evolution du nombre de points clés en fonction de la résolution de l'octave")
  # plt.ylabel("Nombre de points clés conservés")
  # plt.xlabel("Résolution de l'octave")
  # plt.draw()

def main():

 # Paramètres de l'image
 image_name = "lena.jpg"
 image_initiale = mpimg.imread(image_name)[:, :, 1]
 image = image_initiale / 255

 nb_octave = 4
 s = 3




 question1_1(image,s)
 question1_2(image,nb_octave,s)
 question2_2(image,nb_octave,s)
 plt.show()



if __name__=="__main__":
 main()