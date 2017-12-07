import numpy as np
import math
import sys
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
from premierstests import *
from matchingPoints import *


#fichier utilisé pour obtenir les figures ne faisant pas partie du pipeline de base.
#les fonctions portent le nom



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

def question2_3and4(image, nb_octave, s):
 r_courb_principale = 10
 seuil_contraste = 0.03
 DoG_list, sigma_list = differenceDeGaussiennes(image, s, nb_octave)

 extrema_final_list = []
 for octave in range(nb_octave):
  print("Detection des points-clés dans l'octave " + str(octave))
  t = time.time()
  DoG = DoG_list[octave]
  extrema = detectionPointsCles(DoG, sigma_list, seuil_contraste, r_courb_principale, octave)
  extrema_final_list.append(extrema)
  t2 = time.time() - t
  print("Calcul effectué en {0:.2f} secondes".format(t2))


 print("Computing Descriptors...")

 n_keypoint = np.zeros(nb_octave)
 points_cles_orientes_list= np.empty((0,4))

 for octave in range(nb_octave):
  L_list, sigma_list = pyramideDeGaussiennes(image, s, octave)
  points_cles_orientes = orientationPointsCles(extrema_final_list[octave], L_list, sigma_list)
  n_keypoint[octave] = np.size(points_cles_orientes, 0)
  points_cles_orientes_list=np.vstack((points_cles_orientes_list,points_cles_orientes))


 plt.figure(4)
 plt.plot(range(0, nb_octave), n_keypoint, '+-')
 plt.title("Evolution du nombre de points clés en fonction de la résolution de l'octave")
 plt.ylabel("Nombre de points clés conservés")
 plt.xlabel("Résolution de l'octave")
 plt.draw()
 np.save("PointsCles",points_cles_orientes_list)

def question2_5(image, nb_octave, s):
 scale=10
 r_courb_principale = 10
 seuil_contraste = 0.03
 DoG_list, sigma_list = differenceDeGaussiennes(image, s, nb_octave)

 extrema_final_list = []
 for octave in range(nb_octave):
  print("Detection des points-clés dans l'octave " + str(octave))
  t = time.time()
  DoG = DoG_list[octave]
  extrema = detectionPointsCles(DoG, sigma_list, seuil_contraste, r_courb_principale, octave)
  extrema_final_list.append(extrema)
  t2 = time.time() - t
  print("Calcul effectué en {0:.2f} secondes".format(t2))

 print("Computing Descriptors...")

 n_keypoint = np.zeros(nb_octave)
 points_cles_orientes_list = []

 for octave in range(nb_octave):
  L_list, sigma_list = pyramideDeGaussiennes(image, s, octave)
  points_cles_orientes = orientationPointsCles(extrema_final_list[octave], L_list, sigma_list)
  n_keypoint[octave] = np.size(points_cles_orientes, 0)
  points_cles_orientes_list.append(points_cles_orientes)

 fig, ax = plt.subplots()
 ax.imshow(image, cmap='gray')
 c = ('b', 'r', 'g', 'y')
 for octave in range(nb_octave):
  points_cles_orientes=points_cles_orientes_list[octave]
  for i in range(points_cles_orientes.shape[0]):
   y ,x,mag,theta= points_cles_orientes[i, :]
   y,x = np.array([y, x])*2**octave
   circle = plt.Circle((x, y), scale*(octave+1), color=c[octave % 4], fill=False)

   x2 = np.cos(theta) * scale*(octave+1) + x
   y2 = np.sin(theta) * scale*(octave+1) + y
   line = plt.Line2D([x, x2], [y, y2], color=c[octave % 4])
   ax.add_artist(circle)
   ax.add_line(line)
 plt.draw()

def question2_6(image_name,numero):
 descripteurs=getDescriptors(image_name)
 np.save("Descripteurs"+str(numero),descripteurs)

def question3_1(points_image1, points_image2):
 plt.figure(6)
 D=distanceInterPoints(points_image1, points_image2)
 plt.imshow(D)
 plt.colorbar()


def main():

 # Paramètres de l'image
 image_name = "gauche.jpg"
 image_initiale = mpimg.imread(image_name)[:, :, 1]
 image = image_initiale / 255

 nb_octave = 4
 s = 3

 #question1_1(image,s) # Pyramide de gaussiennes
 #question1_2(image,nb_octave,s) # Différence de gaussienne
 #question2_2(image,nb_octave,s) # Compteurs du nombre de points clés (avant orientation)
 #question2_3and4(image, nb_octave, s) # Evolution nombre points cles Sauvegarde points clés
 question2_5(image,nb_octave,s) # tracé des points clés

 #question2_6(image_name,2) # sauvegarde des descripteurs (1 pour gauche et 2 pour droite)

 pointsCles1=np.load("Descripteurs1.npy")
 pointsCles2=np.load("Descripteurs2.npy")
 #question3_1(pointsCles1,pointsCles2)

 plt.show()
 pass


if __name__=="__main__":
 main()