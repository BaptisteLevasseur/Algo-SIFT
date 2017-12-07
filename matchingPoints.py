import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

from homographie import *
from timeDecorator import *




#attention : les deux premiers indices sont la position
@timeit
def distanceInterPoints(points_image1, points_image2):
    nb_points1 = points_image1.shape[0]
    nb_points2 = points_image2.shape[0]
    d = np.zeros((nb_points1,nb_points2))
    for i in range(nb_points1):
        for j in range(nb_points2):
            d[i, j] = np.sqrt(np.sum(np.power(points_image1[i, 2:]-points_image2[j, 2:], 2)))
    return d


def get_n_nearest_points(distance_matrix, n):
    x, y = distance_matrix.shape
    a = np.argsort(distance_matrix.ravel())
    b = np.unravel_index(a, (x, y))
    c = np.stack(b,axis=1)
    return c[:n, :]


def get_nearest_descriptors_couples(indices, points_image1, points_image2):
    return points_image1[indices[:, 0], :2], points_image2[indices[:, 1], :2]


#peut sembler faux si un descripteur d'une image est associé à plusieurs descripteurs d'une autre image : les cercles se superposent
def display_circle_on_points(points1, points2,  image1, image2):
    fig, ax = plt.subplots(2,1)
    ax[0].imshow(image1)
    ax[1].imshow(image2)

    c = ('b', 'r', 'g', 'y')
    for i in range(points1.shape[0]):
        circle1 = plt.Circle((points1[i, 1], points1[i, 0]), 15, color=c[i % 4])
        circle2 = plt.Circle((points2[i, 1], points2[i, 0]), 15, color=c[i % 4])

        ax[0].add_artist(circle1)
        ax[1].add_artist(circle2)

    plt.show()
    pass


def get_final_pic_dimensions(h, image1, image2):
    xmax, ymax = image2.shape[0:2]
    hinv = np.linalg.inv(h)
    final_image_size = np.dot(h,np.array([xmax, ymax, 1]))
    xfinal, yfinal = final_image_size[0:2]
    xfinal = np.max([xfinal, image1.shape[0]])
    yfinal = np.max([yfinal, image1.shape[1]])
    return int(xfinal), int(yfinal)

#attention : ici on bosser avec des images couleur
@timeit
def reconstruct_image(h, image1, image2):
    x1 = image1.shape[0]
    y1 = image1.shape[1]
    x2 = image2.shape[0]
    y2 = image2.shape[1]
    hinv = np.linalg.inv(h)
    xmax, ymax = get_final_pic_dimensions(hinv, image1, image2)
    res = np.zeros((xmax+1, ymax+1, 3))
    res[:x1, :y1, :] = image1[:, :]
    for i in range(0, xmax):
        for j in range(0, ymax):
            i_init, j_init = np.array(np.dot(h, [i, j, 1])[0:2], dtype='int')
            if i_init >= x2 or i_init < 0 or j_init >= y2 or j_init < 0:
                pass
            else:
                res[i, j, :] = image2[i_init, j_init, :]
    return res


def correction_histogramme(image1, image2):
    t1 = np.mean(image1, axis=0)
    t2 = np.mean(image2, axis=0)
    col_means1 = np.mean(t1, axis=0)
    col_means2 = np.mean(t2, axis=0)
    dif_col = col_means1 - col_means2
    image2[:, :, 0] = image2[:, :, 0] + dif_col[0]
    image2[:, :, 1] = image2[:, :, 1] + dif_col[1]
    image2[:, :, 2] = image2[:, :, 2] + dif_col[2]

    image2[image2 < 0] = 0
    image2[image2 > 1] = 1

    return image2



def check_for_superposed_descriptors(desc1, desc2):

    for i in range(desc1.shape[0]-1):
        if i >= desc1.shape[0]:
            break
        d1 = desc1[i]
        for j in range(i+1, desc1.shape[0]):
            if j >= desc1.shape[0]:
                break
            d2 = desc1[j]
            distance = np.sqrt((d1[0] - d2[0])**2 + (d1[1] - d2[1])**2)
            if distance < 2:
                desc1, desc2 = check_for_superposed_descriptors(np.delete(desc1, j, 0), np.delete(desc2, j, 0))

    return desc1, desc2


def final_pipeline(desc1, desc2, image1, image2):
    a = desc1
    b = desc2

    image_initiale1 = mpimg.imread(image1)[:, :, :]
    image_initiale2 = mpimg.imread(image2)[:, :, :]
    image_initiale1 = image_initiale1 / 255
    image_initiale2 = image_initiale2 / 255

    print("Correction de l'exposition...")
    image_initiale2 = correction_histogramme(image_initiale1, image_initiale2)

    print("Calcul les descripteurs les plus proches...")
    c = distanceInterPoints(a, b)
    nb_plus_proche_voisins = 50
    d = get_n_nearest_points(c, nb_plus_proche_voisins)

    #on check si certains descripteurs sont superposés; auquel cas on prend les suivants
    #pour avoir un système inversible

    p1_unpurged, p2_unpurged = get_nearest_descriptors_couples(d, a, b)
    print("Suppression des descripteurs superposés...")
    p1, p2 = check_for_superposed_descriptors(p1_unpurged, p2_unpurged)
    display_circle_on_points(p1[:50, :], p2[:50, :], image_initiale1, image_initiale2)

    A = constructionA(p1[:50, :], p2[:50, :])
    Hsvd = get_H_by_SVD(A)
    HpasSvd = get_H_by_quad(A)

    print("Vérifications : norme euclidienne de A*h par les deux méthodes")
    print(np.linalg.norm(np.dot(A, np.reshape(Hsvd, (9, 1)))))
    print(np.linalg.norm(np.dot(A, np.reshape(HpasSvd, (9, 1)))))

    print("Norme de la différence des deux H")
    print(np.linalg.norm(Hsvd - HpasSvd))


    print("Construction de l'image panoramique...")
    r = reconstruct_image(Hsvd, image_initiale1, image_initiale2)
    fig, ax = plt.subplots()
    ax.imshow(r)
    plt.show()
    pass





if __name__ == '__main__':
    image1 = 'Redgauche.jpg'
    image2 = 'Reddroite.jpg'
    image_initiale1 = mpimg.imread(image1)[:, :, :]
    image_initiale2 = mpimg.imread(image2)[:, :, :]
    image_initiale1 = image_initiale1 / 255
    image_initiale2 = image_initiale2 / 255

    im2cor = correction_histogramme(image_initiale1,image_initiale2)
    fig, ax = plt.subplots(3, 1)
    ax[0].imshow(image1)
    ax[1].imshow(image2)
    ax[2].imshow(im2cor)

    plt.show()
    pass



