import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from homographie import *




#attention : les deu premiers indices sont la position
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
    ax[0].imshow(image1, cmap='gray')
    ax[1].imshow(image2, cmap='gray')

    c = ('b', 'r', 'g', 'y')
    for i in range(points1.shape[0]):
        circle1 = plt.Circle((points1[i, 0], points1[i, 1]), 10, color=c[i % 4], fill=False)
        circle2 = plt.Circle((points2[i, 0], points2[i, 1]), 10, color=c[i % 4], fill=False)

        ax[0].add_artist(circle1)
        ax[1].add_artist(circle2)

    plt.show()














if __name__ == '__main__':
    a = np.array([[100, 120, 1, 2], [500, 500, 3, 4], [300, 420, 10, 10], [400, 100, 100, 100], [500, 220, 8, 9]])
    b = np.array([[100, 100, 1, 2], [300, 400, 4, 3], [300, 400, 9, 10], [100, 300, 200, 0], [500, 200, 7, 9]])


    image_initiale1 = mpimg.imread("lena.jpg")[:, :, 1]

    image_initiale2 = mpimg.imread("lena.jpg")[:, :, 1]
    image_initiale1 = image_initiale1 / 255
    image_initiale2 = image_initiale2 / 255

    c = distanceInterPoints(a,b)
    d = get_n_nearest_points(c,3)
    print(d)
    p1,p2 = get_nearest_descriptors_couples(d,a,b)
    display_circle_on_points(p1,p2,image_initiale1,image_initiale2)

    A = constructionA(p1,p2)
    Hsvd = get_H_by_SVD(A)
    HpasSvd = get_H_by_quad(A)

    print(np.dot(A,np.reshape(Hsvd, (9, 1))))

    print(np.dot(A, np.reshape(HpasSvd, (9, 1))))

    print(Hsvd)
    print(HpasSvd)
    print(Hsvd-HpasSvd)

    x1 = np.array([100, 120, 1])
    x11 = np.array([100, 100, 1])
    x2 = np.array([300, 420, 0])
    x22 = np.array([300, 400, 0])
    x3 = np.array([500, 200, 0])
    x33 = np.array([500, 220, 0])
    print('===========')
    print(np.dot(Hsvd,x1))
    print(np.dot(Hsvd,x2))
    print(np.dot(Hsvd,x3))
    print(np.dot(Hsvd,x11))
    print(np.dot(Hsvd,x22))
    print(np.dot(Hsvd,x33))
    print('===========')
    print(np.dot(HpasSvd, x1))
    print(np.dot(HpasSvd, x2))
    print(np.dot(HpasSvd, x3))
    print(np.dot(HpasSvd, x11))
    print(np.dot(HpasSvd, x22))
    print(np.dot(HpasSvd, x33))





