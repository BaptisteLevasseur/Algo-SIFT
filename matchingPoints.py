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
    pass


def get_final_pic_dimensions(h, image1, image2):
    xmax, ymax = image2.shape[0:2]
    #faut il inverser ici ?
    hinv = np.linalg.inv(h)
    final_image_size = np.dot(hinv,np.array([xmax, ymax, 1]))
    xfinal, yfinal = final_image_size[0:2]
    xfinal = np.max([xfinal, image1.shape[0]])
    yfinal = np.max([yfinal, image1.shape[1]])
    return int(xfinal), int(yfinal)

#attention : ici on bosser avec des images couleur
def reconstruct_image(h, image1, image2):
    x1 = image1.shape[0]
    y1 = image1.shape[1]
    x2 = image2.shape[1]
    y2 = image2.shape[1]

    xmax, ymax = get_final_pic_dimensions(h, image1, image2)
    res = np.zeros((xmax, ymax, 3))
    hinv = np.linalg.inv(h)
    res[:x1, :y1, 0] = image1[:, :]
    res[:x1, :y1, 1] = image1[:, :]
    res[:x1, :y1, 2] = image1[:, :]
    for i in range(0,xmax):
        for j in range(0, ymax):
            i_init, j_init = np.array(np.dot(h, [i, j, 1])[0:2], dtype='int')
            if i_init >= x2 or i_init < 0 or j_init >= y2 or j_init < 0:
                res[i, j, :] = np.array([0, 0, 0])
            else:
                res[i, j, 0] = image2[i_init, j_init]
                res[i, j, 1] = image2[i_init, j_init]
                res[i, j, 2] = image2[i_init, j_init]

    return res


def castToGrayScale(image):
    image_gray = np.zeros(image.shape[0:2])
    image_gray[:] = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    return image_gray


def final_pipeline(desc1, desc2, image1, image2):
    a = desc1
    b = desc2
    image_initiale1 = mpimg.imread(image1)[:, :, 1]

    image_initiale2 = mpimg.imread(image2)[:, :, 1]
    image_initiale1 = image_initiale1 / 255
    image_initiale2 = image_initiale2 / 255
    # image_initiale1 = castToGrayScale(image_initiale1)
    # image_initiale2 = castToGrayScale(image_initiale2)
    c = distanceInterPoints(a, b)
    d = get_n_nearest_points(c, 4)
    print(d)
    p1, p2 = get_nearest_descriptors_couples(d, a, b)
    display_circle_on_points(p1, p2, image_initiale1, image_initiale2)

    A = constructionA(p1, p2)
    Hsvd = get_H_by_SVD(A)
    HpasSvd = get_H_by_quad(A)

    print(np.dot(A, np.reshape(Hsvd, (9, 1))))

    print(np.dot(A, np.reshape(HpasSvd, (9, 1))))

    print(Hsvd)
    print(HpasSvd)
    print(Hsvd - HpasSvd)



    r = reconstruct_image(Hsvd, image_initiale1, image_initiale2)
    fig, ax = plt.subplots()
    ax.imshow(r[:, :, 0], cmap='gray')
    plt.show()
    pass

if __name__ == '__main__':
    a = np.array([[100, 120, 1, 2], [300, 180, 3, 4], [300, 420, 10, 10], [400, 100, 100, 100], [500, 220, 8, 9]])
    b = np.array([[100, 100, 1, 2], [300, 160, 3, 3.9], [300, 400, 9, 10], [100, 300, 200, 0], [500, 200, 7, 9]])


    image_initiale1 = mpimg.imread("gauche.jpg")[:, :, 1]

    image_initiale2 = mpimg.imread("droite.jpg")[:, :, 1]
    image_initiale1 = image_initiale1 / 255
    image_initiale2 = image_initiale2 / 255
    #image_initiale1 = castToGrayScale(image_initiale1)
    #image_initiale2 = castToGrayScale(image_initiale2)
    c = distanceInterPoints(a,b)
    d = get_n_nearest_points(c,4)
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

    r = reconstruct_image(Hsvd,image_initiale1,image_initiale2)
    fig, ax = plt.subplots()
    ax.imshow(r, cmap='gray')
    plt.show()
    pass





