import numpy as np





def constructionA(points1, points2):
    n = points1.shape[0]
    A = np.zeros((2*n,9))

    for i in range(n):
        x1 = points1[i, 0]
        y1 = points1[i, 1]
        x2 = points2[i, 0]
        y2 = points2[i, 1]
        A[2 * i, 0] = x1
        A[2 * i, 1] = y1
        A[2 * i, 2] = 1
        A[2 * i, 3:6] = 0
        A[2 * i, 6] = -x2 * x1
        A[2 * i, 7] = -x2 * y1
        A[2 * i, 8] = -x2

        A[2 * i + 1, :3] = 0
        A[2 * i + 1, 3] = x1
        A[2 * i + 1, 4] = y1
        A[2 * i + 1, 5] = 1
        A[2 * i + 1, 6] = -y2 * x1
        A[2 * i + 1, 7] = -y2 * y1
        A[2 * i + 1, 8] = -y2
    return A

def get_H_by_SVD(A):
    u, s, v = np.linalg.svd(A)
    h = np.reshape(v[-1, :], (3, 3))
    h = h / h[2,2]
    return h

def get_H_by_quad(A):
    AT = np.transpose(A)
    ATA = np.dot(AT,A)
    w, v = np.linalg.eig(ATA)
    arg_vp_min = np.argsort(w)[0]
    v_min = v[:,arg_vp_min]
    h = np.reshape(v_min, (3, 3))
    h = h / h[2,2]
    return h




