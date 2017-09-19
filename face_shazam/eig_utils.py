# -*- coding: utf-8 -*-
""" Eigenvalues and eigenvectors utilities module.
This module is in charge of providing utilities for calculation eigenvalues and eigenvectors.

"""

import numpy as np


def _qr(matrix):
    m, n = matrix.shape
    q = np.identity(m)
    r = matrix.copy()
    for i in range(0, m - 1):
        v = r[i:m, i]
        s = -np.sign(v[0]).item()
        norm = np.linalg.norm(v)
        u = (r[i, i] - (norm * s)).item()
        v = np.divide(v, u)
        v[0] = 1
        tm = np.matmul(v, v.T) * (-s * u) / norm
        r[i:, :] = np.subtract(r[i:m, :], np.matmul(tm, r[i:m, :]))
        q[:, i:] = q[:, i:] - np.matmul(q[:, i:], tm)
    return q, r


def _eig(matrix):
    a = matrix.copy()
    n = a.shape[0]

    for i in range(100):
        q, r = _qr(a)
        a = np.matmul(r, q)

    eigenvalues = []

    i = 0
    while i < n:
        m = a[i:i+2, i:i+2]
        w = m[0, 0]
        if m.shape == (1, 1):
            eigenvalues.append(w)
            i = i + 1
        else:
            x, y, z = m[0, 1], m[1, 0], m[1, 1]
            eigv = np.roots([1, -(w+z), (w*z) - (y*x)])
            if isinstance(eigv[0], complex):
                eigenvalues.append(eigv[0])
                eigenvalues.append(eigv[1])
                i = i + 2
            else:
                eigenvalues.append(eigv[0])
                i = i + 1

    return eigenvalues


def test():
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
    #a = np.matrix([[-1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    # a = np.matrix(np.random.random_sample((240, 240)) * 255)
    # a = np.matrix([[3., -9.], [4., -3.]])
    # e = _eig(a)
    # pr(e)
    # e, v = np.linalg.eig(a)
    # pr(e)


def pr(stuff, message=None):
    if message:
        print(message)
    print(stuff)
    print("")
