# -*- coding: utf-8 -*-

""" Eigenvalues and eigenvectors utilities module.
This module is in charge of providing utilities for calculation eigenvalues and eigenvectors.

"""

import numpy as np


def _is_symmetric(matrix):
    """ Determines whether a matrix is symmetric or not.
            Params:
                matrix (np.matrix): the matrix to check symmetry.
            Returns:
                Bool: Whether the matrix is or not symmetric.
            """
    return np.allclose(matrix, matrix.T)


def _gram_schmidt(matrix):
    """ Calculates the QR decomposition using the Gram Schmidt method.
            Params:
                matrix (np.matrix): the matrix to decompose.
            Returns:
                q (np.ndarray): The Q orthogonal matrix.
                r (np.ndarray): The R upper triangular matrix.
            Reference:
                http://web.mit.edu/18.06/www/Essays/gramschmidtmat.pdf
            """
    m, n = matrix.shape
    q = np.zeros((m, n))
    r = np.zeros((n, n))
    for j in range(n):
        v = matrix[:, j]
        for i in range(j):
            r[i, j] = np.matmul(q[:, i].T, matrix[:, j])
            v = v.squeeze() - np.dot(r[i, j], q[:, i])
        r[j, j] = np.linalg.norm(v)
        q[:, j] = np.divide(v, r[j, j]).squeeze()
    return q, r


def _householder(matrix):
    """ Calculates the QR decomposition using the Householder method.
            Params:
                matrix (np.matrix): the matrix to decompose.
            Returns:
                q (np.ndarray): The Q orthogonal matrix.
                r (np.ndarray): The R upper triangular matrix.
            Reference:
                http://www.cs.cornell.edu/~bindel/class/cs6210-f09/lec18.pdf
            """
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


def _symmetric_eig(matrix, method=_householder, iterations=50, tolerance=1e-4):
    """ Calculates eigenvectors and eigenvalues for a symmetric matrix.
            Params:
                matrix (np.matrix): The symmetric matrix.
                method (function): The qr method to be used.
                iterations (int): Number of iterations.
                tolerance (float): Tolerance.
            Returns:
                e (np.ndarray): A list of eigenvalues.
                v (np.ndarray): A matrix which each column is an eigenvector.
            Reference:
                http://www-users.math.umn.edu/~olver/aims_/qr.pdf (Page 18)
            """
    a = np.matrix(matrix, dtype=np.float64)

    q, r = method(a)
    a = np.matmul(r, q)
    s = q

    for i in range(iterations):
        q, r = method(a)
        a = np.matmul(r, q)
        s = np.matmul(s, q)
        if np.allclose(a, np.diagflat(np.diag(a)), atol=tolerance):
            break

    eigenvalues = np.diag(a)
    return eigenvalues, s


def s_eig(matrix):
    """ Calculates eigenvectors and eigenvalues for a symmetric matrix.
            Params:
                matrix (np.matrix): The symmetric matrix.
            Returns:
                e (np.ndarray): A list of eigenvalues.
                v (np.ndarray): A matrix which each column is an eigenvector.
            """
    if _is_symmetric(matrix):
        return _symmetric_eig(matrix)

    raise ValueError("The input matrix should be symmetric.")

