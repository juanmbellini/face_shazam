# -*- coding: utf-8 -*-

""" Eigenvalues and eigenvectors utilities module.
This module is in charge of providing utilities for calculation eigenvalues and eigenvectors.

"""

import numpy as np


def _is_symmetric(a, tol=1e-4):
    return np.allclose(a, a.T, atol=tol)


def _householder(matrix):
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


def _symmetric_eig(matrix, iterations=50, tolerance=1e-4):
    a = np.matrix(matrix, dtype=np.float64)

    q, r = _householder(a)
    a = np.matmul(r, q)
    s = q

    for i in range(iterations):
        q, r = _householder(a)
        a = np.matmul(r, q)
        s = np.matmul(s, q)
        if np.allclose(a, np.diagflat(np.diag(a)), atol=tolerance):
            break

    eigenvalues = np.diag(a)
    return s, eigenvalues


def s_eig(matrix):
    """ Calculates eigenvectors and eigenvalues for a symmetric matrix.
            Params:
                matrix (np.matrix): The symmetric matrix.
            Returns:
                v (np.ndarray): A matrix which each column is an eigenvector.
                e (np.ndarray): A list of eigenvalues.
            """
    if _is_symmetric(matrix):
        return _symmetric_eig(matrix)

    raise ValueError("The input matrix should be symmetric.")
