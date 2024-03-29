import numpy as np
import matplotlib.pyplot as plt


def householder_reflection(v):
    v = np.asarray(v).reshape(-1, 1)
    I = np.eye(v.shape[0])
    vTv = np.dot(v.T, v)
    H = I - 2 * np.dot(v, v.T) / vTv
    return H


def algorithm1(A, b):
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy()
    for i in range(min(m, n)):
        x = R[i:, i]
        norm_x = np.linalg.norm(x)
        e1 = np.zeros_like(x)
        e1[0] = 1
        v = x + np.copysign(norm_x, x[0]) * e1
        H_sub = householder_reflection(v)
        H_i = np.eye(m)
        H_i[i:, i:] = H_sub
        if np.dot(H_sub[0, :], x) < 0:
            H_sub = -H_sub
            H_i[i:, i:] = H_sub
        R = np.dot(H_i, R)
        Q = np.dot(Q, H_i.T)
    c = np.dot(Q.T, b)
    x = np.zeros((R.shape[1], 1))
    for i in reversed(range(R.shape[1])):
        x[i] = (c[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
    return x


def test1matrix(n, k):
    if n < k:
        print('n < k')
        return None
    while True:
        np.random.seed(1)
        matrix = np.random.rand(n, k)
        if np.linalg.det(matrix) != 0:
            matrix -= np.average(matrix)
            matrix /= np.sum(matrix)
            break
    return matrix


def test2matrix(n, k):
    if n <= k:
        raise ValueError("n must be greater than k for a full column rank matrix.")
    Q, _ = np.linalg.qr(np.random.randn(n, k))
    b = Q @ np.random.randn(k)
    return Q, b


def test3matrix(n, k):
    if n <= k:
        raise ValueError("n must be greater than k for a full column rank matrix.")
    A = np.random.randn(n, k)
    Q, R = np.linalg.qr(A)
    while not np.all(np.diag(R)[:k]):
        A = np.random.randn(n, k)
        Q, R = np.linalg.qr(A)
    b1 = A @ np.random.randn(k)
    random_vector = np.random.randn(n)
    projection = Q @ Q.T @ random_vector
    b2 = random_vector - projection
    while np.linalg.norm(b2) < 1e-10:
        random_vector = np.random.randn(n)
        projection = Q @ Q.T @ random_vector
        b2 = random_vector - projection
    b = b1 + b2
    return A, b
