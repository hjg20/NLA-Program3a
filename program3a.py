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


def generate_nonsingular_matrix(n, k):
    # Prompting the user for the size of the square matrix
    if n < k:
        print('n < k')
        return None

    while True:
        # Generating a random matrix of size n x n
        matrix = np.random.rand(n, k)

        # Checking if the determinant is non-zero (the matrix is nonsingular)
        if np.linalg.det(matrix) != 0:
            break  # If the matrix is nonsingular, break the loop

    return matrix

sizes = range(2, 200, 5)
errors = []
np.random.seed(1)

for i in sizes:
    n = i
    k = i

    A = generate_nonsingular_matrix(n, k)
        
    #A = np.random.rand(10, 8)##
    b = np.random.rand(n, 1)##

    x1 = algorithm1(A, b)

    true_x = np.linalg.lstsq(A, b, rcond=None)[0]

    errors.append(np.abs(np.sum(x1 - true_x)))

plt.plot(sizes, errors)
plt.show()