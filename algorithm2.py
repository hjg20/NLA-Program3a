import numpy as np
import matplotlib.pyplot as plt


def householder_reflection(v):
    u = v / (v[0] + np.copysign(np.linalg.norm(v), v[0]))
    u[0] = 1
    H = np.eye(len(v)) - (2 / np.dot(u, u)) * np.outer(u, u)
    return H


def apply_householder(H, A):
    return np.dot(H, A)


def qr_decomposition(A):
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    
    for i in range(min(m, n)):
        H = np.eye(m)
        H[i:, i:] = householder_reflection(R[i:, i])
        R = apply_householder(H, R)
        Q = apply_householder(H, Q)
    
    return Q.T, R


def solve_triangular(R, b):
    n = R.shape[1] 
    x = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= R[i, j] * x[j]
        x[i] /= R[i, i]
    
    return x


def incremental_least_squares(An, bn, an1, beta_n1):
    An1 = np.vstack((An, an1))
    bn1 = np.append(bn, beta_n1)
    Qn1, Rn1 = qr_decomposition(An1)
    Qt_bn1 = np.dot(Qn1.T, bn1)[:Rn1.shape[1]] 
    x_min_n1 = solve_triangular(Rn1, Qt_bn1)
    return x_min_n1, An1, bn1


def relative_error(manual, numpy_sol):
    return np.linalg.norm(manual - numpy_sol) / np.linalg.norm(numpy_sol)


sizes = range(2, 201, 5)
relative_errors = []

for n in sizes:
    A = np.random.rand(n, n)
    b = np.random.rand(n)
    
    an1 = np.random.rand(1, n)
    beta_n1 = np.random.rand()

    x_min_manual, _, _ = incremental_least_squares(A, b, an1, beta_n1)
    x_min_np, _, _, _ = np.linalg.lstsq(np.vstack((A, an1)), np.append(b, beta_n1), rcond=None)

    error = relative_error(x_min_manual, x_min_np)
    relative_errors.append(error)

plt.plot(sizes, relative_errors)
plt.title('Relative Error vs. Matrix Sizes')
plt.xlabel('Matrix Size n')
plt.ylabel('Relative Error')
plt.grid(True)
plt.show()
