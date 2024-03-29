import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae


<<<<<<< HEAD
=======
def solve_least_squares(H, R, b):
    c = np.dot(H.T, b)
    # x = np.zeros((R.shape[1], 1))
    # for i in reversed(range(R.shape[1])):
    #     x[i] = (c[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
    x = np.linalg.inv(R).dot(c)
    return x


>>>>>>> f170094441ad0e8abd8e6d57ff58c4104dffafc2
def norm(x):
    norm = 0
    for i in x:
        norm += i**2
    norm = np.sqrt(norm)
    return norm


def householder_reflection(v):
    u = v / (v[0] + np.copysign(np.linalg.norm(v), v[0]))
    u[0] = 1
    H = np.eye(len(v)) - (2 / np.dot(u, u)) * np.outer(u, u)
    return H


<<<<<<< HEAD
=======
def algorithm1(R):
    m, n = R.shape
    H = np.identity(m)
    
    for i in range(min(m, n)):
        x = R[i:, i]
        norm_x = norm(x)
        e1 = np.zeros_like(x)
        e1[0] = 1
        v = x + np.copysign(norm_x, x[0]) * e1
        H_sub = householder_reflector(v)
        H_i = np.eye(m)
        H_i[i:, i:] = H_sub
        if np.dot(H_sub[0, :], x) < 0:
            H_sub = -H_sub
            H_i[i:, i:] = H_sub
        R = np.dot(H_i, R)
        H = np.dot(H, H_i.T)
    
    return H[:, :n], R[:n, :]


>>>>>>> f170094441ad0e8abd8e6d57ff58c4104dffafc2
def algorithm2():
    return None


<<<<<<< HEAD
def apply_householder(H, A):
    return np.dot(H, A)
=======
def matrix(rows, cols):
    if rows < cols:
        raise ValueError("n<k")
    random_matrix = np.random.rand(rows, cols)
    H, _ = algorithm1(random_matrix)
    return H[:, :cols]


sizes = range(2,51)

errors = []

for i in sizes:

    n, k = i+1, i


    A = matrix(n, k)
    b = np.random.rand(n,1)

    H, R = algorithm1(A)

    x = solve_least_squares(H, R, b)
    true_x = np.linalg.lstsq(A, b, rcond=None)[0]

    average_true = np.average(true_x)
    average_pred = np.average(x)
    errors.append(np.abs(average_true-average_pred))
>>>>>>> f170094441ad0e8abd8e6d57ff58c4104dffafc2


# Function to perform QR decomposition using Householder reflections
def qr_decomposition(A):
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    
    for i in range(min(m, n)):
        H = np.eye(m)
        H[i:, i:] = householder_reflection(R[i:, i])
        R = apply_householder(H, R)
        Q = apply_householder(H, Q)
    
    return Q.T, R  # Q.T is the actual Q in QR decomposition


<<<<<<< HEAD
# Function to solve Rx = b for an upper triangular matrix R
def solve_triangular(R, b):
    n = R.shape[1]  # Considering the number of columns as the matrix may not be square
    x = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= R[i, j] * x[j]
        x[i] /= R[i, i]
    
    return x


def incremental_least_squares(An, bn, an1, beta_n1):
    """
    Incrementally updates the least squares solution given new data.
    
    :param An: The matrix A at step n (A_n)
    :param bn: The vector b at step n (b_n)
    :param an1: The new row to be added to A (a_{n+1})
    :param beta_n1: The new element to be added to b (beta_{n+1})
    :return: The updated solution x_min at step n+1
    """
    # Step 1: Update An and bn to An+1 and bn+1
    An1 = np.vstack((An, an1))
    bn1 = np.append(bn, beta_n1)
    
    # Step 2: Compute QR decomposition of An1
    Qn1, Rn1 = qr_decomposition(An1)
    
    # Step 3: Solve the least squares problem to find x_min at step n+1
    Qt_bn1 = np.dot(Qn1.T, bn1)[:Rn1.shape[1]]
    x_min_n1 = solve_triangular(Rn1, Qt_bn1)
    
    return x_min_n1








# Example usage with dummy data:
A = np.random.rand(5, 3)
b = np.random.rand(5)
an1 = np.random.rand(1, 3)
beta_n1 = np.random.rand(1)

Q,R = qr_decomposition(A)
Qtb = np.dot(Q.T, b)[:R.shape[1]]
x1 = solve_triangular(R, Qtb)
print(x1)
print(np.linalg.lstsq(A,b))

# Get the solution for x_min at step n+1
x_min_n1 = incremental_least_squares(A, b, an1, beta_n1)
print(x_min_n1)

=======
plt.plot(df['Sizes'], df['Errors'])
plt.xlabel('Matrix Sizes')
plt.ylabel('LS Error')
plt.show()
>>>>>>> f170094441ad0e8abd8e6d57ff58c4104dffafc2
