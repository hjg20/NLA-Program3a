import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae


def solve_least_squares(H, R, b):
    c = np.dot(H.T, b)
    # x = np.zeros((R.shape[1], 1))
    # for i in reversed(range(R.shape[1])):
    #     x[i] = (c[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
    x = np.linalg.inv(R).dot(c)
    return x


def norm(x):
    norm = 0
    for i in x:
        norm += i**2
    norm = np.sqrt(norm)
    return norm


def householder_reflector(v):
    v = np.asarray(v).reshape(-1, 1)
    I = np.eye(v.shape[0])
    vTv = np.dot(v.T, v)
    H = I - 2 * np.dot(v, v.T) / vTv
    return H


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


def algorithm2():
    return None


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


df = pd.DataFrame()
df['Sizes'] = sizes
df['Errors'] = errors


plt.plot(df['Sizes'], df['Errors'])
plt.xlabel('Matrix Sizes')
plt.ylabel('LS Error')
plt.show()
