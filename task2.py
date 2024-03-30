import numpy as np
import matplotlib.pyplot as plt


def x_true(t):
    return np.sin(t) + t * np.cos(t)**2


def create_L_matrix(n):
    L = np.eye(n) - np.eye(n, k=1)
    L = L[:-1] 
    return L


def regularized_least_squares(b, L, lam, A):
    ALA = A.T @ A + lam * L.T @ L
    ALb = A.T @ b
    x_rls = np.linalg.solve(ALA, ALb)
    return x_rls


lambdas = [1, 10, 100, 1000]
n_values = [100, 200, 300, 400, 500]
plt.figure(figsize=(15, 10))
for lam in lambdas:
    for n in n_values:
        t = np.linspace(0, 4, n)
        xtrue = x_true(t)
        w = np.random.randn(n)
        b = xtrue + w
        A = np.eye(n)
        L = create_L_matrix(n)
        x_rls = regularized_least_squares(b, L, lam, A)
        error = np.linalg.norm(xtrue - x_rls) / np.linalg.norm(xtrue)
        plt.plot(t, xtrue, label=f'True Signal n={n}')
        plt.plot(t, x_rls, label=f'Recovered Signal n={n}, λ={lam}')
        plt.title('True vs Recovered Signals')
        plt.xlabel('Time t')
        plt.ylabel('Signal')
        plt.legend()
        print(f'Error for λ={lam} and n={n}: {error}')
plt.show()
