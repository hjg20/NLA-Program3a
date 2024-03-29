import algorithm1 as a1
import numpy as np
import matplotlib.pyplot as plt

test = input('Which test are you performing? (1, 2, or 3)')

if test == "1":
    # Test Problem 1: n = k, square nonsingular A
    sizes = range(10, 200, 5)
    errors = []
    for i in sizes:
        n, k = i, i
        
        A = a1.test1matrix(n, k)
        b = np.random.rand(n, 1)

        x1 = a1.algorithm1(A, b)

        true_x = np.linalg.lstsq(A, b, rcond=None)[0]

        errors.append(np.abs(np.sum(x1 - true_x)/np.sum(true_x)))

    plt.plot(sizes, errors)
    plt.xlabel('Matrix Sizes, n')
    plt.ylabel('Relative Error')
    plt.title('Relative Error vs. Matrix Sizes')
    plt.show()

elif test == "2":
    # Test Problem 2: n>k, rectangular full column rank, b in range(A)
    sizes = range(10, 200, 5)
    errors = []
    for i in sizes:
        n, k = i+5, i
        
        A, b = a1.test2matrix(n, k)

        x1 = a1.algorithm1(A, b)

        true_x = np.linalg.lstsq(A, b, rcond=None)[0]

        errors.append(np.abs(np.sum(x1 - true_x)/np.sum(true_x)))

    plt.plot(sizes, errors)
    plt.xlabel('Matrix Sizes, n')
    plt.ylabel('Relative Error')
    plt.title('Relative Error vs. Matrix Sizes')
    plt.show()

elif test == "3":
    # Test Problem 3: n>k, rectangular full column rank, b = b_1 + b_2 where b_1 in range(A) and b_2 not in range(A)
    sizes = range(10, 200, 5)
    errors = []
    for i in sizes:
        n, k = i+5, i
        
        A, b = a1.test3matrix(n, k)

        x1 = a1.algorithm1(A, b)

        true_x = np.linalg.lstsq(A, b, rcond=None)[0]

        errors.append(np.abs(np.sum(x1 - true_x)/np.sum(true_x)))

    plt.plot(sizes, errors)
    plt.xlabel('Matrix Sizes, n')
    plt.ylabel('Relative Error')
    plt.title('Relative Error vs. Matrix Sizes')
    plt.show()










