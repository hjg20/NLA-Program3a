# Implementing the Schur Algorithm step by step in Python

# We are starting with the functions that do not involve direct matrix operations such as downshift and hyperbolic rotation

# Helper function to create a downshift matrix
def downshift_matrix(size):
    """Creates a downshift matrix of a given size."""
    # Initialize a square matrix of zeros
    Z = [[0]*size for _ in range(size)]
    for i in range(1, size):
        Z[i][i-1] = 1
    return Z

# Function to perform hyperbolic rotation
def hyperbolic_rotation(a, b):
    """
    Computes the hyperbolic rotation matrix U for vector [a, b].
    """
    # Ensure the square root argument is always positive
    epsilon = 1e-10  # small value to prevent division by zero
    r = (max(a**2 - b**2, epsilon))**0.5
    c = a / r
    s = -b / r
    # The 2x2 rotation matrix
    U = [[c, s],
         [s, -c]]
    return U

# We can now test these functions to ensure they are working correctly.
# Let's test with a simple example where size=4 and a=3, b=4.
size_example = 4
a_example, b_example = 3, 4

# Create a downshift matrix of size 4
Z = downshift_matrix(size_example)

# Compute a hyperbolic rotation matrix for a=3, b=4
U = hyperbolic_rotation(a_example, b_example)

# Show the results
print(Z, U)
