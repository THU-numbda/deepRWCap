import numpy as np
import matplotlib.pyplot as plt

def generate_2d_grf(size, scale):
    # Create a 2D grid of coordinates
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)

    # Flatten the coordinate matrices and stack them
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Compute the squared distance matrix for each pair of points
    sq_dist = np.sum((positions[:, np.newaxis, :] - positions[:, :, np.newaxis])**2, axis=0)

    # Compute the covariance matrix
    covariance = np.exp(-sq_dist / (2 * scale**2))

    # Generate a Gaussian random field
    mean = np.zeros(size * size)
    grf = np.random.multivariate_normal(mean, covariance)
    grf = grf.reshape((size, size))

    return grf

# Generate and plot the 2D GRF
size = 10  # Reduced size for computational efficiency
scale = 0.2
grf_2d = generate_2d_grf(size, scale)

plt.imshow(grf_2d, cmap='viridis')
plt.colorbar()
plt.title('2D Gaussian Random Field')
plt.show()
