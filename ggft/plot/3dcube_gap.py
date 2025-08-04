import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def value_cube(n, data):
    # Generate colors for each block
    colors = np.empty((n, n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                colors[i, j, k] = data[i + j*n + k*n*n]
    return colors

def draw_separated_lattice_cube(n, gap: int, data, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Calculate the size of the grid including gaps
    grid_size = n + int(gap * (n - 1))

    # Initialize an empty grid
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)

    # Populate the grid, leaving gaps
    for i in range(n):
        for j in range(n):
            for k in range(n):
                grid_i = i + int(i * gap)
                grid_j = j + int(j * gap)
                grid_k = k + int(k * gap)
                grid[grid_i, grid_j, grid_k] = True

    values = value_cube(n, data)

    norm = mcolors.Normalize(vmin=np.min(values), vmax=np.max(values))
    scalar_mappable = cm.ScalarMappable(norm=norm, cmap='viridis')

    # Get RGBA colors for the values
    rgba_colors = np.empty(grid.shape, dtype=object)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                grid_i = i + int(i * gap)
                grid_j = j + int(j * gap)
                grid_k = k + int(k * gap)
                rgba_colors[grid_i, grid_j, grid_k] = scalar_mappable.to_rgba(values[i, j, k])
    # rgba_colors = scalar_mappable.to_rgba(colors)
    # print(rgba_colors)

    # Now you can use rgba_colors for coloring plots, etc.

    cb = plt.colorbar(mappable=scalar_mappable, ax=ax)
    cb.set_label('Value')

    # Plotting the voxels
    ax.voxels(grid, facecolors=rgba_colors, edgecolor='grey', shade=False)

    # Set the aspect ratio to be equal
    ax.set_box_aspect([1,1,1])

    # Set the viewing angle
    ax.view_init(elev=20, azim=30)

    # plt.show()
    plt.savefig(filename)


import sys
sys.path.append('../py')  # Adjust the path as needed

from utils import parse_raw 

data, gf = parse_raw("../data_p/gft_23_1.bin", np.float32)

n = int(np.cbrt(data.shape[1]))
gap = 0 
id = 6
# plt.plot(gf[id])
for i in range(10):
    draw_separated_lattice_cube(n, gap, data[i], f"data{i}.pdf")

