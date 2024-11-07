import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# Function to create a fairway with features
def generate_fairway_with_features(length=200, width=50):
    # Create the fairway rectangle
    fairway_vertices = np.array([
        [0, 0],
        [length, 0],
        [length, width],
        [0, width]
    ])

    # Coordinates and heights for the single mound
    mound_position = [100, width / 2]  # Center of the fairway at 100 yards
    mound_size = [80, 30]
    mound_height = 2
    mound_slope = 0.03  # 3% slope

    return fairway_vertices, mound_position, mound_size, mound_height, mound_slope

# Generate fairway data with features
fairway_vertices, mound_position, mound_size, mound_height, mound_slope = generate_fairway_with_features()

# Combine fairway rectangle and mound
x = np.concatenate([fairway_vertices[:, 0], [mound_position[0]]])
y = np.concatenate([fairway_vertices[:, 1], [mound_position[1]]])
z = np.concatenate([np.zeros_like(fairway_vertices[:, 0]), [mound_height]])

# Define a grid within the fairway boundaries
grid_x, grid_y = np.meshgrid(np.arange(0, 201, 1), np.arange(0, 51, 1))

# Interpolate heights using griddata
grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')

# Create a filled contour plot with smooth edges
contour = plt.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis', extend='both')

# Create the first contour line
contour_line = plt.contour(grid_x, grid_y, grid_z, levels=[0], colors='black', linewidths=2)

# Add a solid circumference line around the mound
theta = np.linspace(0, 2*np.pi, 100)
x_mound_circumference = mound_position[0] + mound_size[0]/2 * np.cos(theta)
y_mound_circumference = mound_position[1] + mound_size[1]/2 * np.sin(theta)
plt.plot(x_mound_circumference, y_mound_circumference, color='black', linewidth=2)

# Use quiver to show the elevation direction
arrow_spacing = 10
x_arrow = grid_x[::arrow_spacing, ::arrow_spacing]
y_arrow = grid_y[::arrow_spacing, ::arrow_spacing]
u = np.gradient(grid_z[::arrow_spacing, ::arrow_spacing], axis=1)
v = np.gradient(grid_z[::arrow_spacing, ::arrow_spacing], axis=0)

plt.quiver(x_arrow, y_arrow, u, v, scale=10, scale_units='xy', angles='xy', color='red', width=0.005)

# Plot the mound
plt.scatter(mound_position[0], mound_position[1], c='red', marker='^', label='Mound')

# Remove colorbar
plt.colorbar(contour, label='Height').remove()

plt.title('Elevation Contour Plot within Fairway with Single Mound')
plt.xlabel('X (yards)')
plt.ylabel('Y (yards)')
plt.legend()
plt.show()
