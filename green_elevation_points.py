import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, griddata
from simplification.cutil import simplify_coords
from matplotlib import cm
import cv2

from earthelevation import calcElevation

 
def getGreenElevationPoints(image, waypoints):
    # Convert to numpy array for easy manipulation
    print("start drawing elevation points")
    waypoints_np = np.array(waypoints)
    print(waypoints)

    # Simplify waypoints with an epsilon value that reduces complexity but retains shape
    epsilon = 1e-6  # Adjust this to control simplification
    simplified_nds = simplify_coords(waypoints_np, epsilon)

    # Use a smoothing spline to interpolate the boundary
    #s=0 the spline will go exactly through all points, no smoothing. the higher the value the more the spline will deviate from the original points. 
    """ By setting per=1, the spline is forced to wrap around, making the curve loop back to the start. 
    This is useful if you have a boundary or path that should be continuous (such as a polygon or a loop). """

    tck, u = splprep([simplified_nds[:, 0], simplified_nds[:, 1]], s=0, per=1)
    smooth_lat, smooth_lon = splev(np.linspace(0, 1, 100), tck)
    print(smooth_lat, smooth_lon)

    # Set the grid spacing to 3 yards (assuming each degree represents a reasonable yard distance for illustration)
    grid_spacing = 1 / 111000  # Convert 3 yards to degrees for lat/lon


    # Create 3-yard grid within boundary bounds
    lat_min, lat_max = min(smooth_lat) - grid_spacing, max(smooth_lat) + grid_spacing
    lon_min, lon_max = min(smooth_lon) - grid_spacing, max(smooth_lon) + grid_spacing

    # Generate grid coordinates
    lat_grid = np.arange(lat_min, lat_max, grid_spacing)
    lon_grid = np.arange(lon_min, lon_max, grid_spacing)

    # Initialize lists to store grid points and their elevations
    grid_points = []
    elevations = []

    # Loop through each grid point, check if it is within the boundary, and get elevation
    for lat in lat_grid:
        for lon in lon_grid:
            # Placeholder for a function to check if (lat, lon) is inside smoothed polygon
            # if point_inside_polygon(lat, lon, smooth_lat, smooth_lon):
            elevation = calcElevation(lat, lon)
            if elevation is not None:
                grid_points.append((lat, lon))
                elevations.append(elevation)
                

    # Convert lists to numpy arrays
    grid_points = np.array(grid_points)
    elevations = np.array(elevations)
    
    print("the color is detected as green and hence a green is drawn")
    grid_lat_lons = np.array(grid_points)
    elevations = np.array(elevations)
        
    # Normalize elevations to a color map (e.g., viridis from Matplotlib)
    norm = plt.Normalize(vmin=np.min(elevations), vmax=np.max(elevations))
    colormap = cm.get_cmap("viridis")
    # Draw each grid point on the image
    for (lat, lon), elevation in zip(grid_points, elevations):
        x, y = map_to_image_coords(lat, lon)
        color = colormap(norm(elevation))[:3]  # Get RGB color (ignore alpha channel)
        color_bgr = tuple((np.array(color) * 255).astype(int)[::-1])  # Convert RGB to BGR and scale to 0-255
        cv2.circle(image, (x,y), radius=3, color=color_bgr, thickness=-1)  # Draw filled circle
        
    print("done drawing elevation points")
    return (image)

"""  # Interpolate elevations onto a structured grid for contour plotting
    lon_grid_mesh, lat_grid_mesh = np.meshgrid(lon_grid, lat_grid)
    elevation_grid = griddata(grid_points, elevations, (lat_grid_mesh, lon_grid_mesh), method='cubic')


    # Plotting
    plt.figure(figsize=(10, 10))
    plt.plot(smooth_lon, smooth_lat, '-', color='pink', linewidth=2, label="Smoothed Boundary")

    # Contour plot for elevation
    contour = plt.contour(lon_grid_mesh, lat_grid_mesh, elevation_grid, levels=10, linewidths=0.5, colors='black')
    plt.clabel(contour, inline=True, fontsize=8)

    # Scatter plot of grid points colored by elevation
    grid_lat_lons = np.array(grid_points)
    plt.scatter(grid_lat_lons[:, 1], grid_lat_lons[:, 0], c=elevations, cmap='viridis', marker='o')
    plt.colorbar(label='Elevation (m)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.title("3-Yard Grid Points with Elevation")
    plt.show() """
    
    

""" # Create grid lines
lat_min, lat_max = waypoints_np[:, 0].min() - grid_spacing, waypoints_np[:, 0].max() + grid_spacing
lon_min, lon_max = waypoints_np[:, 1].min() - grid_spacing, waypoints_np[:, 1].max() + grid_spacing

# Generate grid points for plotting
lat_grid = np.arange(lat_min, lat_max, grid_spacing)
lon_grid = np.arange(lon_min, lon_max, grid_spacing)

# Draw the grid
for lat in lat_grid:
    plt.plot([lon_min, lon_max], [lat, lat], 'k--', linewidth=0.5)
for lon in lon_grid:
    plt.plot([lon, lon], [lat_min, lat_max], 'k--', linewidth=0.5)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.title("Smoothed Boundary with 3-Yard Grid")
plt.show() """
