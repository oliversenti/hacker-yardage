import numpy as np
import requests

import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from scipy.interpolate import splprep, splev
import matplotlib.tri as tri
import time  # To add delays if needed to avoid rate limits

from earthelevation import calcElevation
import Constants  # Import your API key here


# Fetching elevations and storing them
key = Constants.API_KEY

# Define the green boundary as a polygon with your provided waypoints (lat, lon).
waypoints = [(1.3244957, 103.9689049), (1.3244684, 103.9689045), (1.3244508, 103.9689068),
             (1.3244360, 103.9689163), (1.3244210, 103.9689273), (1.3244129, 103.9689478),
             (1.3244095, 103.9689685), (1.3244064, 103.9689864), (1.3243967, 103.9690116),
             (1.3243909, 103.9690394), (1.3243860, 103.9690659), (1.3243859, 103.9690873),
             (1.3243900, 103.9691108), (1.3244022, 103.9691279), (1.3244188, 103.9691381),
             (1.3244383, 103.9691436), (1.3244570, 103.9691473), (1.3244796, 103.9691476),
             (1.3245012, 103.9691438), (1.3245164, 103.9691367), (1.3245281, 103.9691292),
             (1.3245418, 103.9691136), (1.3245511, 103.9690930), (1.3245579, 103.9690708),
             (1.3245610, 103.9690448), (1.3245605, 103.9690182), (1.3245599, 103.9689939),
             (1.3245575, 103.9689751), (1.3245539, 103.9689562), (1.3245487, 103.9689438),
             (1.3245388, 103.9689281), (1.3245278, 103.9689134), (1.3245140, 103.9689068)]
polygon = Polygon(waypoints)

# Convert to numpy array for easy manipulation
waypoints_np = np.array(waypoints)

# Use a smoothing spline to interpolate the boundary
#s=0 the spline will go exactly through all points, no smoothing. the higher the value the more the spline will deviate from the original points. 
""" By setting per=1, the spline is forced to wrap around, making the curve loop back to the start. 
This is useful if you have a boundary or path that should be continuous (such as a polygon or a loop). """
""" tck: This is the tuple containing the spline representation (knots, coefficients, and degree of the spline).
u: This is a parameter array that represents the positions of the waypoints along the spline, typically in the range of 0 to 1. """

tck, u = splprep([waypoints_np[:, 0], waypoints_np[:, 1]], s=1e-3, per=1)
print("tck: ", tck)
print("u: ", u)
smooth_lat, smooth_lon = splev(np.linspace(0, 1, 1000), tck)

# Plotting the smoothed boundary and grid
plt.figure(figsize=(10, 10))
plt.plot(smooth_lon, smooth_lat, '-', color='red', linewidth=2, label="Smoothed Boundary")
# Plot the polygon boundary (closed loop)

# Generate grid points spaced 3 yards apart within the bounding box of the polygon.
grid_spacing = 0.000025  # Roughly 3 yards in lat/lon coordinates.
min_lat, min_lon, max_lat, max_lon = polygon.bounds
latitudes = np.arange(min_lat, max_lat, grid_spacing)
longitudes = np.arange(min_lon, max_lon, grid_spacing)

# Initialize lists to store grid points within the polygon and their elevations
grid_lat_lons = []
elevations = []

# Identify points within the polygon
for lat in latitudes:
    for lon in longitudes:
        point = Point(lat, lon)
        if polygon.contains(point):
            try:
                elevation = calcElevation(lat, lon)
                grid_lat_lons.append([lat, lon])
                elevations.append(elevation)
                time.sleep(0.1)  # Optional delay for rate limit
            except (IndexError, KeyError) as e:
                print(f"Error fetching elevation for ({lat}, {lon}): {e}")
                continue

# Convert grid_lat_lons to a NumPy array for plotting
grid_lat_lons = np.array(grid_lat_lons)
latitudes = grid_lat_lons[:, 0]
longitudes = grid_lat_lons[:, 1]

# Plotting the polygon (green boundary) and grid points with connecting lines
#plt.figure(figsize=(10, 8))

# Plot the polygon boundary (closed loop)
polygon_boundary = np.array(waypoints + [waypoints[0]])  # Closing the polygon
plt.plot(polygon_boundary[:, 1], polygon_boundary[:, 0], 'g-', linewidth=2, label='Green Boundary')

# Plot lines between grid points (connect adjacent points)
for i in range(len(latitudes)):
    for j in range(i+1, len(latitudes)):
        lat1, lon1 = latitudes[i], longitudes[i]
        lat2, lon2 = latitudes[j], longitudes[j]
        
        # Check if both points are close enough to form a grid (approximately 3-yard spacing)
        if abs(lat1 - lat2) <= grid_spacing and abs(lon1 - lon2) <= grid_spacing:
            plt.plot([lon1, lon2], [lat1, lat2], 'b-', linewidth=0.5)  # Draw blue line connecting points

# Customize the plot
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("3-Yard Grid Overlay on Green Boundary with Connecting Lines")
plt.legend()
plt.grid(False)  # Remove the default grid
plt.show()

# Add labels and a color bar.
plt.colorbar(label='Elevation (feet)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Contour Plot with Elevations')

# Display the plot.
plt.show()