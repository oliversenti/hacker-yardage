import numpy as np
import requests

import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import matplotlib.tri as tri
import time  # To add delays if needed to avoid rate limits

import Constants  # Import your API key here

# Fetching elevations and storing them
key = Constants.API_KEY

def calcElevation(lat, lon):
    lat = str(lat)
    lon = str(lon)
    url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat},{lon}&key={key}"
    json_response = requests.get(url).json()
    print(json_response)
    elevation = json_response['results'][0]['elevation']
    return round(elevation, 2)

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

# Plotting the polygon (green boundary) and grid points
plt.figure(figsize=(10, 8))

# Plot the polygon boundary (closed loop)
polygon_boundary = np.array(waypoints + [waypoints[0]])  # Closing the polygon
plt.plot(polygon_boundary[:, 1], polygon_boundary[:, 0], 'g-', linewidth=2, label='Green Boundary')

# Plot grid points inside the polygon
plt.scatter(longitudes, latitudes, c='blue', s=10, label='Grid Points (3 yd spacing)')

# Customize the plot
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("3-Yard Grid Overlay on Green Boundary")
plt.legend()
plt.grid(True)
plt.show()


# Add labels and a color bar.
plt.colorbar(label='Elevation (feet)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Contour Plot with Elevations')

# Display the plot.
plt.show()