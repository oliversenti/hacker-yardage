import requests
import json
import matplotlib.pyplot as plt
import Constants  # Import your API key here
from PIL import Image, ImageDraw, ImageFont

# Fetching elevations and storing them
key = Constants.API_KEY

def calcElevation(lat, lon):
    lat = str(lat)
    lon = str(lon)
    url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={lat},{lon}&key={key}"
    json_response = requests.get(url).json()
    elevation = json_response['results'][0]['elevation']
    return round(elevation, 2)

# Example list of lat/lon points (you'd replace this with your full list)
waypoints = [
    (1.3244957, 103.9689049),
    (1.3244684, 103.9689045),
    (1.3244508, 103.9689068),
    (1.3244360, 103.9689163),
    (1.3244210, 103.9689273),
    (1.3244129, 103.9689478),
    (1.3244095, 103.9689685),
    (1.3244064, 103.9689864),
    (1.3243967, 103.9690116),
    (1.3243909, 103.9690394),
    (1.3243860, 103.9690659),
    (1.3243859, 103.9690873),
    (1.3243900, 103.9691108),
    (1.3244022, 103.9691279),
    (1.3244188, 103.9691381),
    (1.3244383, 103.9691436),
    (1.3244570, 103.9691473),
    (1.3244796, 103.9691476),
    (1.3245012, 103.9691438),
    (1.3245164, 103.9691367),
    (1.3245281, 103.9691292),
    (1.3245418, 103.9691136),
    (1.3245511, 103.9690930),
    (1.3245579, 103.9690708),
    (1.3245610, 103.9690448),
    (1.3245605, 103.9690182),
    (1.3245599, 103.9689939),
    (1.3245575, 103.9689751),
    (1.3245539, 103.9689562),
    (1.3245487, 103.9689438),
    (1.3245388, 103.9689281),
    (1.3245278, 103.9689134),
    (1.3245140, 103.9689068),
    (1.3244957, 103.9689049)  # Closing the loop back to the starting point
]


# Fetch elevations for each point
elevations = [calcElevation(lat, lon) for lat, lon in waypoints]

# Create a new image with background color or load a base map image
img = Image.new('RGB', (800, 600), color=(255, 255, 255))  # Blank image for simplicity
draw = ImageDraw.Draw(img)

# Plot points and display elevations
for i, (lat, lon) in enumerate(waypoints):
    x, y = int(i * 20 + 50), int(300 - elevations[i])  # Adjust x, y for visual alignment
    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="blue", outline="blue")  # Plot point
    draw.text((x + 5, y), f"{elevations[i]}m", fill="black")  # Label elevation
    
for i in range(len(elevations) - 1):
    slope = elevations[i+1] - elevations[i]
    color = "red" if slope > 0 else "green"  # Uphill in red, downhill in green
    x1, y1 = int(i * 20 + 50), int(300 - elevations[i])
    x2, y2 = int((i + 1) * 20 + 50), int(300 - elevations[i + 1])
    draw.line((x1, y1, x2, y2), fill=color, width=2)


# Display the resulting image
img.show()