from PIL import Image, ImageDraw
import svgwrite
import matplotlib.pyplot as plt
import numpy as np
import math
import numpy as np
from PIL import Image, ImageDraw


def draw_bezier_curve_png(control_points, filename):
    """Draws a Bezier curve between multiple control points and saves it as a PNG image.

    Args:
        control_points: A list of tuples representing the (x, y) coordinates of the control points.
        filename: The filename to save the image as (e.g. 'output.png').
    """
    # Create a new image with a white background
    img = Image.new('RGB', (4960, 7016), color='white')
    
    # Create a draw object
    draw = ImageDraw.Draw(img)

    # Draw the Bezier curve
    draw.line(control_points, fill='black', width=5, joint='curve')
    
    # Save the image as a PNG file
    img.save(filename, 'PNG')



def draw_vector_graphic_svg(points, filename):
    """Draws a vector graphic between multiple points and saves it as an SVG file.

    Args:
        points: A list of tuples representing the (x, y) coordinates of the points.
        filename: The filename to save the SVG file as (e.g. 'output.svg').
    """
    # Create a new SVG object
    dwg = svgwrite.Drawing(filename=filename)

    # Create a path object and add the points to it
    path = dwg.path(d='M{},{}'.format(points[0][0], points[0][1]))
    for point in points[1:]:
        path.push('S{},{}'.format(point[0], point[1]))

        

    # Set the style of the path
    path.stroke(color='red', width='1', linecap='round')


    # Add the path to the SVG object
    dwg.add(path)

    # Save the SVG file
    dwg.save()\



def draw_smooth_curve(points, filename):
    # Calculate the number of segments between the points
    n_segments = len(points) - 1
    
    # Create an array of the x and y coordinates of the points
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    
    # Use numpy to generate a set of points along the curve
    t = np.linspace(0, 1, n_segments * 10)
    fx = np.poly1d(np.polyfit(np.arange(len(x)), x, 2))(t)
    fy = np.poly1d(np.polyfit(np.arange(len(y)), y, 2))(t)
    
    # Use PIL to draw the curve and save it to a file
    image = Image.new('RGBA', (int(max(x)) + 10, int(max(y)) + 10), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    for i in range(len(fx) - 1):
        draw.line((fx[i], fy[i], fx[i + 1], fy[i + 1]), fill=(0, 0, 0, 255), width=2, joint='curve')
    image.save(filename)


        
# Define the control points of the Bezier curve
control_points = [(200, 300), (500, 100), (900, 700), (1200, 300)]

# Call the draw_bezier_curve_png function
draw_bezier_curve_png(control_points, 'output_bezier.png') 

# Call the draw_vector_graphic_svg function
draw_vector_graphic_svg(control_points, 'output.svg')


# Define the center, width, and height of the oval
center = (300, 300)
width = 200
height = 100

# Calculate the x and y coordinates of the 50 points around the oval
points = []
for i in range(50):
    angle = 2 * math.pi * i / 50
    x = center[0] + width * math.cos(angle)
    y = center[1] + height * math.sin(angle)
    points.append((x, y))

# Call the draw_smooth_curve function
draw_smooth_curve(control_points, 'smooth_curve.png')