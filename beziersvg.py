import svgwrite
import numpy as np

def drawbeziersvg(nds):
  # Create a new SVG drawing 
  dwg = svgwrite.Drawing("bezier_curve.svg", profile="tiny")

  # Create a path element to represent the curve
  print("nodes ", nds)
  path = dwg.path(d="M{}".format(nds[0]))

  # Generate cubic Bezier curve commands
  for i in range(1, len(nds) - 2, 3):
    path.push("C{},{} {},{} {},{} ".format(nds[i][0], nds[i][1], nds[i+1][0], nds[i+1][1], nds[i+2][0], nds[i+2][1]))
  # Set path attributes
  path.stroke("blue")
  path.fill("none")
  

  # Add the path element to the SVG drawing
  dwg.add(path)

  # Save the SVG drawing to a file
  dwg.save()