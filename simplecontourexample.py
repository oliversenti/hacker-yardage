import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.patheffects as patheffects

# Create a straight line
x = np.array([0, 10])
y = np.array([5, 5])

# Create a plot
fig, ax = plt.subplots()

# Apply alternating tick lengths with a spacing of 5 between them
ticks = [
    patheffects.withTickedStroke(angle=60, length=2, foreground='black'),
    patheffects.withTickedStroke(angle=60, length=2, foreground='red')
]

# Draw ticks
for i in range(0, len(x)-1, 5):
    tick_length = i % 2  # Toggle between 0 and 1 for the pattern (1 black, 1 red)
    line = Line2D(x[i:i+2], y[i:i+2], color='none', lw=2, path_effects=[ticks[tick_length]])
    ax.add_line(line)

# Set axis limits
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

plt.show()
