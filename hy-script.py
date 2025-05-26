
# hacker yardage
# see readme for instructions on mapping a course in OSM


# import all the formulas we need

from hyformulas import *

import os


# Enter bounding box coordinates from OSM here:

latmin = 47.6075     # minimum latitude
lonmin = 8.5650    # minimum longitude
latmax = 47.5987     # maximum latitude
lonmax = 8.5856    # maximum longitude


# do you want to replace existing output files? default is False
# change this to True if you want to overwrite existing files in the output folder

replace_existing = False


# colors for each feature can be customized here

fairway_color = '#85d87e'
tee_box_color = '#85d87e'
green_color = '#a1f29b'
rough_color = '#2ca65e'
tree_color = '#6b9e3f'
water_color = '#8bb2f9'
sand_color = '#ffeea1'
text_color = '#000000'
image_bg_color = '#ffffff'


# how wide are the holes?
# (objects that are more than this number of yards
# from the center line will be filtered out)

hole_width = 50  # yards


# do you want to filter more aggressively near the tees?
# (this can help ignore random bunkers, etc.)
# enter a fraction

short_filter = 1.0

# SVG colors
colors = {
    "fairways": fairway_color,
    "tee boxes": tee_box_color,
    "greens": green_color,
    "rough": rough_color,
    "trees": tree_color,
    "water": water_color,
    "sand": sand_color,
    "text": text_color,
    "woods": tree_color,
    "background": image_bg_color,
}

# passing colors to a dict for the yardage book script as BGR

""" colors = {"fairways":hexToBGR(fairway_color),"tee boxes":hexToBGR(tee_box_color),
"greens":hexToBGR(green_color),"rough":hexToBGR(rough_color),"trees":hexToBGR(tree_color),
"water":hexToBGR(water_color),"sand":hexToBGR(sand_color),"text":hexToBGR(text_color),
"woods":hexToBGR(tree_color)} """


# calculate a medium range filter from the short range filter
# (you could also customize this if you want)

med_filter = (short_filter + 1) / 2


# generate the yardage book

if __name__ == "__main__":
    book = generateYardageBook(latmin, lonmin, latmax, lonmax, replace_existing, colors,
                               filter_width=hole_width, short_factor=short_filter, med_factor=med_filter)
