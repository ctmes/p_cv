import cv2
import numpy as np
import matplotlib.pyplot as plt
from classes import *
from helpers import *

# read in image and convert to HSI colour space
# img = cv2.imread("data/camera 71/2022_12_15_15_51_19_944_rgb.png")
# img = cv2.imread("data/camera 72/2022_12_15_15_51_19_956_rgb.png")
# img = cv2.imread("data/camera 73/2022_12_15_15_51_19_934_rgb.png")
img = cv2.imread("data/camera 74/2022_12_15_15_51_19_951_rgb.png")

# img = cv2.imread("data/camera 11/2022_12_15_15_51_19_927_rgb_left.png")
# img = cv2.imread("data/camera 11/2022_12_15_15_51_19_927_rgb_right.png")

hsv_original_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define HSI min and max for colour segmentation
g_min = (55,35,60)
g_max = (85,255,255)
r_min = (115,60,60)
r_max = (180,255,255)
b_min = (90,60,60)
b_max = (122,255,255)

# create the mask for each colour
gmask = cv2.inRange(hsv_original_img, g_min, g_max)
rmask = cv2.inRange(hsv_original_img, r_min, r_max)
bmask = cv2.inRange(hsv_original_img, b_min, b_max)

# combine masks
total_mask = gmask | rmask | bmask

# get centeroids of each found and filtered connected component
centeroids = filter_centeroids(total_mask)

# filter points based on if they make an ellipse with their 5 closest neighbours
targets = get_targets(centeroids,img)

for target in targets:
    target.order_points()

# visualise reduction
color = iter(plt.cm.rainbow(np.linspace(0, 1, len(targets))))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap=plt.cm.gray)
for target in targets:
    c = next(color)
    for point in target.points:
        y,x = point.centeroid
        plt.plot(x, y, marker="o", markersize=5, markeredgecolor=c, markerfacecolor=c)
    y,x = target.points[0].centeroid
    plt.text(x, y, target.name)
plt.show()
