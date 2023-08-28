from skimage.measure import label, regionprops
from scipy.spatial import KDTree
from skimage.measure import EllipseModel
import numpy as np
import cv2
from classes import *

def pixel_col(point,img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x,y = point
    r,g,b = img_rgb[round(x)][round(y)]
    if r>g and r>b:
        return 'R'
    elif r<b and g<b:
        return 'B'
    else:
        return 'G'

def filter_centeroids(total_mask):
    labels = label(total_mask)
    regions = regionprops(labels)

    # filter regions
    regions = [
        item for item in regions
        # filter based on size
        if item.area_filled > 2
        and item.area_filled < 165
        # filter based on circle eccentricity
        and item.eccentricity < 0.9
    ]

    # define all point centeroids
    centeroids = []
    for props in regions:
        centeroids.append(props.centroid)
    return centeroids

def get_targets(centeroids, img):
    targets = []
    for point in centeroids:
        tree = KDTree(centeroids)
        points = []
        distances, indices = tree.query([point], k=6)
        for index in indices[0]:
            points.append(centeroids[index])
        a_points = np.array(points)
    
        ell = EllipseModel()
        ell.estimate(a_points)

        max_res = max(ell.residuals(a_points))

        if abs(max_res) < 0.2:
            target = Target([])
            for point in points:
                target.points.append(Point(point,pixel_col(point,img)))
                centeroids.remove(point)
            targets.append(target)
    return targets
