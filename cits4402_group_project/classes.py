import heapq
import numpy as np

def sort_y(point):
  y,_ = point.centeroid
  return y

def sort_x(point):
  _,x = point.centeroid
  return x

class Point:
  def __init__(self, centeroid, colour):
    self.centeroid = centeroid
    self.colour = colour

  def __repr__(self):
    return f"{self.colour}"

class Target:
  def __init__(self, points):
    self.points = points
    self.name = ""

  def __repr__(self):
    return self.name

  def order_points(self):
    org = self.points
    ord = [0]*6

    top = heapq.nsmallest(1, org, key = sort_y)
    bottom = heapq.nlargest(1, org, key = sort_y)

    ord[0] = top[0]
    ord[3] = bottom[0]
    org.remove(top[0])
    org.remove(bottom[0])
  
    rightside = heapq.nlargest(2, org, key = sort_x)
    leftside = heapq.nsmallest(2, org, key = sort_x)

    top_right = heapq.nsmallest(1, rightside, key = sort_y)
    bottom_right = heapq.nlargest(1, rightside, key = sort_y)
    top_left = heapq.nsmallest(1, leftside, key = sort_y)
    bottom_left = heapq.nlargest(1, leftside, key = sort_y)

    ord[1] = top_right[0]
    ord[2] = bottom_right[0]
    ord[4] = bottom_left[0]
    ord[5] = top_left[0]

    self.points = ord
    self._create_name()

  def _create_name(self):
    points = self.points
    name = ""
    for i in range(1,len(points)):
      name = f"{name}{points[i].colour}"
    self.name = name
