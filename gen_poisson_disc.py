import math
import random
import matplotlib.pyplot as plt
import numpy as np

# Helper functions
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

##
def generate_point_around(
    point, 
    radius,
    width,
    height):

    r = radius * (random.random() + 1)
    angle = 2 * math.pi * random.random()
    new_x = point[0] + r * math.cos(angle)
    new_y = point[1] + r * math.sin(angle)
    return new_x, new_y

##
def in_bounds(
    point,
    width,
    height):
    return 0 <= point[0] < width and 0 <= point[1] < height

##
def fits(
    point,
    grid_size,
    grid,
    cols,
    rows,
    radius):

    col = int(point[0] / grid_size)
    row = int(point[1] / grid_size)
    for i in range(max(col - 2, 0), min(col + 3, cols)):
        for j in range(max(row - 2, 0), min(row + 3, rows)):
            neighbor = grid[i][j]
            if neighbor is not None and distance(point, neighbor) < radius:
                return False
    return True

##
def initialize_grid(
    rows,
    cols):

    return [[None for _ in range(rows)] for _ in range(cols)]

##
def create(
    radius,
    grid_size,
    width,
    height,
    num_points,
    show_plots = False):
    
    start_point = (5.0, 5.0)
    
    active = [start_point]
    points = [start_point]

    
    k = 100
    #radius = 1.0
    #grid_size = 1.0 #radius / math.sqrt(2)
    #width = 10
    #height = 10
    #num_points = 32

    cols, rows = int(width / grid_size) + 1, int(height / grid_size) + 1
    grid = initialize_grid(rows = rows, cols = cols)

    rand_index = random.randint(0, len(active) - 1)
    point = active[rand_index]

    while len(points) <= num_points:
        found = False
        for _ in range(k):
            new_point = generate_point_around(
                point = point, 
                radius = radius,
                width = width,
                height = height)
            if (in_bounds(
                    point = new_point, 
                    width = width, 
                    height = height) and 
                fits(
                    point = new_point, 
                    grid_size = grid_size, 
                    grid = grid, 
                    cols = cols, 
                    rows = rows, 
                    radius = radius)
            ):
                
                points.append(new_point)
                active.append(new_point)
                col = int(new_point[0] / grid_size)
                row = int(new_point[1] / grid_size)
                grid[col][row] = new_point
                found = True
                break

        if not found:
            active.pop(rand_index)
            point = active[rand_index]

    normalized_points = []
    normalized_x = []
    normalized_y = []
    for i in range(1, len(points)):
        point = points[i]
        new_point = (
            (point[0] - 5.0) / 5.0,
            (point[1] - 5.0) / 5.0
        ) 
        normalized_points.append(new_point)
        normalized_x.append(new_point[0])
        normalized_y.append(new_point[1])

    if show_plots == True:
        plt.scatter(x = normalized_x, y = normalized_y)
        plt.show()

    return normalized_points
