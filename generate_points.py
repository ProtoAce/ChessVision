from skimage.transform import hough_line, hough_line_peaks
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import linear_model
from numpy.linalg import norm
import itertools

plot = True
# function from - https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = int(det(d, xdiff) / div)
    y = int(det(d, ydiff) / div)
    return x, y


def point_to_line_dist(line, point):
    x1 = 0
    y1 = (x1-line['x0']) * line['slope'] + line['y0']
    
    p1 = np.array((line['x0'], line['y0']))
    p2 = np.array((x1, y1))
    dist = np.abs(np.cross(p2 - p1, p1- point))/norm(p2 - p1)
    return dist

def point_to_point_dist(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def dedupe_points(points):
    for it, point in points.iterrows():
        for it2, point2 in points.iterrows():
            if it < it2:
                if point_to_point_dist((point['x'], point['y']), (point2['x'], point2['y'])) < 25:
                    points.drop([it2], inplace=True)
    return points
    
def get_points_from_lines(lines, points):
        
    for it, line in lines.iterrows():
        for it2, line2 in lines.iterrows():
            if it < it2:
                x1_1 = line['x0']
                y1_1 = line['y0']
                x1_2 = 0
                y1_2 = line['y_int']

                x2_1 = line2['x0']
                y2_1 = line2['y0']
                x2_2 = 0
                y2_2 = line2['y_int']

                l1 = [(x1_1, y1_1), (x1_2, y1_2)]
                l2 = [(x2_1, y2_1), (x2_2, y2_2)]

                x, y = line_intersection(l1, l2)
                if x > 0 and x < 700 and y > 0 and y < 500:
                    points.loc[len(points)] = ([x, y, it, it2])
    return points

def get_points_from_corners(lines, corners, points, jnbX, jnbY):
    for corner in corners:
        for it, line in lines.iterrows():
            if point_to_line_dist(line, corner) < 10:
                x_board = None
                y_board = None
                if line['direction'] == 'horizontal':
                    y_board = line['index_horizontal']
                    x_board = int(jnbX.predict([corner[0]])[0])
                else:
                    x_board = line['index_vertical']
                    y_board = int(jnbY.predict([corner[1]])[0])
                x = int(corner[0])
                y = int(corner[1])

                points.loc[len(points)] = ([x,y, x_board, y_board])
    return points


def generate_points(lines, corners, jnbX, jnbY):
    points = pd.DataFrame(columns = ['x', 'y', 'x_board', 'y_board'])

    
    points = get_points_from_lines(lines, points)
    
    points = get_points_from_corners(lines, corners, points, jnbX, jnbY)
    
    points = dedupe_points(points)
    
    indecies = list(itertools.product(np.arange(0,9), np.arange(0,9)))
    current_points = [tuple([point['x_board'], point['y_board']]) for it, point in points.iterrows()]
    missing_points = [point for point in indecies if point not in current_points]
    prev_missing_points = float('inf')

    
    if plot:
        image = cv2.imread('images/chessboardWebcam.jpg')
        for iter, line in lines.iterrows():
            plt.axline((line['x0'], line['y0']), slope=line['slope'])
        for it, point in points.iterrows():
            x = int(point['x'])
            y = int(point['y'])
            cv2.circle(image,(x,y),4,255,-1)

        plt.imshow(image),plt.show()

          
        



