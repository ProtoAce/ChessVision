import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jenkspy
import itertools
import scipy
from numpy.linalg import norm
import cv2
plot = False


#gets all possible assignments of each line to the line positions on a chess board
# TODO - add compatibility for horizontal lines
def get_possible_assignments(lines):
    vertical_lines_count = len(lines)

    assignment_positions = [list(i) for i in itertools.product([0, 1], repeat=9) if sum(i) == vertical_lines_count]
    assignments = []

    for assignment_position in assignment_positions:
        line_index = 0
        assignment = []
        for index, position in enumerate(assignment_position):
            if position == 1:
                assignment.append((index, lines.iloc[line_index]['x_int']))
                line_index += 1
                if line_index > vertical_lines_count:
                    break
        assignments.append(assignment)
    return assignments

#perform a linear regression on the x intercept (verttical lines) or average height (horizontal lines) of each line
#to see which index each line fits best on the board
def get_best_assignments(assignments):
    bestScore = float("-inf")
    bestAssignment = []
    for assignment in assignments:
        assignment = pd.DataFrame(assignment, columns = ['x', 'y'])
        score = scipy.stats.linregress(assignment['x'], assignment['y']).rvalue
        if score > bestScore:
            bestScore = score
            bestAssignment = [assignment]
        # elif score == bestScore:
        #     bestAssignment.append(assignment)

    return bestAssignment    

#calculate how well the shi tomasi corners fit each line assignment and return the best one
#ignore corners that are too far away from any line
def get_best_shift(assignments, lines, jnbX, jnbY):
    bestShift = []
    for shift in assignments:
        for it, assignment in shift.iterrows():
            line = lines.loc[lines['x_int'] == assignment['y']]
            #predict which category the vertical line goes into based on its
            #x value at y=400. I use y = 400 because the image is warped and the
            #jenks natural breaks dont corrolate as well on the far side of the board
            index = jnbX.predict((400-line['y_int'])/line['slope'])[0]
            bestShift.append([index, assignment['y']])
    return bestShift
    
    


def cluster_corners(corners):
    x_values = corners[:, 0]
    y_values = corners[:, 1]

    jnbX = jenkspy.JenksNaturalBreaks(9)
    jnbX.fit(x_values)
    x_clusters = [int(1.0*sum(x)/len(x)) for x in jnbX.groups_]
    jnbY = jenkspy.JenksNaturalBreaks(9)
    jnbY.fit(y_values)
    y_clusters = [int(1.0*sum(x)/len(x)) for x in jnbY.groups_]
    
    if plot == True:
        image = cv2.imread('images/chessboardWebcam.jpg')

        for x in x_clusters:
            for y in y_clusters:
                cv2.circle(image,(x,y),3,255,-1)
            cv2.circle(image,(x,y),3,255,-1)

        plt.imshow(image),plt.show()
    return jnbX, jnbY


def point_to_line_distance(line, point):
    x1 = 0
    y1 = (x1-line['x']) * line['slope'] + line['y']
    
    p1 = np.array((line['x'], line['y']))
    p2 = np.array((x1, y1))
    dist = np.abs(np.cross(p2 - p1, p1- point))/norm(p2 - p1)
    return dist



def get_line_index(lines, jnbX, jnbY):
    horizontal_lines = lines[lines['direction'] == 'horizontal']
    vertical_lines = lines[lines['direction'] == 'vertical'].sort_values(by=['x_int'])

    assignments = get_possible_assignments(vertical_lines)
    best_assignments = get_best_assignments(assignments)
    best_shift = get_best_shift(best_assignments, lines, jnbX, jnbY)
    return best_shift
    




def assign_lines(angles, dists, corners): 

    data = {'x0': dists*np.cos(angles),
                'y0': dists*np.sin(angles),
                'slope': np.tan(angles+np.pi/2),
                'angle': angles,
                'dist': dists}

    lines = pd.DataFrame(data)


    #positive slope for all slopes
    # lines['anglePos'] = lines['angle']
    # lines.loc[lines['anglePos'] < 0, ['anglePos']] = lines['anglePos'] + np.pi
    
    lines['y_int'] = lines['y0'] - lines['slope']*lines['x0']
    lines['x_int'] = -lines['y_int']/lines['slope']
    

    #sorting lines into vertical and horizontal categories(idk which one is which yet)
    lines.sort_values(by=['angle'], inplace=True)
    breaks = jenkspy.jenks_breaks(lines['angle'], n_classes=2)
    lines['direction'] = pd.cut(lines['angle'], 
                                bins=breaks, 
                                labels=['vertical', 'horizontal'],
                                include_lowest=True)
    
    jnbX, jnbY= cluster_corners(corners)

    best_shift = pd.DataFrame(get_line_index(lines, jnbX, jnbY), columns=['index', 'x_int'])
    print(best_shift)
    lines = pd.merge(lines, best_shift, on="x_int", how="left")

    print(lines)

    if plot == True:
        fig, ax = plt.subplots()
        ax[0].scatter(corners[:, 0], corners[:, 1], c='r', s=10)
        for iter, line in lines.iterrows():
            ax[0].axline((line['x0'], line['y0']), slope=line['slope'])
        ax[0].scatter(lines['x0'], lines['y0'], c='b', s=10)
    