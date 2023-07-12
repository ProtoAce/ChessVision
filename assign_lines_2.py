import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jenkspy
import itertools
import scipy
from numpy.linalg import norm
plot = False


#gets all possible assignments of each line to the line positions on a chess board
def get_possible_assignments(vertical_lines, horizontal_lines):
    vertical_lines_count = len(vertical_lines)
    horizontal_lines_count = len(horizontal_lines)

    assignment_positions_vertical = [list(i) for i in itertools.product([0, 1], repeat=9) if sum(i) == vertical_lines_count]
    assignment_positions_horizontal = [list(i) for i in itertools.product([0, 1], repeat=9) if sum(i) == horizontal_lines_count]
    assignments_vertical = []
    assignments_horizontal = []

    for assignment_position_vertical in assignment_positions_vertical:
        line_index = 0
        assignment_vertical = []
        for index, position in enumerate(assignment_position_vertical):
            if position == 1:
                assignment_vertical.append((index, vertical_lines.iloc[line_index]['x_int']))
                line_index += 1
                if line_index > vertical_lines_count:
                    break
        assignments_vertical.append(assignment_vertical)

    for assignment_position_horizontal in assignment_positions_horizontal:
        line_index = 0
        assignment_horizontal = []
        for index, position in enumerate(assignment_position_horizontal):
            if position == 1:
                assignment_horizontal.append((index, horizontal_lines.iloc[line_index]['avg_height']))
                line_index += 1
                if line_index > horizontal_lines_count:
                    break
        assignments_horizontal.append(assignment_horizontal)


    return assignments_vertical, assignments_horizontal

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
def get_best_shift(assignments_x, assignments_y, lines, jnbX, jnbY):
    bestShift_x = []
    for shift in assignments_x:
        for it, assignment in shift.iterrows():
            line = lines.loc[lines['x_int'] == assignment['y']]
            #predict which category the vertical line goes into based on its
            #x value at y=400. I use y = 400 because the image is warped and the
            #jenks natural breaks dont corrolate as well on the far side of the board
            index = jnbX.predict((400-line['y_int'])/line['slope'])[0]
            bestShift_x.append([index, assignment['y']])

    bestShift_y = []
    for shift in assignments_y:
        for it, assignment in shift.iterrows():
            # line = lines.loc[lines['avg_height'] == assignment['y']]
            #predict which category the horizontal line goes into based on its
            #avergage height.
            index = jnbY.predict(assignment['y']).flatten()[0]
            bestShift_y.append([index, assignment['y']])
    return bestShift_x, bestShift_y
    
    


def point_to_line_distance(line, point):
    x1 = 0
    y1 = (x1-line['x']) * line['slope'] + line['y']
    
    p1 = np.array((line['x'], line['y']))
    p2 = np.array((x1, y1))
    dist = np.abs(np.cross(p2 - p1, p1- point))/norm(p2 - p1)
    return dist



def get_line_index(lines, jnbX, jnbY):
    horizontal_lines = lines[lines['direction'] == 'horizontal'].sort_values(by=['avg_height'])
    vertical_lines = lines[lines['direction'] == 'vertical'].sort_values(by=['x_int'])

    assignments_x, assignments_y = get_possible_assignments(vertical_lines, horizontal_lines)
    best_assignments_x = get_best_assignments(assignments_x)
    best_assignments_y = get_best_assignments(assignments_y)
    best_shift_x, best_shift_y = get_best_shift(best_assignments_x, best_assignments_y, lines, jnbX, jnbY)
    return (best_shift_x), (best_shift_y)
    
#from the missing indecies, use clusters and linear regression to generate a line for each cluster  
def generate_lines(lines,corners, jnbX, jnbY):
    indecies = np.arange(0,9)
    line_indecies_horizontal = lines[lines['direction'] == 'horizontal']['index_horizontal']
    line_indecies_vertical = lines[lines['direction'] == 'vertical']['index_vertical']
    missing_indecies_horizontal = [x for x in indecies if x not in line_indecies_horizontal]
    missing_indecies_vertical = [x for x in indecies if x not in line_indecies_vertical]
    
    for missing_index_horizontal in missing_indecies_horizontal:
        #get the corners that are in the missing index cluster
        corners_in_cluster = np.array([corner for corner in corners if corner[1] in jnbY.groups_[missing_index_horizontal]])
        #perform linear regression on the corners to get the line
        linregress = scipy.stats.linregress(corners_in_cluster)
        x0 = corners_in_cluster[0,0]
        y0 = corners_in_cluster[0,1]
        slope = linregress.slope
        y_int = linregress.intercept
        x_int = (y_int-y0)/slope
        direction = 'horizontal'
        index_horizontal = missing_index_horizontal
        new_line = {'x0': x0, 'y0': y0, 'slope': slope, 'y_int': y_int, 'x_int': x_int, 'direction': direction, 'index_horizontal': index_horizontal}
        lines = lines._append(new_line, ignore_index=True)
        
        linregY = corners_in_cluster[:, 0] * linregress.slope + linregress.intercept
        plt.title("rating vs time")
        plt.xlabel("time")
        plt.ylabel("rating")
        plt.xticks(rotation = 25)
        plt.plot(corners_in_cluster[:, 0], linregY, 'r-')
        plt.scatter(corners[:, 0], corners[:, 1], c='r', s=10)
        plt.scatter(corners_in_cluster[:, 0], corners_in_cluster[:, 1], s=8, c='b')
        for iter, line in lines.iterrows():
            plt.axline((line['x0'], line['y0']), slope=line['slope'])
        # ax[0].scatter(lines['x0'], lines['y0'], c='b', s=10)
        plt.show()


    for missing_index_vertical in missing_indecies_vertical:
        #get the corners that are in the missing index cluster
        corners_in_cluster = np.array([corner for corner in corners if corner[0] in jnbX.groups_[missing_index_vertical]])
        #perform linear regression on the corners to get the line
        linregress = scipy.stats.linregress(corners_in_cluster)
        x0 = corners_in_cluster[0,0]
        y0 = corners_in_cluster[0,1]
        slope = linregress.slope
        y_int = linregress.intercept
        x_int = (y_int-y0)/slope
        direction = 'vertical'
        index_vertical = missing_index_vertical
        new_line = {'x0': x0, 'y0': y0, 'slope': slope, 'y_int': y_int, 'x_int': x_int, 'direction': direction, 'index_vertical': index_vertical}
        lines = lines._append(new_line, ignore_index=True)
    print(lines)
    return lines

    



def assign_lines(angles, dists, corners, jnbX, jnbY): 

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
   
    tmpLines = pd.DataFrame(lines[lines['direction'] == 'horizontal'])
    tmpLines['y_1'] = tmpLines['y0'] + 700*tmpLines['slope']
    tmpLines['avg_height'] = (tmpLines['y_1'] + tmpLines['y0'])/2
    lines = pd.merge(lines, tmpLines[['y0', 'avg_height']], on="y0", how="left")
        
    # jnbX, jnbY= cluster_corners(corners)

    best_shift_x, best_shift_y = get_line_index(lines, jnbX, jnbY)
    best_shift_x = pd.DataFrame(best_shift_x, columns=['index_vertical', 'x_int'])
    best_shift_y = pd.DataFrame(best_shift_y, columns=['index_horizontal', 'avg_height'])
    lines = pd.merge(lines, best_shift_x, on="x_int", how="left")
    lines = pd.merge(lines, best_shift_y, on="avg_height", how="left")

    # lines = generate_lines(lines, corners, jnbX, jnbY)

    if plot == True:
        fig, ax = plt.subplots()
        ax.scatter(corners[:, 0], corners[:, 1], c='r', s=10)
        for iter, line in lines.iterrows():
            ax.axline((line['x0'], line['y0']), slope=line['slope'])
        # ax[0].scatter(lines['x0'], lines['y0'], c='b', s=10)
        plt.show()

    return lines
    