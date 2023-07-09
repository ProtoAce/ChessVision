import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import jenkspy
from skimage.measure import LineModelND, ransac
from numpy.linalg import norm
from shapely import LineString

def purge_inliers(corners, lines):
    visited = np.zeros(len(corners))

    new_corners = []
    for index, corner in enumerate(corners):
        minDist = float('inf')
        for iter, line in lines.iterrows():
            x1 = 30
            y1 = (x1-line['x0']) * line['slope'] + line['y0']
            
            p1 = np.array((line['x0'], line['y0']))
            p2 = np.array((x1, y1))
            dist = np.abs(np.cross(p2 - p1, p1- corner))/norm(p2 - p1)
            if dist < minDist: minDist = dist
            if minDist < 30: 
                visited[index] += 1
                minDist = float('inf')
        if visited[index] < 2: new_corners.append(corner)
    return np.array(new_corners)

def get_points_on_line(line):
    p1 = None
    p2 = None
    x1 = 0
    y1 = (x1-line['x0']) * line['slope'] + line['y0']
    if(y1 < 500 and y1 > 0):
        p1 = (x1, y1)
    if p1 == None:
        x1 == 700
        y1 = (x1-line['x0']) * line['slope'] + line['y0']
        if(y1 < 500 and y1 > 0):
            p1 = (x1, y1)
    y2 = 0
    x2 = (y2 - line['y0'])/line['slope'] + line['x0']
    if p1 == None:
        p1 = (x2, y2)
    else:
        p2 = (x2, y2)
    if p2 == None:
        y2 = 500
        x2 = (y2 - line['y0'])/line['slope'] + line['x0']
        p2 = (x2, y2)
    
    return p1, p2


def distance_between_lines(line1, line2):
    (x1_1, y1_1), (x1_2, y1_2) = get_points_on_line(line1)
    (x2_1, y2_1), (x2_2, y2_2) = get_points_on_line(line2)

    line1 = LineString([[x1_1, y1_1], [x1_2, y1_2]])
    line2 = LineString([[x2_1, y2_1], [x2_2, y2_2]])
    dis = line1.distance(line2)
    return dis

def check_duplicate_lines(newLine, lines):
    for iter, line in lines.iterrows():
        if distance_between_lines(newLine, line) < 10:
            if abs(newLine['slope'] - line['slope']) < 0.5:
                return True
    return False

#sequential ransac to generate chess grid
def generate_new_lines(lines, corners):
    print(lines)
    if len(lines) < 18:
        model_robust, inline = ransac(corners, LineModelND, min_samples=2,
                                        residual_threshold=10, max_trials=1000)
        
        line_x = corners[:, 0]
        line_y_robust = model_robust.predict_y(line_x)
        x, y = line_x[0], line_y_robust[0]
        x2, y2 = line_x[-1], line_y_robust[-1]
        slope = (y2-y)/(x2-x)

        duplicate = check_duplicate_lines({'x0': x, 'y0': y, 'slope': slope}, lines)

         
        fig, ax = plt.subplots()
        ax.scatter(corners[:, 0], corners[:, 1], c='r', s=10)
        ax.axline((x, y), slope= slope)
        ax.set_xlim(0, 700)
        ax.set_ylim(0, 500)
        for iter, line in lines.iterrows():
            ax.axline((line['x0'], line['y0']), slope=line['slope'])
        ax.scatter(lines['x0'], lines['y0'], c='b', s=10)
        plt.show()

        if duplicate:
            tmpCorners = corners[ ~inline]
            lines = generate_new_lines(lines, tmpCorners)
            lines = generate_new_lines(lines, corners)
            
           
        else:
            lines.loc[len(lines)] = {'x0': x, 'y0': y, 'slope': slope}
            corners = purge_inliers(corners, lines)
            # pd.concat([lines, generate_new_lines(lines, corners)], ignore_index=True).drop_duplicates(inplace = True)
            lines = generate_new_lines(lines, corners)
            # print(lines)
            # return lines

    return lines

            


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
                                labels=['horizontal', 'vertical'],
                                include_lowest=True)


    fig, ax = plt.subplots()

    # plot corners and lines
    # ax[0].scatter(corners[:, 0], corners[:, 1], c='r', s=10)
    # for iter, line in lines.iterrows():
    #     ax[0].axline((line['x0'], line['y0']), slope=line['slope'])
    # ax[0].scatter(lines['x0'], lines['y0'], c='b', s=10)
    
    # print(corners.shape)

    # corners = purge_inliers(corners, lines)
    # print(corners.shape)

    # ax[1].scatter(corners[:, 0], corners[:, 1], c='r', s=10)
    # for iter, line in lines.iterrows():
    #     ax[1].axline((line['x0'], line['y0']), slope=line['slope'])
    # ax[1].scatter(lines['x0'], lines['y0'], c='b', s=10)
    # plt.show()


    #sequential ransac
    corners = purge_inliers(corners, lines)
    lines = generate_new_lines(lines, corners)
    # tmpCorners = None
    # usingTmp = False

    # rng=np.random.default_rng()
    # while corners.shape[0] > 10 and len(lines) < 18:
    #     if usingTmp:
    #         model_robust, inline = ransac(tmpCorners, LineModelND, min_samples=2,
    #                                     residual_threshold=1, max_trials=1000)
    #         usingTmp = False
    #     else:
    #         model_robust, inline = ransac(corners, LineModelND, min_samples=2,
    #                                     residual_threshold=1, max_trials=1000)
        
    #     line_x = corners[:, 0]
    #     line_y_robust = model_robust.predict_y(line_x)
    #     x, y = line_x[0], line_y_robust[0]
    #     x2, y2 = line_x[-1], line_y_robust[-1]
    #     slope = (y2-y)/(x2-x)

    #     duplicate = check_duplicate_lines({'x0': x, 'y0': y, 'slope': slope}, lines)

    #     if not duplicate:
    #         lines.loc[len(lines)] = {'x0': x, 'y0': y, 'slope': slope}
    #     else:
    #         tmpCorners = pd.DataFrame(corners, columns=['x', 'y'])
    #         tmpCorners['inline'] = inline
    #         tmpCorners = tmpCorners[tmpCorners['inline'] == False]
    #         usingTmp = True
            


        # fig, ax = plt.subplots(2)

        # # plot corners and lines
        # ax[0].scatter(corners[:, 0], corners[:, 1], c='r', s=10)
        # for iter, line in lines.iterrows():
        #     ax[0].axline((line['x0'], line['y0']), slope=line['slope'])
        # ax[0].scatter(lines['x0'], lines['y0'], c='b', s=10)
        
        # print(corners.shape)

        # corners = purge_inliers(corners, lines)
        # print(corners.shape)

        # ax[1].scatter(corners[:, 0], corners[:, 1], c='r', s=10)
        # for iter, line in lines.iterrows():
        #     ax[1].axline((line['x0'], line['y0']), slope=line['slope'])
        # ax[1].scatter(lines['x0'], lines['y0'], c='b', s=10)
        # # plt.show()


    
    fig, ax = plt.subplots()
    ax.scatter(corners[:, 0], corners[:, 1], c='r', s=10)
    for iter, line in lines.iterrows():
        ax.axline((line['x0'], line['y0']), slope=line['slope'])
    ax.scatter(lines['x0'], lines['y0'], c='b', s=10)

    # ax.plot(corners[inliers, 0], corners[inliers, 1], '.b', alpha=0.6,
    #         label='Inlier data')
    # ax.plot(corners[outliers, 0], corners[outliers, 1], '.r', alpha=0.6,
    #         label='Outlier data')
    # ax.plot(line_x, line_y_robust, '-b', label='Robust line model')
    # ax.legend(loc='lower left')
    plt.show()

    #   plot corners and lines

