import cv2
import jenkspy
import matplotlib.pyplot as plt

plot = False

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