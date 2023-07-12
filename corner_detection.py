import numpy as np
import cv2
from matplotlib import pyplot as plt


plot = False

def shi_tomasi(image):

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray,300,0.05,20)
    corners = np.int0(corners)
    

    if plot:
        for i in corners:
            x,y = i.ravel()
            cv2.circle(image,(x,y),3,255,-1)

        plt.imshow(image),plt.show()
    
    len, _, wid = corners.shape
    corners = np.array(corners.reshape(len, wid))
    return corners