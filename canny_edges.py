import cv2
import numpy as np

cv2.namedWindow('Result')
img = cv2.imread("./images/chessboardWebcam.jpg")
v1 = 0
v2 = 0

def doEdges():
    edges = cv2.Canny(img,v1,v2)
    edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
    res = np.concatenate((img,edges),axis = 0)
    cv2.imshow('Result',edges)
def setVal1(val):
    global v1
    v1 = val
    doEdges()
def setVal2(val):
    global v2
    v2 = val
    doEdges()

cv2.createTrackbar('Val1','Result',0,1000,setVal1)
cv2.createTrackbar('Val2','Result',0,1000,setVal2)

cv2.imshow('Result',img)
cv2.waitKey(0)
cv2.destroyAllWindows
