from capture_photo import cameraFeed
from line_detection import hough_transform
from corner_detection import shi_tomasi
from assign_lines import assign_lines
from assign_lines_2 import assign_lines as assign_lines_2
from generate_points import generate_points
from cluster_corners import cluster_corners
from multiprocessing import Process
import cv2



if __name__ == '__main__':
    # cam = cv2.VideoCapture(0)

    # image = cameraFeed(cam)
    image = cv2.imread('images/chessboardWebcam.jpg')
    _, angles, dists = hough_transform(image)
    corners = shi_tomasi(image)
    jnbX, jnbY= cluster_corners(corners)

    lines = assign_lines_2(angles, dists, corners, jnbX, jnbY)
    points = generate_points(lines, corners, jnbX, jnbY)
    



    cv2.waitKey(0)

    # cam.release()
    cv2.destroyAllWindows()
    


