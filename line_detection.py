from skimage.transform import hough_line, hough_line_peaks
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn import linear_model

plot = True

def hough_transform(image):

    # image = cv2.imread("./images/chessboardWebcam.jpg")
    # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    rows,cols,channels= image.shape 

        
    # M = cv2.getRotationMatrix2D((cols/2,rows/2),-30,1) 

    # image = cv2.warpAffine(image,M,(cols,rows)) 

    # Converting image to greyscale and then getting the edge map
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_edge = cv2.Canny(image_grey, 250,  250)



    # #precision of 1 degree (dividing by 180)
    test_angles = np.linspace(-np.pi/2, np.pi/2, 360)

    # #performing hough tranform
    h, theta, d = hough_line(image_edge, test_angles)

    threshold = 0.68* np.amax(h)
    num_peaks = 18
    min_distance = 25
    min_angle = int(np.pi/8)

    # #finding location of peaks
    # h, q, d = hough_line_peaks(h, theta, d, threshold= threshold, num_peaks = num_peaks, min_distance=min_distance)
    # return h, q, d

    if plot:
        # Generating figure 1
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()

        ax[0].imshow(image_edge, cmap='gray')
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        angle_step = 0.5 * np.diff(theta).mean()
        d_step = 0.5 * np.diff(d).mean()
        bounds = [np.rad2deg(theta[0] - angle_step),
                np.rad2deg(theta[-1] + angle_step),
                d[-1] + d_step, d[0] - d_step]
        ax[1].imshow(np.log(1 + h), extent=bounds, cmap='gray', aspect=1 / 1.5)
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')

        ax[2].imshow(image_edge, cmap='gray')
        ax[2].set_ylim((image_edge.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')

    

        for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold= threshold, num_peaks = num_peaks, min_distance=min_distance)):
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            # print(angle, dist)
            ax[2].axline((x0, y0), slope=np.tan(angle + np.pi/2))

        plt.tight_layout()
        plt.show()
    h, q, d = hough_line_peaks(h, theta, d, threshold= threshold, num_peaks = num_peaks, min_distance=min_distance)
    return h, q, d



