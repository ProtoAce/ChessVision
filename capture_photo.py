import cv2


def cameraFeed(cam):

    while True:
            
        result, image = cam.read()

        if result:
            cv2.resize(image, None, fx = 0.5, fy= 0.5, interpolation=cv2.INTER_AREA)
            cv2.imshow('test', image)
            
            c = cv2.waitKey(1)
            if c == 27:
                break
            elif c == ord('p'):
                return image
                # cv2.imwrite('./images/chessboardWebcam.jpg', image)

        else:
            print('no image detected')

    