import numpy as np
import cv2


def main():
    x = 0
    cap1 = cv2.VideoCapture(1)  # Camera 1
    # cap2 = cv2.VideoCapture(1)  # Camera 2

    # @click.command()
    # @click.option('--imwidth', default=1, prompt='Image width')
    # @click.option('--imheight', default=1, prompt='Image height')
    # @click.option('imgs_directory', prompt="Directory to save images in")
    # @click.option('extension', prompt='Image extension')
    i=0
    while(i<50):
        # Capturing frame by frame
        ret, img1 = cap1.read()  # Capture frame by frame from Camera 1
        # ret, img2 = cap2.read()  # Capture frame by frame from Camera 2
        # r1 = imwidth / img1.shape[1]  # Dividing by the width
        # ratio1 = (imwidth, int(img1.shape[0] * r1))
        # # multiplying by the height/width
        # # r2 = imwidth / img2.shape[1]
        # # ratio2 = (imwidth, int(img2.shape[1] * r2))
        # img_res1 = cv2.resize(img1, ratio1, interpolation=cv2.INTER_AREA)
        # # INTER_AREA optimum for scaling down
        # img_res2 = cv2.resize(img2, ratio2, interpolation=cv2.INTER_AREA)
        cv2.imshow("IMG1", img1)
        cv2.waitKey(10)
        gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (8,5),None)

    # If found, add object points, image points (after refining them)
        objpoints = []
        if ret == True:
            # objpoints.append(objp)

            # corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            # imgpoints.append(corners2)

        # Draw and display the corners
            new = cv2.drawChessboardCorners(img1, (7,6), corners,ret)
            cv2.imshow("chess",new)
            cv2.imwrite('images/img'+str(i)+'.png',gray)
            cv2.waitKey(20)
            i = i+1


if __name__ == '__main__':
    main()