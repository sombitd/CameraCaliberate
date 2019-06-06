
##hello here i am doing timepass
import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# # in mm
ssqaure = 25.2
objp = np.zeros((5*3,3), np.float32)
objp[:,:2] = np.mgrid[0:5,0:3].T.reshape(-1,2)*ssqaure 
# axis = np.float32([[44.6,0,0], [0,44.6,0], [0,0,44.6]]).reshape(-1,3)
imgpoints = []
objpoints =[]
images = glob.glob('./images/*.png')

gray = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (5,3),None) 

    # If found, add object points, image points (after refining them)
 
    if ret == True:
        print("found " , fname, " image")
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(5,3),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (5,5), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(50)
imgpoints=np.array(imgpoints)
print imgpoints.shape
objpoints=np.array(objpoints)
# objpoints = objpoints[0,:,0:2]
print objpoints.shape
# obj=[]
# for i in range (0,20):
#     obj.append(objpoints)
# # obj=np.array(obj)
# # obj = np.reshape(obj,(20,42,1,-1))
# print obj
cv2.destroyAllWindows()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print "K ",mtx
print "D ",dist
print gray.shape
