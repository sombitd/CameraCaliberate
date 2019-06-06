import cv2 
import numpy as np 
# img = cv2.imread("images/img20.png")
# cv2.imshow("hell",img)
# cv2.waitKey(0)
# img = np.array(img)
# print img.shape
# h , w = img.shape[:2]
h =480
w = 640
mtx  =[[794.52118914  , 0.   ,      328.86300616],
 [  0.       ,  797.58943063 ,235.15754589],
 [  0.         ,  0.         ,  1.        ]]
dist =[[ 3.13280610e-01 ,-3.02495868e+00, -5.84216874e-03, -2.90694305e-02,
   9.11182788e+00]]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
newcameramtx = np.array(newcameramtx)
cap = cv2.VideoCapture(1)
while (True):
	ret , img =cap.read()
	dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
	cv2.imshow("hell",img)
	cv2.imshow("y",dst)
	cv2.waitKey(0)
