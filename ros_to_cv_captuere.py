

#!/usr/bin/env python
from __future__ import print_function

# import roslib
# roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
i =0
class image_converter:

  def __init__(self):
    # self.image_pub = rospy.Publisher("image_topic_2",Image)
    self.index =i
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/image_raw",Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    (rows,cols,channels) = cv_image.shape


    # print (cv_image.shape)
    cv2.imshow("Image window", cv_image)
    cv2.waitKey(10)

    ret, corners = cv2.findChessboardCorners(cv_image, (5,3),None)
    print (ret)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print (i)
        cv2.imwrite('/home/sombit/camera/images/img'+str(self.index)+'.png',cv_image)
        cv2.waitKey(20)
        self.index = self.index +1   

    

    # try:
    #   # self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    # except CvBridgeError as e:
    #   print(e)

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)