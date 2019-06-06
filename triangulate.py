import cv2 as cv 
from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv
from mpl_toolkits.mplot3d import Axes3D
image = cv.imread("images/img23.png",0)
image2 = cv.imread("images/img24.png",0)
# RESIZE_RATIO = 0.6

# fx = 718.856*RESIZE_RATIO
# fy = 718.856*RESIZE_RATIO
# cx = 607.1928*RESIZE_RATIO
# cy = 185.2157*RESIZE_RATIO
# # print (image2.shape)
# img1 = cv.resize(image,None,fx=RESIZE_RATIO,fy=RESIZE_RATIO)
# img2 = cv.resize(image2,None,fx=RESIZE_RATIO,fy=RESIZE_RATIO)
gray1= cv.cvtColor(image,cv.COLOR_BGR2GRAY)
gray2= cv.cvtColor(image2,cv.COLOR_BGR2GRAY)
print (gray2.shape)
fig, (ax1, ax2) = plt.subplots(1, 2)
# cv.imshow("",img1)
# cv.imshow(":",image2)
# cv.waitKey(0)
sift = cv.xfeatures2d.SIFT_create()        
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
matchesMask = [[0,0] for i in range(0,len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv.drawMatchesKnn(image,kp1,image2,kp2,matches,None,**draw_params)
good=[]
cv.imshow("matches" , img3)
cv.waitKey(0)
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

print ( "good matches",len(good))
src=[]
dst=[]
i_kp=[]
j_kp=[]
for i in range (0,len(good)):
	src.append(kp1[good[i].queryIdx].pt);
	dst.append(kp2[good[i].trainIdx].pt);
	# i_kp.push_back(m[0].queryIdx);
 #    j_kp.push_back(m[0].trainIdx);
src=np.array(src)
dst=np.array(dst)
mask = np.array(len(src))
F,mask =cv.findEssentialMat(dst, src)        # i have the essential matrix
src_n=[]
dst_n=[]

for i in range (0,len(mask)):
    if(mask[i]):
        src_n.append(kp1[good[i].queryIdx].pt);
        dst_n.append(kp2[good[i].trainIdx].pt);
# for i in range(0,len(src_n)):

print (len(src_n))


K =[[796.84546274,0.,336.39652933],
 [  0.,806.50270936,244.72860953],
 [  0., 0.,1.        ]]
K= np.array(K)
src_n = np.asarray(src_n)
dst_n = np.asarray(dst_n)
points, R, t, mask2 = cv.recoverPose(F,dst_n,src_n,K,50)
T = np.eye(4,dtype =float)
invR = np.transpose(R)
invt = np.multiply(np.matmul(invR,t),-1)
rt = inv(R) #inv
rt = rt.tolist()
rt[0].append(invt[0][0])
rt[1].append(invt[1][0])
rt[2].append(invt[2][0])
rt = np.asarray(rt)
print ("rt " ,rt)
print ("T " ,T[0:3])
# print ("rt " ,rt)
print ("dst_n " ,dst_n.shape)
print ("src_n " ,src_n.shape)
for i in range (0,len(src_n)):
    x= int(dst_n[i][0])
    y= int(dst_n[i][1])
    print((x,y),"the image points ")
    cv.circle(gray2,(x,y), 8, (0,0,255), -1)

cv.imshow("hi",gray2)
cv.waitKey(0)
src_n = np.transpose(src_n)
dst_n = np.transpose(dst_n)
T = np.matmul(K,T[0:3])
rt = np.matmul(K,rt)
print (T)
src_img = []
for i in range (0,len(src_n)):
    src_n[0][i] = src_n[0][i]-340
    src_n[1][i] = src_n[1][i]-110
    dst_n[0][i] = dst_n[0][i]-340
    dst_n[1][i] = dst_n[1][i]-110
    
D = cv.triangulatePoints(T,rt,src_n,dst_n)
print (D.shape)
print (D[2][1])
# print (D[3])
x =[]
y = []
z = []
for i in range (0,len(D[0])):
	x.append((D[0][i])/D[3][i])
	y.append((D[1][i])/D[3][i])
	z.append(D[2][i]/D[3][i])
x = np.array(x)
print ((x,y,z),"the 3d co-ordinates")
# z = np.multiply(z,100)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
# ax.set_xlim3d(0, 3)
# ax.set_ylim3d(0, 3)
# ax.set_zlim3d(0, 3)
plt.show()
cv.destroyAllWindows()
# x =[[364],[111],[300]]
# B = np.matmul(K , x)
# x= np.array(x)
# a= B[0]/B[2]
# b = B[1]/B[2]
# print (B)
# print (K)
# print (x)

# cv.circle(image2,(a,b),8,(0,255,0),-1)
# cv.imshow("well ",image2)
# cv.waitKey(0)