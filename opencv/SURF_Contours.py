import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread("capture/image_10_gray.jpg")  # queryImage
img2 = cv2.imread("one_pic/grayImage.jpg")  # trainImage
imgGray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgGray1, 150, 255, 0)
_, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# print("The number of contours are: " + str(len(contours)))
contours_img = cv2.drawContours(img1, contours, -1, (0, 255, 0), 3)
imgGray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret2, thresh2 = cv2.threshold(imgGray2, 150, 255, 0)
_, contours2, _ = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_img2 = cv2.drawContours(img2, contours2, -1, (0, 255, 0), 3)
# Initiate SIFT detector
sift = cv2.xfeatures2d.SURF_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(contours_img, None)
kp2, des2 = sift.detectAndCompute(contours_img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in xrange(len(matches))]

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]

draw_params = dict(
    matchColor=(0, 255, 0),
    singlePointColor=(255, 0, 0),
    matchesMask=matchesMask,
    flags=0,
)

img3 = cv2.drawMatchesKnn(
    contours_img, kp1, contours_img2, kp2, matches, None, **draw_params
)
plt.imshow(img3,), plt.show()

