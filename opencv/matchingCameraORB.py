import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread("images/image_0.jpg", cv.IMREAD_GRAYSCALE)
# image = cv.resize(image, (712, 712))
capture = cv.VideoCapture(0)
sift = cv.xfeatures2d.SURF_create()
kp1, des1 = sift.detectAndCompute(image, None)  # queryImage
# image = cv.drawKeypoints(image, kp1, image)
# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(k=50)  # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params, search_params)

while True:
    _, frame = capture.read()
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(grayFrame, None)  # None is no filter # trainImage
    # grayFrame = cv.drawKeypoints(grayFrame, kp2, grayFrame)
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    # matchesMask = [[0,0] for i in xrange(len(matches))]
    # ratio test as per Lowe's paper
    # for i,(m,n) in enumerate(matches):
    #     if m.distance < 0.7*n.distance:
    #         matchesMask[i] = [1, 0]
    # draw_params = dict(matchColor = (0,255,0),
    #                singlePointColor = (255,0,0),
    #                matchesMask = matchesMask,
    #                flags=0)

    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append(m)
    if len(good) > 10:
        query_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        train_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        # Perspective transform
        height, width = image.shape
        points = np.float32([[0, 0], [0, height], [width, height], [width, 0]]).reshape(
            -1, 1, 2
        )
        # points = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32).reshape(-1, 1, 2)

        # print(matrix)
        # print(matrix)
        dst = cv.perspectiveTransform(points, matrix)

        homo = cv.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 2)
        cv.imshow("Homo", homo)
    else:
        print("gray frame")
        # cv.imshow("GrayFrame", grayFrame)
        # print(query_pts)
    # img3 = cv.drawMatches(image,kp1,grayFrame,kp2,good,grayFrame)
    # cv.imshow("Image", image)
    # cv.imshow("GrayFrame", grayFrame)
    # cv.imshow("img3 ", img3)
    key = cv.waitKey(1)
    if key == 27:
        break


capture.release()
cv.destroyAllWindows()
