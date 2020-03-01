import cv2 as cv
import numpy as np
import glob

img1 = cv.imread("one_pic/image_10.jpg", 0)
ret, thresh = cv.threshold(img1, 127, 255, 0)
contours, hierarchy, _ = cv.findContours(thresh, 2, 1)
all_images_to_compare = []
titles = []
for f in glob.iglob("one_pic/*"):

    image = cv.imread(f, 0)
    titles.append(f)
    all_images_to_compare.append(image)


for image_to_compare, title in zip(all_images_to_compare, titles):

    ret, thresh2 = cv.threshold(image_to_compare, 127, 255, 0)
    cnt1 = contours[0]
    contours, hierarchy, _ = cv.findContours(thresh2, 2, 1)
    cnt2 = contours[0]
    print("Title: " + title)
    ret = cv.matchShapes(cnt1, cnt2, 1, 0.0)
    print(ret)
