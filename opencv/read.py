import glob
import cv2
import numpy as np


original = cv2.imread("capture/image.jpg")
# Sift and Flann
sift = cv2.xfeatures2d.SURF_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Load all the images
all_images_to_compare = []
titles = []
for f in glob.iglob("one_pic/*"):

    image = cv2.imread(f)
    titles.append(f)
    all_images_to_compare.append(image)

percentiage = 0
for image_to_compare, title in zip(all_images_to_compare, titles):
    # 1) Check if 2 images are equals
    if original.shape == image_to_compare.shape:
        print("The images have same size and channels")
        difference = cv2.subtract(original, image_to_compare)
        b, g, r = cv2.split(difference)
        if (
            cv2.countNonZero(b) == 0
            and cv2.countNonZero(g) == 0
            and cv2.countNonZero(r) == 0
        ):
            print("How good it's the matche: {0:.1f}%".format(percentiage))

    # 2) Check for similarities between the 2 images

    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
    print("Kp1: " + str(len(kp_1)))
    print("Kp2: " + str(len(kp_2)))  # 1 med 30

    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_points = []
    for m, n in matches:
        if m.distance < 0.76 * n.distance:
            good_points.append(m)

    number_keypoints = 0

    if len(kp_1) <= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)

    percentiage = len(good_points) * 100 / number_keypoints
    print("Title: " + title)
    if percentiage > 60:
        print("How good it's the matche: {0:.1f}%".format(percentiage))
        result = cv2.drawMatches(
            original, kp_1, image_to_compare, kp_2, good_points, None
        )
        img = cv2.resize(result, None, fx=0.2, fy=0.2)

        cv2.imshow("The result", img)
        cv2.imwrite("diff/feature_matching.jpg", result)
    else:
        print("How good it's the matche: {0:.1f}%".format(percentiage))
