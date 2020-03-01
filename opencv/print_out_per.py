import glob
import cv2
import numpy as np


original = cv2.imread("mixpic/image_0.jpg")
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
original = cv2.GaussianBlur(original, (5, 5), 0)

# Sift and Flann
sift = cv2.xfeatures2d.SURF_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)
# Load all the images
all_images_to_compare = []
titles = []
for f in glob.iglob("mixpic/*"):

    image = cv2.imread(f)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    titles.append(f)
    all_images_to_compare.append(image)


for image_to_compare, title in zip(all_images_to_compare, titles):
    # 1) Check if 2 images are equals
    # if original.shape == image_to_compare.shape:
    #     print("The images have same size and channels")
    #     difference = cv2.subtract(original, image_to_compare)
    #     b, g, r = cv2.split(difference)
    #     if (
    #         cv2.countNonZero(b) == 0
    #         and cv2.countNonZero(g) == 0
    #         and cv2.countNonZero(r) == 0
    #     ):
    #         print("Similarity: 100% (equal size and channels)")

    # 2) Check for similarities between the 2 images

    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
    matches = flann.knnMatch(desc_1, desc_2, k=2)
    good_points = []
    for m, n in matches:
        if m.distance > 0.8 * n.distance:
            good_points.append(m)
    number_keypoints = 0
    if len(kp_1) > len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)
    print("Title: " + title)
    percentage_similarity = len(good_points) * 100 / number_keypoints
    print("Similarity: " + str(int(percentage_similarity)) + "\n")
    cv2.imshow("Ori", original)
    img3 = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
    cv2.imshow("img3", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

