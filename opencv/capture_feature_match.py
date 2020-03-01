import numpy as np
import cv2
import glob

# cap = cv2.VideoCapture(0)


# def make_480p():
#     cap.set(3, 640)
#     cap.set(4, 512)


# make_480p()

# while True:
#     # Capture frame-by-frame
#     if cap.isOpened():
#         ret, frame = cap.read()
#         # b = cv2.resize(frame, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
#         # Our operations on the frame come here
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(gray, (5, 5), 0)
#         _, blur = cv2.threshold(blur, 120, 255, 0)

#         # Display the resulting frame
#         cv2.imshow("frame", blur)
#         key = cv2.waitKey(1)
#         if key == ord("q"):
#             break

#         if key == ord("c"):

#             img_name = "capture/image.jpg"
#             cv2.imwrite(img_name, blur)
#             print("{} lagret".format(img_name))

#     else:
#         ret = False
#         break

# When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

original = cv2.imread("capture/image.jpg")
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
original = cv2.GaussianBlur(original, (5, 5), 0)
_, original = cv2.threshold(original, 120, 255, 0)

# Sift and Flann
sift = cv2.xfeatures2d.SURF_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Load all the images
all_images_to_compare = []
titles = []
for f in glob.iglob("mixpic/*"):

    image = cv2.imread(f)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _, image = cv2.threshold(image, 120, 255, 0)

    titles.append(f)
    all_images_to_compare.append(image)

percentiage = 0
for image_to_compare, title in zip(all_images_to_compare, titles):
    # 1) Check if 2 images are equals
    # if original.shape == image_to_compare.shape:
    #     print("The images have same size and channels")
    #     print("Title of same size and channels: " + title)

    #     difference = cv2.subtract(original, image_to_compare)
    #     b, g, r = cv2.split(difference)
    #     if (
    #         cv2.countNonZero(b) == 0
    #         and cv2.countNonZero(g) == 0
    #         and cv2.countNonZero(r) == 0
    #     ):
    #         print("How good it's the matche: {0:.1f}%".format(percentiage))

    # 2) Check for similarities between the 2 images

    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
    print("Kp1: " + str(len(kp_1)))
    print("Kp2: " + str(len(kp_2)))  # 1 med 30

    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_points = []
    for m, n in matches:
        if m.distance < 0.89 * n.distance:
            good_points.append(m)
    number_keypoints = 0

    if len(kp_1) > len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)

    percentiage = len(good_points) * 100 / number_keypoints
    f = open("diff/per.txt", "a")
    f.write("{:.0f}\n".format(percentiage))

    # np.savetxt("diff/per.txt", [percentiage], fmt="%s")
    print("Title: " + title)
    if percentiage > 60:
        print("How good it's the matche: {0:.1f}%".format(percentiage))
        result = cv2.drawMatches(
            original, kp_1, image_to_compare, kp_2, good_points, None
        )
        img = cv2.resize(result, None, fx=0.3, fy=0.3)

        cv2.imshow("The result", img)
        cv2.imwrite("diff/feature_matching.jpg", result)
    else:
        print("How good it's the matche: {0:.1f}%".format(percentiage))

    f = open("diff/per.txt", "r+")
    nums = f.readlines()
    nums = [int(i) for i in nums]
    print(max(nums))
    print(nums)
    f.seek(0)
    f.truncate()

