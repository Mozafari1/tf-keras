import numpy as np
import cv2
import glob
import xlsxwriter
import time

cap = cv2.VideoCapture(1)
count = 0
workbook = xlsxwriter.Workbook("diff/match_{}.xlsx".format(count))
worksheet = workbook.add_worksheet()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    original = frame
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    original = cv2.GaussianBlur(original, (5, 5), 0)
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

    percentiage = 0
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
        #         print("How good it's the matche: {0:.1f}%".format(percentiage))

        # 2) Check for similarities between the 2 images

        kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
        # print("Kp1: " + str(len(kp_1)))
        # print("Kp2: " + str(len(kp_2)))  # 1 med 30

        matches = flann.knnMatch(desc_1, desc_2, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.81 * n.distance:
                good_points.append(m)

        number_keypoints = 0

        if len(kp_1) <= len(kp_2):
            number_keypoints = len(kp_1)
        else:
            number_keypoints = len(kp_2)

        percentiage = len(good_points) * 100 / number_keypoints
        print("Title: " + title)
        if percentiage > 50:
            print("How good it's the matche: {0:.1f}%".format(percentiage))
            result = cv2.drawMatches(
                original, kp_1, image_to_compare, kp_2, good_points, None
            )
            img = cv2.resize(result, None, fx=0.7, fy=0.7)

            cv2.imshow("The result", img)

            name_of_img = "diff/feature_matching.jpg"

            cv2.imwrite(name_of_img, result)

            row = 0
            column = 0
            content = ([title, "Med ", name_of_img, "{0:.1f}%".format(percentiage)],)

            for name, med, org_name, per in content:
                worksheet.write(row, column, name)
                worksheet.write(row, column + 1, med)
                worksheet.write(row, column + 2, org_name)
                worksheet.write(row, column + 3, per)
                row += 1
            workbook.close()
            count += 1
        else:
            print("How good it's the matche: {0:.1f}%".format(percentiage))

    # Display the resulting frame
    cv2.imshow("frame", original)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
