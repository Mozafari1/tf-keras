import cv2
import numpy as np

original = cv2.imread("one_pic/image_10.jpg")
duplicate = cv2.imread("capture/image.jpg")

if original.shape == duplicate.shape:
    print("The images have same size and channels")
else:
    print("the shape of image is not equal")
difference = cv2.subtract(original, duplicate)
b, g, r = cv2.split(difference)
if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
    print("The images are completely Equal")
else:
    print("The image is not equal")

cv2.imshow("Original", original)
cv2.imshow("Duplicate", duplicate)
cv2.waitKey(0)
cv2.destroyAllWindows()

