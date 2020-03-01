import cv2 as cv
import numpy as np

img = cv.imread("capture/image.jpg", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("one_pic/image_10.jpg", cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (712, 600))
img2 = cv.resize(img2, (712, 600))
diff = cv.subtract(img, img2)
res = not np.any(diff)

if (res) is True:
    print("The image is the same")
else:
    cv.imwrite("diff/res.jpg", diff)
    print("the image is not the same")

