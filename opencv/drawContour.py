import cv2 as cv
import numpy as np

img = cv.imread("image/IMG_0538.jpg")
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(imgGray, (5, 5), 0)
ret, thresh = cv.threshold(blur, 130, 255, 0)
_, contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)


img2 = cv.imread("take_picture/image_0.jpg")
imgGray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

blur2 = cv.GaussianBlur(imgGray2, (5, 5), 0)
ret2, thresh2 = cv.threshold(blur2, 130, 255, 0)

_, contours2, _ = cv.findContours(thresh2, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
number_of_cont1 = int(len(contours))
number_of_cont2 = int(len(contours2))

print("the numer of contours = ", number_of_cont1)
# print(contours[0])
print("the numer of contours2 = ", number_of_cont2)

# print(contours2[-1])
# for i in range(0, number_of_cont1):
#     x = contours[i] == contours2[i]
#     print(x)

# with open("diff/count1.txt", "a") as f:
#     with np.printoptions(threshold=np.inf):
#         for x in contours:
#             # i = contours[x] == contours2[x]

#             f.write(str(x))

# print(row)

# np.savetxt("diff/count0.txt", row, fmt="%s")
# print(contours, type)
cont = cv.drawContours(blur2, contours2, -1, (0, 255, 0), 3)
# cont = cv.drawContours(img2, contours2, -1, (0, 232, 0), 3)

cv.imshow("Image", img2)
cv.imshow("Blur", blur2)

# np.savetxt("diff/count.txt", contours, fmt="%s")
# cv.imwrite("capture/grayImage.jpg", cont)

cv.waitKey(0)
cv.destroyAllWindows()
