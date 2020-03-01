import numpy as np
import cv2
import glob

cap = cv2.VideoCapture(1)


# def make_480p():
#     cap.set(3, 640)
#     cap.set(4, 512)


# make_480p()
count = 0
while True:
    # Capture frame-by-frame
    if cap.isOpened():
        ret, frame = cap.read()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, blur = cv2.threshold(blur, 120, 255, 0)

        # Display the resulting frame
        cv2.imshow("frame", blur)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

        if key == ord("c"):

            img_name = "mixpic/image_{}.jpg".format(count)
            cv2.imwrite(img_name, blur)
            print("{} lagret".format(img_name))
            count += 1

    else:
        ret = False
        break

# When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
