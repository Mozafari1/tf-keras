import numpy as np
import cv2
import glob

cap = cv2.VideoCapture(1)


# def make_480p():
#     cap.set(3, 640)
#     cap.set(4, 512)


# make_480p()

while True:
    # Capture frame-by-frame
    if cap.isOpened():
        ret, frame = cap.read()
        # b = cv2.resize(frame, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
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

            img_name = "capture/image.jpg"
            cv2.imwrite(img_name, blur)
            print("{} lagret".format(img_name))

    else:
        ret = False
        break
