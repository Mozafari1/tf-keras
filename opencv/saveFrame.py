import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    if cap.isOpened():
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow("frame", gray)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

        if key == ord("c"):

            img_name = "capture/image.jpg"
            cv2.imwrite(img_name, frame)
            print("{} lagret".format(img_name))

    else:
        ret = False
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

