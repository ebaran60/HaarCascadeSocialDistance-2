# import the necessary packages
import numpy as np
import cv2
import math

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# cv2.startWindowThread()

# open webcam video stream
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("VIRAT_S_010204_05_000856_000890.avi")

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output_hog.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640, 480))

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # resizing for faster detection
        frame = cv2.resize(frame, (640, 480))
        # using a greyscale picture, also for faster detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # detect people in the image
        # returns the bounding boxes for the detected objects
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        i = 0
        person_count = 0
        center = []
        original_cord = []
        for (xA, yA, xB, yB) in boxes:
            center.append((int((xA+xB)/2),int((yA+yB)/2)))
            original_cord.append((xA, yA, xB, yB))
            cv2.circle(frame,(int((xA+xB)/2),int((yA+yB)/2)), 0, (0, 0, 255), -1)
            person_count = person_count + 1

            if i>0 :
                for j in range(1, person_count):
                    distance = math.sqrt(math.pow(center[i][0] - center[j - 1][0], 2) + math.pow(center[i][1] -center[j - 1][1], 2) * 1.0)
                    print("distance", distance)
                    if (distance < 50):
                        cv2.rectangle(frame, (original_cord[i][0], original_cord[i][1]),
                                      (original_cord[i][2],original_cord[i][3]),
                                      (0, 0, 255), 2)
                        cv2.rectangle(frame, (original_cord[j - 1][0], original_cord[j - 1][1]),
                                      (original_cord[j - 1][2],original_cord[j - 1][3]),
                                      (0, 0, 255), 2)
                        cv2.line(frame, center[i], center[j - 1], (0, 0, 255), 1)
            i = i +1

        # Write the output video
        out.write(frame.astype('uint8'))
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)