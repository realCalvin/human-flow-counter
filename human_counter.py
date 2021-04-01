import cv2
import imutils
import numpy as np

# Load video into opencv
cap = cv2.VideoCapture('data_2.mp4')

backSub = cv2.createBackgroundSubtractorKNN()

# Declare two video frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    # Determine difference between frame1 and frame
    diff = cv2.absdiff(frame1, frame2)

    # Convert difference into greyscale because it is easier for contour detection
    grey = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Blur the greyscale frame
    blur = cv2.GaussianBlur(grey, (5, 5), 0)

    # Find threshold of blurred greyscale frame
    _, thresh = cv2.threshold(grey, 25, 255, cv2.THRESH_BINARY)

    # Dilate the threshold image to fill in holes which results in better contours
    dilated = cv2.dilate(thresh, None, iterations=15)

    # Determine contour
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # For each contour
    sortedContours = sorted(contours, key=lambda x: cv2.contourArea(x))
    for contour in sortedContours:
        # Define coordinates for rectangle outline
        (x, y, w, h) = cv2.boundingRect(contour)
        # Skip any contours if the area is less than 700 (irrelevant objects)
        if (cv2.contourArea(contour) < 7000):
            continue

        # Draw rectangle
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    # Show it on cv2 gui
    cv2.imshow("Data Footage", frame1)
    cv2.imshow("Blured Greyscale", blur)

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
