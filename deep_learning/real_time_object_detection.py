# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

counter = 0

# construct and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

CLASSES = ["person"]

# load trained model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize video stream
print("[INFO] starting video stream...")
vs = FileVideoStream("../data/data_4.mp4").start()

# loop over the frames from the video stream
while True:
	# get frame and resize
	frame = vs.read()
	if frame is None:
		break
	frame = imutils.resize(frame, width=1500)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# pass the blob through the network and obtain the detections and predictions
	net.setInput(blob)
	detections = net.forward()

    # Draw vertical lines
	cv2.line(frame, (1350, 0), (1350, 1000), (255, 255, 0), thickness=2)

	# loop over each detection
	for i in np.arange(0, detections.shape[2]):
		# determine confidence of prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by confidence level
		if confidence > args["confidence"]:
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# generate label
			label = "{}: {:.2f}%".format('Human', confidence*100)

			# draw rectangle
			cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 0), 2)
			
			# draw centroid
			centerCoordinate = (int(startX+((endX-startX)/2)), int(startY+((endY-startY)/2)))
			print(centerCoordinate)
			cv2.circle(frame, centerCoordinate, 2, (255, 255, 0), 2)

			if (centerCoordinate[0] >= 1345 and centerCoordinate[0] <= 1355):
				counter += 1
				continue

			# draw label
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
			cv2.putText(frame, "Counter: {}".format(counter), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

	# show the output frame
	cv2.imshow("Frame", frame)
	if cv2.waitKey(40) == 27:
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()