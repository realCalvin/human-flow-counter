from imutils.video import FileVideoStream
from PIL import Image
import numpy as np
import imutils
import cv2
from util.tracker import *
from util.aws import *

object_counter = {}
counter = 0

tracker = EuclideanDistTracker()

CLASSES = ["person"]

# load trained model
print("Loading trained model...")
net = cv2.dnn.readNetFromCaffe(
		'./model/MobileNetSSD_deploy.prototxt.txt', 
		'./model/MobileNetSSD_deploy.caffemodel'
		)

# initialize video stream
print("Starting video stream...")
vid = FileVideoStream("./data/data_1.mp4").start()

# loop over the frames from the video stream
while True:
	# get frame and resize to width 1500
	frame = vid.read()
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

    # draw vertical line for counter
	cv2.line(frame, (1270, 0), (1270, 1000), (255, 255, 0), thickness=2)

	objects = []

	# loop over each detection
	for i in np.arange(0, detections.shape[2]):
		# determine confidence of prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by confidence level
		if confidence > 0.2:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			if endY-startY > 500:
				objects.append([startX, startY, endX-startX, endY-startY, confidence])

	# generate id and track objects w/ euclidean distance
	boxes = tracker.update(objects)
	for box in boxes:
		x, y, w, h, id, conf = box

		# generate label
		label = "{} {}: {:.2f}%".format('Human', id, conf*100)
		cv2.putText(frame, label, (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)

		# draw rectangle border
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# draw centroid
		centerCoordinate = (int(x+(w/2)), int(y+((h)/2)))
		cv2.circle(frame, centerCoordinate, 2, (255, 255, 0), 2)

		# handle counter
		if (centerCoordinate[0] >= 1240 and centerCoordinate[0] <= 1280):
			# check if the id is counted
			if id not in object_counter:
				object_counter[id] = 1
				counter += 1
				
				# crop image
				cropped_frame = frame[y:y+h, x:x+w]
				cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
				img_crop = Image.fromarray(cropped_frame)
				filename = './extracted_images/person_' + str(id) + '.png'
				img_crop.save(filename)

				# pass image into aws rekognition
				detect_face(filename)
		
		# display counter
		cv2.putText(frame, "Counter: {}".format(counter), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

	# show the output frame
	cv2.imshow("Frame", frame)
	if cv2.waitKey(40) == 27:
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vid.stop()