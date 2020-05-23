#!/usr/bin/env python

# USAGE
# python predict_video.py --model model/activity.model --label-bin model/lb.pickle --input example_clips/Spam.mp4 --output output/spam_128avg.avi --size 128

# import the necessary packages
from keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
import keras.backend as K
K.set_floatx('float32')
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained serialized model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to  label binarizer")

ap.add_argument("-s", "--size", type=int, default=128,
	help="size of queue for averaging")
args = vars(ap.parse_args())

# load the trained model and label binarizer from disk
print("[INFO] loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

# initialize the image mean for mean subtraction along with the
# predictions queue
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])

# initialize the video stream, pointer to output video file, and
# frame dimensions

vs = cv2.VideoCapture(0)

(W, H) = (None, None)

# loop over frames from the video file stream
while True:
	grabbed, frame = vs.read()
    


	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
	     break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# clone the output frame, then convert it from BGR to RGB
	# ordering, resize the frame to a fixed 96x96, and then
	# perform mean subtraction
	output = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (96, 96)).astype("float32")
	frame -= mean

	# make predictions on the frame and then update the predictions
	# queue
	preds = model.predict(np.expand_dims(frame, axis=0))[0]
	Q.append(preds)

	# perform prediction averaging over the current history of
	# previous predictions
	results = np.array(Q).mean(axis=0)
	i = np.argmax(results)
	label = lb.classes_[i]
     
	if label == 0:
         text = "object type: activity1: {}".format(label)
         cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 5)
         
       
	if label == 1:
         text = "object type: activity2: {}".format(label)
         cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 5)
    

      
	if label == 2:
         text = "object type: activity3: {}".format(label)
         cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 5)
      
	if label == 3:
         text = "object type: activity4: {}".format(label)
         cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 5)                                                                  
     
         
    
	if label == 4:
         text = "object type: activity5: {}".format(label)
         cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 5)


	# show the output image
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		 break

# release the file pointers
print("[INFO] cleaning up...")
vs.release()