# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 19:39:56 2019

@author: ramy_
"""



from imutils.video import VideoStream
from imutils.video import FPS
import datetime
import argparse
import imutils
import time
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

from flask import Response
from flask import Flask
from flask import render_template
import threading

import pickle
model = pickle.load(open('final_RF_model.sav', 'rb'))

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
 
# initialize a flask object
app = Flask(__name__)

print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs = cv2.VideoCapture(0)
time.sleep(2.0)
fps = FPS().start()

## NOTE: another way: video_capture = cv2.VideoCapture(0)
# https://realpython.com/face-detection-in-python-using-a-webcam/


cascPath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

while True:
    frame = vs.read()
    frame = imutils.resize(frame[1], width=700)
    input_face = np.asarray(frame, dtype='uint8')
    input_face = cv2.cvtColor(input_face, cv2.COLOR_RGB2GRAY)
    detected_faces = faceCascade.detectMultiScale(
            input_face,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(10, 10),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    if detected_faces != ():
        for (x, y, w, h) in detected_faces:
            horizontal_offset = int(0.15*w)
            vertical_offset = int(0.2*h)
            extracted_face = input_face[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
        extracted_face = Image.fromarray(extracted_face)
        extracted_face = extracted_face.resize((164, 164))
        extracted_face = np.asarray(extracted_face)
        extracted_face = np.asarray(extracted_face)
    else:
        input_face = Image.fromarray(input_face)
        input_face = input_face.resize((164, 164))
        input_face = np.asarray(input_face)
        extracted_face = np.asarray(input_face)

    nx, ny = extracted_face.shape
    extracted_face_dim = extracted_face.reshape((nx*ny))
    extracted_face_dim = extracted_face_dim.reshape((1, -1))
    prediction = model.predict(extracted_face_dim)
    
    if prediction == 0:
        label = "No smile"
    elif prediction == 1:
        label = "Smile"
     
	# draw the prediction on the frame
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
	
	# show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
 
	# update the FPS counter
    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()