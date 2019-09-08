# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
#import datetime
import imutils
import time
import cv2
import numpy as np
from PIL import Image
import pickle

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(src=0).start()
#fps = FPS().start()
vs = cv2.VideoCapture(0)
time.sleep(2.0)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

model = pickle.load(open('final_RF_model.sav', 'rb'))

cascPath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

def detect_motion(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
    global vs, outputFrame, lock

	# initialize the motion detector and the total number of frames
	# read thus far
    total = 0

	# loop over frames from the video stream
    while True:
        frame = vs.read()
        frame = imutils.resize(frame[1], width=600)
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
        input_face = np.asarray(frame, dtype='uint8')
        input_face = cv2.cvtColor(input_face, cv2.COLOR_RGB2GRAY)
        if total > frameCount:
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
            try:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            except:
                pass
            #print ("No face detected")
		# if the total number of frames has reached a sufficient
		# number to construct a reasonable background model, then
		# continue to process the frame
#        if total > frameCount:
#			# detect motion in the image
#            motion = md.detect(gray)
#
#			# cehck to see if motion was found in the frame
#            if motion is not None:
#				# unpack the tuple and draw the box surrounding the
#				# "motion area" on the output frame
#                (thresh, (minX, minY, maxX, maxY)) = motion
#                cv2.rectangle(frame, (minX, minY), (maxX, maxY),
#					(0, 0, 255), 2)
		
		# update the background model and increment the total number
		# of frames read thus far
        total += 1

		# acquire the lock, set the output frame, and release the
		# lock
        with lock:
            outputFrame = frame.copy()
		
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()