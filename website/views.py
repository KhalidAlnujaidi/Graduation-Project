from flask import Blueprint, render_template, request, flash, jsonify, Response
from flask_login import login_required, current_user
from .models import Note
from . import db
import json
import cv2
from PIL import Image
import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np
import torch
from pathlib import Path




views = Blueprint('views', __name__)

# Load custom model # best is very shitty
model = torch.hub.load('ultralytics/yolov5', 'yolov5s') # , "custom", path = "./best_camel.pt", force_reload=True
# Desired classes
classes = [0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

# Set Model Settings
model.eval()
model.conf = 0.75  # confidence threshold (0-1)
model.iou = 0.45  # NMS Intersection over Union (IoU) threshold (0-1)
model.classes = classes # have model only detect wanted classes

# # Load the model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# # Specify the classes you want to detect
# classes = [0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # Person, Bird, Cat, Dog, Horse, Sheep, Cow, Elephant, Bear, Zebra, Giraffe

# # Get the image or video
# img = 'path_to_your_image_or_video'  # replace with your image or video path

# # Perform the detection
# results = model(img, size=640)  # adjust the size to your needs

# # Filter the results to only include the specified classes
# filtered_results = results.xyxy[0][results.xyxy[0][:, 5].isin(classes)]

# Generate webcam connection
def gen_frames():
    
    cap=cv2.VideoCapture(0)
    
    # Read until video is completed
    while(cap.isOpened()):
        
        # Capture frame-by-fram 
        success, frame = cap.read()
        if success == True:

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
        ## remove ai model guessing and such    
            img = Image.open(io.BytesIO(frame))
            results = model(img, size=640)


            #convert remove single-dimensional entries from the shape of an array
            img = np.squeeze(results.render()) #RGB
            # read image as BGR
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR

        else:
            print("error connecting to webcam")
            break
        
        # Encode BGR image to bytes so that cv2 will convert to RGB
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        print(frame)
        
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@views.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST': 
        note = request.form.get('note')#Gets the note from the HTML 

        if len(note) < 1:
            flash('Note is too short!', category='error') 
        else:
            new_note = Note(data=note, user_id=current_user.id)  #providing the schema for the note 
            db.session.add(new_note) #adding the note to the database 
            db.session.commit()
            flash('Note added!', category='success')

    return render_template("home.html", user=current_user)



