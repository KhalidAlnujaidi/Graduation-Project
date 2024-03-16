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
import pandas as pd
import json
import time




views = Blueprint('views', __name__)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Load custom model # best is very shitty
model = torch.hub.load('ultralytics/yolov5', 'yolov5m').to(device)
# Desired classes
classes = [0, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
# 21 is bear

# Set Model Settings
model.eval()
model.conf = 0.55  # confidence threshold (0-1)
model.iou = 0.45  # NMS Intersection over Union (IoU) threshold (0-1)
model.classes = classes # have model only detect 


# Desired class for recording (Bear)
desired_class_for_recording = 'bear'
# Define recording parameters
record_start_threshold = 0.5  # Adjust as needed
record_stop_threshold = 0.2   # Adjust as needed
record_duration = 10          # Duration to record after the bear leaves (in seconds)
is_recording = False
record_start_time = 0
output_folder = "media"
# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

name_frequency_dict = {}
# Define the start recording function
def start_recording(img_BGR):
    global is_recording, record_start_time, video_writer
    is_recording = True
    record_start_time = time.time()
    print("Start Recording...")
    # Define the video writer
    video_filename = os.path.join(output_folder, f"recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' instead of 'XVID'
    video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (img_BGR.shape[1], img_BGR.shape[0]))

# Define the stop recording function
def stop_recording():
    global is_recording, video_writer
    is_recording = False
    print("Stop Recording.")
    # Release the video writer
    video_writer.release()

# Generate webcam connection
def gen_frames():
    global name_frequency_dict, is_recording, record_start_time

    cap=cv2.VideoCapture(0)

    # Read until video is completed
    while(cap.isOpened()):
        
        # Capture frame-by-frame 
        success, frame = cap.read()
        if success == True:

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes() 
            img = Image.open(io.BytesIO(frame))
            results = model(img, size=640)

            names_column = results.pandas().xyxy[0]['name']

            name_frequency_dict = {}

            # Iterate through the names and update the dictionary
            for name in names_column:
                if name in name_frequency_dict:
                    name_frequency_dict[name] += 1
                else:
                    name_frequency_dict[name] = 1

            # Print or use the dictionary as needed
            print(name_frequency_dict)

            # Check if the desired class (Bear) is detected
            detected_bear = desired_class_for_recording in name_frequency_dict.keys()

            if detected_bear and not is_recording:
                start_recording(img_BGR)

            if is_recording:
                # Write the frame to the video file
                video_writer.write(img_BGR)

            # Check if the bear is no longer detected and stop recording after a delay
            if is_recording and not detected_bear and (time.time() - record_start_time) > record_duration:
                stop_recording()

            # Convert remove single-dimensional entries from the shape of an array
            img = np.squeeze(results.render()) #RGB
            # Read image as BGR
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR

        else:
            print("error connecting to webcam")
            break
        
        # Encode BGR image to bytes so that cv2 will convert to RGB
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

@views.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def get_name_frequency_dict():
    # Your logic to obtain the dictionary
    return name_frequency_dict
@views.route('/get_name_frequency_dict')
def get_name_frequency_dict_route():
    return jsonify(get_name_frequency_dict())

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

import os
from flask import send_from_directory

@views.route('/play/<path:video_name>')
def play_video(video_name):
    videos_folder = r'C:\Users\khalid\Desktop\Grad Project\media'  # has to be absolute  Path to your videos folder
    return send_from_directory(videos_folder, video_name, as_attachment=True)

@views.route('/media', methods=['GET', 'POST'])
@login_required
def media():
    # Specify the path to the directory containing videos
    video_directory = r'C:\Users\khalid\Desktop\Grad Project\media'

    # Get a list of video files in the directory
    

    video_files = [file for file in os.listdir(video_directory) if file.endswith('.mp4')]

    return render_template("media.html", video_files=video_files, user=current_user)