'''
    Used to demonstrate trained model.
    Loads keypoint data from output/ directory and projects it on video as well as model predictions.
    Saves video with predictions and keyframe data to video file
'''

import json
import numpy as np
import cv2
import os
import pickle

# Load model
f = open('../full_dataset_model.sav','rb')
m = pickle.load(f)
# Load keypoints
keypoint_path = "output/"
keypoints = []
for file in os.listdir(keypoint_path):
    f = open(keypoint_path + file)
    data = json.load(f)
    if data['people'] != []:
        keypoints.append(np.array( [data['people'][0]['pose_keypoints_2d']] ))
    
    

# Play video while analyzing each frame
cap = cv2.VideoCapture("out.avi")
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height)
result = cv2.VideoWriter('keypoint_out.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size) 

frame_ndx = 0
while True:
    if cap.grab():
        flag, frame = cap.retrieve()
        pred = str(m.predict(keypoints[frame_ndx]))
        cv2.putText(frame,pred,(10,500), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
        result.write(frame)
        if not flag:
            continue
        else:
            cv2.imshow('video', frame)
    if cv2.waitKey(10) == 27:
        break
    
    frame_ndx += 1
