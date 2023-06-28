import cv2
from cv2 import VideoWriter
from cv2 import VideoWriter_fourcc
import os
import sys
import torch
from torchvision import models
import albumentations as A  # our data augmentation library
# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")
import time
from datetime import datetime
from torchvision.utils import draw_bounding_boxes
from pycocotools.coco import COCO
# Now, we will define our transforms
from albumentations.pytorch import ToTensorV2
import winsound
import dropbox
import smtplib
from email.message import EmailMessage
from email.utils import make_msgid
import yaml
import random


# User parameters
SET_VIDEO_RECORD_FORCE = False
IMAGE_SCALER = 2.0


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )



# Main()
# =============================================================================
# Starting stopwatch to see how long process takes
start_time = time.time()

# Dropbox key
# dbx = dropbox.Dropbox('')

# Windows beep settings
frequency = 700  # Set Frequency To 2500 Hertz
duration = 80  # Set Duration To 1000 ms == 1 second



cv2.namedWindow("preview")
settings = yaml.safe_load( open("config.yaml") )
camera_ip_info = settings['camera_ip_info']
from_addr   = settings['from_addr']
to_addr     = settings['to_addr']
password        = settings['password']

vc = cv2.VideoCapture(camera_ip_info)
video_fps = round( vc.get(cv2.CAP_PROP_FPS) ) # PLACE THIS IN THE 14.0 SLOT BELOW AT VideoWriter(~,~,...)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
print("OKAY", rval)
# For recording video
if SET_VIDEO_RECORD_FORCE:
    winsound.Beep(frequency, 40)
    now = datetime.now()
    now = now.strftime("%Y_%m_%d-%H_%M_%S")
    video = VideoWriter("Saved_Videos/"+"Immediate_Recording-"+now+".mp4", 
                        VideoWriter_fourcc(*'mp4v'), 14.0, 
                        (int(frame.shape[1]), 
                         int(frame.shape[0])
                         )
                        )

# Start FPS timer
fps_start_time = time.time()
ii = 0
video_stopper_delayer_index = 0
tenScale = 100
currently_recording = False

# This while just keeps the party going!
while True:
    if  rval:
        # cv2.imshow("preview", frame)
        cv2.imshow("preview", cv2.resize(frame, 
                                         (int(frame.shape[1]/IMAGE_SCALER), 
                                          int(frame.shape[0]/IMAGE_SCALER)
                                          ), 
                                         interpolation = cv2.INTER_AREA
                                         ) 
                   )
        # cv2.setWindowProperty("preview", cv2.WND_PROP_TOPMOST, 1)
        rval, frame = vc.read()
        if rval:
            frame_unedited = frame.copy()
        
        # Write frame to the video file
        if SET_VIDEO_RECORD_FORCE:
            video.write(frame_unedited)
        
        # Breaks if escape key pressed
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
        
        ii += 1
        if ii % tenScale == 0:
            now = datetime.now()
            now = now.strftime("%Y/%m/%d - %H:%M:%S")
            fps_end_time = time.time()
            fps_time_lapsed = fps_end_time - fps_start_time
            sys.stdout.write('\033[2K\033[1G')
            print("  ", round(tenScale/fps_time_lapsed, 2), "FPS - " + now, 
                  end="\r")
            fps_start_time = time.time()
    else:
        vc = cv2.VideoCapture(camera_ip_info)
        
        if vc.isOpened(): # try to get the first frame
            rval, frame = vc.read()
        else:
            rval = False
    
    # Breaks if escape key pressed (referenced from previous line)
    if key == 27: # exit on ESC
        break

# Release web camera stream
vc.release()
cv2.destroyWindow("preview")

# Release video output file stream
if SET_VIDEO_RECORD_FORCE:
    video.release()

# =============================================================================

print("\nDone!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)