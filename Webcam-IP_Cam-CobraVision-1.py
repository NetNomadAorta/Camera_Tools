import cv2
from cv2 import VideoWriter
from cv2 import VideoWriter_fourcc
import os
import sys
import torch
from torchvision import models
# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")
import time
from datetime import datetime
from torchvision.utils import draw_bounding_boxes
import torchvision
# Now, we will define our transforms
from albumentations.pytorch import ToTensorV2
import winsound
import dropbox
import smtplib
from email.message import EmailMessage
from email.utils import make_msgid
import yaml
import random
import json
import numpy as np
from Configuration import *
from Detector import *


# User parameters
SET_VIDEO_RECORD_FORCE = False
SET_VIDEO_RECORD_ON_OD = True
EMAILER_TOGGLE = True
IMAGE_SCALER = 2.7
SKIP_FRAMES = 20




if (__name__=="__main__"):
    # =============================================================================
    # Starting stopwatch to see how long process takes
    start_time = time.time()

    DetectConfig = DetectionConfiguration(model_name="YOLOv7")  # model_name: "YOLOv7" or "ResNet50"
    DetectConfig.printSettings()
    Detector = Detector(DetectConfig)
    Detector.setup()

    detection_method = "Security"

    # Windows beep settings
    frequency = 700  # Set Frequency To 2500 Hertz
    duration = 80  # Set Duration To 1000 ms == 1 second

    cv2.namedWindow("preview")
    settings = yaml.safe_load( open("config.yaml") )
    camera_ip_info = settings['camera_ip_info'][0]
    from_addr   = settings['from_addr']
    to_addr     = settings['to_addr']
    password        = settings['password']

    vc = cv2.VideoCapture(camera_ip_info)
    video_fps = round( vc.get(cv2.CAP_PROP_FPS) ) # PLACE THIS IN THE 14.0 SLOT BELOW AT VideoWriter(~,~,...)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

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
    frames_to_pred = SKIP_FRAMES

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
            else:
                continue

            # Object detection part
            # -------------------------------------------------------------------------
            if SET_VIDEO_RECORD_ON_OD or EMAILER_TOGGLE:
                if ii % frames_to_pred == 0:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    coordinates, scores, class_indexes = Detector.detect(image_rgb)
                    image = Detector.dim_polygon_section(frame, alpha=0.2)

                    if detection_method == "Safety":
                        image = Detector.safety(image, coordinates, scores, class_indexes, is_det_frame=True)
                    elif detection_method == "Security":
                        image, classes_found = Detector.security(image, coordinates, scores, class_indexes, is_det_frame=True)
                # -------------------------------------------------------------------------
                for coordinate_index, coordinate in enumerate(coordinates):
                    start_point = ( int(coordinate[0]), int(coordinate[1]) )
                    end_point = ( int(coordinate[2]), int(coordinate[3]) )
                    color = (255, 0, 255)
                    thickness = 3
                    cv2.rectangle(frame, start_point, end_point, color, thickness)

                    start_point_text = (start_point[0], max(start_point[1]-10,0) )
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 2
                    thickness = 2
                    cv2.putText(frame, str(class_indexes[coordinate_index]),
                                start_point_text, font, fontScale, color, thickness)

            # Records video if object found
            if ii % 5 == 0:
                if SET_VIDEO_RECORD_ON_OD or EMAILER_TOGGLE:
                    if len(classes_found) > 0:
                        if not currently_recording:
                            winsound.Beep(frequency, 40)
                            now_nicer_format = datetime.now()
                            now_nicer_format = now_nicer_format.strftime("%Y/%m/%d - %H:%M:%S")
                            now = datetime.now()
                            now = now.strftime("%Y_%m_%d-%H_%M_%S")
                            labels_string = ""
                            labels_string_nicer_format_subject = ""
                            labels_string_nicer_format_body = ""
                            for label in classes_found:
                                labels_string += "-" + label
                                labels_string_nicer_format_subject += ", " + label
                                labels_string_nicer_format_body += ", " + label.lower()

                            if SET_VIDEO_RECORD_ON_OD:
                                video = VideoWriter("Saved_Videos/"+now+labels_string+".mp4",
                                                    VideoWriter_fourcc(*'MP4V'), 4.0,
                                                    (int(frame_unedited.shape[1]),
                                                     int(frame_unedited.shape[0])
                                                     )
                                                    )

                            vid_writer_index = 1
                            email_frame_index = 0

                        else:
                            if SET_VIDEO_RECORD_ON_OD:
                                video.write(frame_unedited)

                            vid_writer_index += 1

                            if EMAILER_TOGGLE:
                                if email_frame_index <= 3 and ii % frames_to_pred == 0:
                                    email_frame_index += 1

                                    # Emailer section
                                    # ---------------------------------------------------------
                                    subject = 'Found ' + labels_string_nicer_format_subject[2:]
                                    body = ('The A.I. script has found a ' + labels_string_nicer_format_body[2:]
                                            + " at " + now_nicer_format + ".")
                                    msg = EmailMessage()
                                    msg.add_header('from', from_addr)
                                    msg.add_header('to', ', '.join(to_addr))
                                    msg.add_header('subject', subject)

                                    # Saves images
                                    if email_frame_index <= 1:
                                        now_sub = datetime.now()
                                        now_sub = now_sub.strftime("%Y_%m_%d-%H_%M_%S")

                                        # For naming images
                                        img_1_name = now_sub + "-image_1"
                                        img_2_name = now_sub + "-image_2"
                                        img_3_name = now_sub + "-image_3"

                                        cv2.imwrite("./Images/Screenshot_Images/{}.jpg".format(img_1_name),
                                                    frame_unedited[
                                                        int(min(coordinates[:, 1])):int(max(coordinates[:, 3])),
                                                        int(min(coordinates[:, 0])):int(max(coordinates[:, 2]))
                                                        ]
                                                    )
                                    elif email_frame_index <= 2:
                                        cv2.imwrite("./Images/Screenshot_Images/{}.jpg".format(img_2_name),
                                                    frame_unedited[
                                                        int(min(coordinates[:, 1])):int(max(coordinates[:, 3])),
                                                        int(min(coordinates[:, 0])):int(max(coordinates[:, 2]))
                                                        ]
                                                    )
                                    elif email_frame_index <= 3:
                                        cv2.imwrite("./Images/Screenshot_Images/{}.jpg".format(img_3_name),
                                                    frame_unedited[
                                                        int(min(coordinates[:, 1])):int(max(coordinates[:, 3])),
                                                        int(min(coordinates[:, 0])):int(max(coordinates[:, 2]))
                                                        ]
                                                    )

                                    if email_frame_index >= 3:
                                        attachment_cid_1 = make_msgid()
                                        attachment_cid_2 = make_msgid()
                                        attachment_cid_3 = make_msgid()

                                        msg.set_content(
                                            '<b>%s</b><br/><br/><img src="cid:%s"/><br/><br/><img src="cid:%s"/><br/><br/><img src="cid:%s"/><br/>' % (
                                                body, attachment_cid_1[1:-1], attachment_cid_2[1:-1], attachment_cid_3[1:-1]), 'html'
                                            )

                                        with open("./Images/Screenshot_Images/{}.jpg".format(img_1_name), 'rb') as f:
                                            msg.add_related(
                                                f.read(), 'image', 'jpeg', cid=attachment_cid_1)

                                        with open("./Images/Screenshot_Images/{}.jpg".format(img_2_name), 'rb') as f:
                                            msg.add_related(
                                                f.read(), 'image', 'jpeg', cid=attachment_cid_2)

                                        with open("./Images/Screenshot_Images/{}.jpg".format(img_3_name), 'rb') as f:
                                            msg.add_related(
                                                f.read(), 'image', 'jpeg', cid=attachment_cid_3)

                                        try:
                                            server = smtplib.SMTP('smtp.gmail.com', 587)
                                            server.ehlo()
                                            server.starttls()
                                            server.ehlo()
                                            server.login(from_addr, password)
                                            server.send_message(msg, from_addr=from_addr, to_addrs=to_addr)
                                            server.quit()
                                        except:
                                            print('Something went wrong...')

                                        # os.remove("payload_1.jpg")
                                        # os.remove("payload_2.jpg")
                                        # os.remove("payload_3.jpg")
                                    # # ---------------------------------------------------------





                        currently_recording = True
                        video_stopper_delayer_index = 0

                    else:
                        # If statement basically delays when to stop recording incase
                        #  the predicter didn't catch an object thus stop recording prematurely
                        if video_stopper_delayer_index >= 20:
                            if currently_recording:
                                if SET_VIDEO_RECORD_ON_OD:
                                    video.release()

                                # # Uploads to dropbox
                                # with open("Saved_Videos/"+now+".mp4", "rb") as f:
                                #         try:
                                #             dbx.files_upload(f.read(), '/Backyard_Footage/'+now+".mp4")
                                #             os.remove('/Backyard_Footage/'+now+".mp4")
                                #         except:
                                #             print("EHHH")

                            currently_recording = False
                        else:
                            if currently_recording:
                                if SET_VIDEO_RECORD_ON_OD:
                                    video.write(frame_unedited)

                        video_stopper_delayer_index += 1

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
                # print("  ", round(tenScale/fps_time_lapsed, 2), "FPS - " + now,
                #       end="\r")
                print("  ", round(tenScale/fps_time_lapsed, 2), "FPS - " + now)
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