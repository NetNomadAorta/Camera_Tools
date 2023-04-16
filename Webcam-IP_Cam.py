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
SAVE_NAME_OD = "./Models-OD/Animals-ResNet50-0.model"
DATASET_PATH = "./Training_Data/" + SAVE_NAME_OD.split("./Models-OD/",1)[1].split("-",1)[0] +"/"
MIN_SCORE    = 0.70
SET_VIDEO_RECORD_FORCE = False
SET_VIDEO_RECORD_ON_OD = True
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

dataset_path = DATASET_PATH

#load classes
coco = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
categories = coco.cats
n_classes_1 = len(categories.keys())
categories

classes_1 = [i[1]['name'] for i in categories.items()]


# lets load the faster rcnn model
model_1 = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model_1.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model_1.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes_1)


# TESTING TO LOAD MODEL
if os.path.isfile(SAVE_NAME_OD):
    checkpoint = torch.load(SAVE_NAME_OD)
    model_1.load_state_dict(checkpoint)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use GPU to train
model_1 = model_1.to(device)

model_1.eval()
torch.cuda.empty_cache()

color_list = ['green', 'magenta', 'turquoise', 'red', 'green', 'orange', 'yellow', 'cyan', 'lime']



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

transforms_1 = A.Compose([
    # A.Resize(int(frame.shape[0]/IMAGE_SCALER), int(frame.shape[1]/2IMAGE_SCALER)),
    ToTensorV2()
])

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
        
        # Object detection part
        # -------------------------------------------------------------------------
        if ii % 20 == 0:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            transformed_image = transforms_1(image=image)
            transformed_image = transformed_image["image"]
            
            with torch.no_grad():
                prediction_1 = model_1([(transformed_image/255).to(device)])
                pred_1 = prediction_1[0]
            
            dieCoordinates = pred_1['boxes'][pred_1['scores'] > MIN_SCORE]
            die_class_indexes = pred_1['labels'][pred_1['scores'] > MIN_SCORE]
            # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
            die_scores = pred_1['scores'][pred_1['scores'] > MIN_SCORE]
            
            dieCoordinates = dieCoordinates[die_class_indexes == 4]
            die_scores = die_scores[die_class_indexes == 4]
            die_class_indexes = die_class_indexes[die_class_indexes == 4]
            
            # dieCoordinates_cat_2 = dieCoordinates[die_class_indexes == 7]
            # dieCoordinates = torch.cat((dieCoordinates, dieCoordinates_cat_2), 0)
            
            # die_class_indexes_cat_2 = die_class_indexes[die_class_indexes == 7]
            # die_class_indexes = torch.cat((die_class_indexes, die_class_indexes_cat_2), 0)
            
            # die_scores_cat_2 = die_scores[die_class_indexes == 7]
            # die_scores = torch.cat((die_scores, die_scores_cat_2), 0)
            
            die_class_indexes = die_class_indexes.tolist()
            die_scores = die_scores.tolist()
            
            
            
            # labels_found = [str(int(die_scores[index]*100)) + "% - " + str(classes_1[class_index]) 
            #                 for index, class_index in enumerate(die_class_indexes)]
            labels_found = [str(classes_1[class_index]) 
                            for index, class_index in enumerate(die_class_indexes)]
            classes_found = [str(classes_1[class_index]) 
                             for index, class_index in enumerate(die_class_indexes)]
            
            predicted_image = draw_bounding_boxes(transformed_image,
                boxes = dieCoordinates,
                # labels = [classes_1[i] for i in die_class_indexes], 
                labels = labels_found, # SHOWS SCORE AND INDEX IN LABEL
                width = 2,
                colors = [color_list[i] for i in die_class_indexes]
                )
            
            boxes_widened = dieCoordinates
            # Widens boxes
            for i in range(len(dieCoordinates)):
                box_width = dieCoordinates[i,2]-dieCoordinates[i,0]
                box_height = dieCoordinates[i,3]-dieCoordinates[i,1]
                
                # Width
                boxes_widened[i, 0] = max(dieCoordinates[i][0] - int(box_width/2), 0)
                boxes_widened[i, 2] = min(dieCoordinates[i][2] + int(box_width/2), transformed_image.shape[2])
                
                # Height
                boxes_widened[i, 1] = max(dieCoordinates[i][1] - int(box_height/2), 0)
                boxes_widened[i, 3] = min(dieCoordinates[i][3] + int(box_height/2), transformed_image.shape[1])
            
            dieCoordinates = boxes_widened
            
            # Changes image back to a cv2 friendly format
            frame = predicted_image.permute(1,2,0).contiguous().numpy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # -------------------------------------------------------------------------
        else:
            for dieCoordinate_index, dieCoordinate in enumerate(dieCoordinates):
                start_point = ( int(dieCoordinate[0]), int(dieCoordinate[1]) )
                end_point = ( int(dieCoordinate[2]), int(dieCoordinate[3]) )
                color = (255, 0, 255)
                thickness = 3
                cv2.rectangle(frame, start_point, end_point, color, thickness)
                
                start_point_text = (start_point[0], max(start_point[1]-10,0) )
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 2
                thickness = 2
                cv2.putText(frame, labels_found[dieCoordinate_index], 
                            start_point_text, font, fontScale, color, thickness)
        
        # Records video if object found
        if ii % 5 == 0:
            if len(die_class_indexes) > 0:
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
                    
                    
                else:
                    if SET_VIDEO_RECORD_ON_OD:
                        video.write(frame_unedited)
                    
                    vid_writer_index += 1
                    if vid_writer_index <= 22 and ii % 20 == 0:
                        
                        # Emailer section
                        # ---------------------------------------------------------
                        subject     = 'Found ' + labels_string_nicer_format_subject[2:]
                        body        = ('The A.I. script has found a '+ labels_string_nicer_format_body[2:] 
                                        + " at " + now_nicer_format +".")
                        msg = EmailMessage()
                        msg.add_header('from', from_addr)
                        msg.add_header('to', ', '.join( to_addr ) )
                        msg.add_header('subject', subject)
                        
                        
                        # Saves images
                        if vid_writer_index <= 6:
                            now_sub = datetime.now()
                            now_sub = now_sub.strftime("%Y_%m_%d-%H_%M_%S")
                            
                            # For naming images
                            img_1_name = now_sub+"-image_1"
                            img_2_name = now_sub+"-image_2"
                            img_3_name = now_sub+"-image_3"
                            
                            cv2.imwrite("./Images/Screenshot_Images/{}.jpg".format(img_1_name), 
                                        frame_unedited[
                                            int(min(dieCoordinates[:, 1])):int(max(dieCoordinates[:, 3])), 
                                            int(min(dieCoordinates[:, 0])):int(max(dieCoordinates[:, 2]))
                                            ]
                                        )
                        elif vid_writer_index <= 12:
                            cv2.imwrite("./Images/Screenshot_Images/{}.jpg".format(img_2_name), 
                                        frame_unedited[
                                            int(min(dieCoordinates[:, 1])):int(max(dieCoordinates[:, 3])), 
                                            int(min(dieCoordinates[:, 0])):int(max(dieCoordinates[:, 2]))
                                            ]
                                        )
                        elif vid_writer_index <= 18:
                            cv2.imwrite("./Images/Screenshot_Images/{}.jpg".format(img_3_name), 
                                        frame_unedited[
                                            int(min(dieCoordinates[:, 1])):int(max(dieCoordinates[:, 3])), 
                                            int(min(dieCoordinates[:, 0])):int(max(dieCoordinates[:, 2]))
                                            ]
                                        )
                       
                        if vid_writer_index >= 18:
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
            print("  ", round(tenScale/fps_time_lapsed, 2), "FPS - " + now, 
                  end="\r")
            fps_start_time = time.time()
    else:
        vc = cv2.VideoCapture('rtsp://admin:9562432315@192.168.86.31:554/h264Preview_01_main')
        
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