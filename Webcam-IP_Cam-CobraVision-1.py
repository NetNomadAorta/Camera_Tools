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


# User parameters
# SAVE_NAME_OD = "./Models-OD/Animals-0.model"
SAVE_NAME_OD = r"C:\Users\troya\.spyder-py3\periscope\Models\Vehicle_PPL_PPE.model".replace("\\", "/")
# DATASET_PATH = "./Training_Data/" + SAVE_NAME_OD.split("./Models-OD/",1)[1].split("-",1)[0] +"/"
DATASET_PATH = r"C:\Users\troya\.spyder-py3\periscope".replace("\\", "/") + "/Training_Data/" + SAVE_NAME_OD.split("/Models/", 1)[1].split(".model", 1)[0] + "/"
MIN_IMAGE_SIZE          = int(1080*2.0) # Minimum size of image (ASPECT RATIO IS KEPT SO DONT WORRY). So for 1600x2400 -> 800x1200
MIN_SCORE    = 0.70
MIN_OBJ_FRACT_VEHICLE = 0.025 # Default: 0.02
MIN_OBJ_FRACT_PERSON = 0.027 # Default: 0.014
FRACTION_BOX_NEEDED_TO_MOVE = 0.50
FRACTION_BOX_NEEDED_TO_MOVE_PERSON_ONLY = 0.5
DIM_EXCLUSION_ZONE = True
SET_VIDEO_RECORD_FORCE = False
SET_VIDEO_RECORD_ON_OD = True
EMAILER_TOGGLE = True
IMAGE_SCALER = 2.7


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )


def put_text(image, label, start_point, font, fontScale, color, thickness):
    cv2.putText(image, label, start_point, font, fontScale, (0, 0, 0), thickness + 2)
    cv2.putText(image, label, start_point, font, fontScale, color, thickness)


# This will check if a vehicle has moved and label it as active
# def check_if_active(results_log, coordinate_current, image_path):
def check_if_active(results_log, coordinate_current):
    is_active = True

    if len(results_log) == 0:
        return is_active

    for index, data_entries in reversed(list(enumerate(results_log))):
        for data in data_entries:
            # if "MULTI" in image_path:
            #     cam_index = image_path.split("MULTI_")[1].split("/")[0]
            #     cam_name = "MULTI_{}".format(cam_index)
            #     if cam_name not in data['camera_name'] and "Unknown_Camera" not in data['camera_name']:
            #         continue

            if "Vehicle" not in data['label']:
                if (len(list(results_log)) - index) <= 4 * 15:
                    fraction_box_needed_to_move = FRACTION_BOX_NEEDED_TO_MOVE_PERSON_ONLY
                else:
                    continue
            else:
                fraction_box_needed_to_move = FRACTION_BOX_NEEDED_TO_MOVE

            coordinate_past = data['coordinate']  # Gets coordinates of previous detection's object

            fraction = fraction_box_needed_to_move  # Percentage of change of coordinates needed to claim if object is active or has moved
            width = coordinate_past[2] - coordinate_past[0]
            width_fract = width * fraction
            height = coordinate_past[3] - coordinate_past[1]
            height_fract = height * fraction

            # x1_min_current = min(coordinate_current[0], coordinate_current[2])
            # y1_min_current = min(coordinate_current[1], coordinate_current[3])
            # x2_min_current = max(coordinate_current[0], coordinate_current[2])
            # y2_min_current = max(coordinate_current[1], coordinate_current[3])
            #
            # # Getting the min and max of coordinates might be redundant as coorinates[0] is normally the top left coordinate
            # x1_min_past = min(coordinate_past[0], coordinate_past[2])
            # y1_min_past = min(coordinate_past[1], coordinate_past[3])
            # x2_min_past = max(coordinate_past[0], coordinate_past[2])
            # y2_min_past = max(coordinate_past[1], coordinate_past[3])

            # x1_diff = abs(x1_min_current - x1_min_past)
            # y1_diff = abs(y1_min_current - y1_min_past)
            # x2_diff = abs(x2_min_current - x2_min_past)
            # y2_diff = abs(y2_min_current - y2_min_past)

            x1_diff = abs(coordinate_current[0] - coordinate_past[0])
            y1_diff = abs(coordinate_current[1] - coordinate_past[1])
            x2_diff = abs(coordinate_current[2] - coordinate_past[2])
            y2_diff = abs(coordinate_current[3] - coordinate_past[3])

            if x1_diff < width_fract and y1_diff < height_fract and x2_diff < width_fract and y2_diff < height_fract:
                is_active = False
                return is_active

    return is_active  # Do I need this line, because I already have return above?


def dim_polygon_section(image_path, image, exclusion_zones, alpha=0.4):
    cam_index = "Placeholder"
    in_excluded_zone = False

    # if "MULTI" in image_path:
    #     cam_index = image_path.split("MULTI_")[1].split("/")[0]
    #
    #     polygon_coordinates = exclusion_zones["MULTI_{}".format(cam_index)]["Security"]

    cam_index = image_path.split("MULTI_")[1].split("/")[0]

    polygon_coordinates = exclusion_zones["MULTI_{}".format(cam_index)]["Security"]

    for polygon_coordinate in polygon_coordinates:

        if len(polygon_coordinate) > 2:
            pts = np.array(polygon_coordinate, np.int32)
            pts = pts.reshape((-1, 1, 2))

            overlay = image.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 0, 255))

            image[:] = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return image


# Tests to see if coordinates are within a certain zone. If so, then exclude it.
def excludes_zone(image_path, coordinate, exclusion_zones):
    cam_index = "Placeholder"
    in_excluded_zone = False

    if "MULTI" in image_path:
        cam_index = image_path.split("MULTI_")[1].split("/")[0]

        polygon_coordinates = exclusion_zones["MULTI_{}".format(cam_index)]["Security"]

        point_to_check = (((coordinate[2] + coordinate[0]) / 2), coordinate[3])  # Bottom center of bounding box

        for polygon_coordinate in polygon_coordinates:
            contour = np.array(polygon_coordinate, dtype=np.int32)
            result = cv2.pointPolygonTest(contour, point_to_check, False)
            if result >= 0:
                in_excluded_zone = True

    return in_excluded_zone



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

if SET_VIDEO_RECORD_ON_OD or EMAILER_TOGGLE:
    #load classes
    f = open(os.path.join(dataset_path, "train", "_annotations.coco.json"))
    # f = open(dataset_path + "train/_annotations.coco.json")
    data = json.load(f)
    n_classes_1 = len(data['categories'])
    classes_1 = [i['name'] for i in data["categories"]]


    # lets load the faster rcnn model
    model_1 = models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                       min_size=MIN_IMAGE_SIZE,
                                                       max_size=MIN_IMAGE_SIZE*2
                                                       )
    in_features = model_1.roi_heads.box_predictor.cls_score.in_features # we need to change the head
    model_1.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes_1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU to train

    # TESTING TO LOAD MODEL
    if os.path.isfile(SAVE_NAME_OD):
        checkpoint = torch.load(SAVE_NAME_OD, map_location=device)
        model_1.load_state_dict(checkpoint)



    model_1 = model_1.to(device)

    model_1.eval()
    torch.cuda.empty_cache()

    color_list = ['green', 'magenta', 'turquoise', 'red', 'green', 'orange', 'yellow', 'cyan', 'lime']



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

transforms_1 = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

if os.path.isfile("results_log-1.json"):
    f = open("results_log-1.json")
    results_log = json.load(f)
else:
    print("NO_DATA")
    results_log = []

if os.path.isfile("exclusion_zones.json"):
    f = open("exclusion_zones.json")
    exclusion_zones = json.load(f)
else:
    print("NO_DATA for exclusion zone!")
    exclusion_zones = []


# Start FPS timer
fps_start_time = time.time()
ii = 0
video_stopper_delayer_index = 0
tenScale = 100
currently_recording = False
frames_to_pred = 20

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
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                height = image.shape[0]
                width = image.shape[1]
                min_height_vehicle = height * MIN_OBJ_FRACT_VEHICLE
                min_width_vehicle = width * MIN_OBJ_FRACT_VEHICLE
                min_height_person = height * MIN_OBJ_FRACT_PERSON

                transformed_image = transforms_1(image)

                with torch.no_grad():
                    prediction_1 = model_1([(transformed_image/255).to(device)])
                    pred_1 = prediction_1[0]

                coordinates_tensor = pred_1['boxes'][pred_1['scores'] > MIN_SCORE]
                coordinates = coordinates_tensor.clone().detach().cpu().numpy().tolist()
                class_indexes = pred_1['labels'][pred_1['scores'] > MIN_SCORE].cpu().numpy().tolist()
                # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
                scores = pred_1['scores'][pred_1['scores'] > MIN_SCORE].cpu().numpy().tolist()

                labels_found = [str(int(scores[index] * 100)) + "% - " + str(classes_1[class_index])
                                for index, class_index in enumerate(class_indexes)]
                classes_found = [str(classes_1[class_index])
                                 for index, class_index in enumerate(class_indexes)]

                predicted_image_cv2 = transformed_image.permute(1, 2, 0).contiguous().numpy()
                predicted_image_cv2 = cv2.cvtColor(predicted_image_cv2, cv2.COLOR_RGB2BGR)

                coords_person = []
                scores_person = []
                texts_person = []
                coords_helmet = []
                coords_safety_vest = []
                coords_glasses = []
                coords_vehicle = []
                scores_vehicle = []
                coords_person_inactive = []
                scores_person_inactive = []
                coords_vehicle_inactive = []
                scores_vehicle_inactive = []
                texts_vehicle = []
                # Places text and bounding boxes around objects
                for coordinate_index, coordinate in enumerate(coordinates):
                    height_coordinate = abs(coordinate[3] - coordinate[1])
                    width_coordinate = abs(coordinate[2] - coordinate[0])

                    start_point = (int(coordinate[0]), int(coordinate[1]))
                    end_point = (int(coordinate[2]), int(coordinate[3]))
                    color = (255, 255, 255)
                    # thickness = 3
                    # cv2.rectangle(predicted_image_cv2, start_point, end_point, color, thickness)

                    start_point_text = (start_point[0], max(start_point[1] - 5, 0))
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 0.5
                    thickness = 1
                    # put_text(image, label, start_point, font, fontScale, color, thickness)
                    # put_text(predicted_image_cv2, labels_found[coordinate_index],
                    #          start_point_text, font, fontScale, color, thickness)

                    if "Person" in classes_1[class_indexes[coordinate_index]]:

                        if scores[coordinate_index] < 0.75:  # Minimum confidence level Person has to have
                            continue

                        color_box = (0, 0, 255)

                        if height_coordinate > min_height_person:
                            # in_excluded_zone = excludes_zone(image_path, coordinate,
                            #                                  exclusion_zones)  # Checks if coordinates are in zone of interest
                            in_excluded_zone = False

                            if not in_excluded_zone:

                                # is_active = check_if_active(results_log, coordinate, image_path)
                                is_active = check_if_active(results_log, coordinate)

                                if is_active:

                                    if height_coordinate > min_height_person:
                                        coords_person.append(coordinate)
                                        scores_person.append(scores[coordinate_index])

                                        label = "Active Person"

                                        widen_by = 5
                                        put_text(predicted_image_cv2, label,
                                                 (int(coordinate[0]), max(int(coordinate[1]) - 5 - widen_by, 0)),
                                                 font, fontScale, (0, 0, 255), thickness)

                                        widen_by = 5
                                        cv2.rectangle(predicted_image_cv2,
                                                      (int(coordinate[0]) - widen_by, int(coordinate[1]) - widen_by),
                                                      (int(coordinate[2]) + widen_by, int(coordinate[3]) + widen_by),
                                                      color_box, thickness)
                                else:
                                    color_box = (255, 255, 255)
                                    coords_person_inactive.append(coordinate)
                                    scores_person_inactive.append(scores[coordinate_index])

                    elif "Vehicle" in classes_1[class_indexes[coordinate_index]]:
                        color_box = (0, 0, 255)

                        if height_coordinate > min_height_vehicle:
                            # in_excluded_zone = excludes_zone(image_path, coordinate,
                            #                                  exclusion_zones)  # Checks if coordinates are in zone of interest
                            in_excluded_zone = False


                            if not in_excluded_zone:

                                # is_active = check_if_active(results_log, coordinate, image_path)
                                is_active = check_if_active(results_log, coordinate)

                                if is_active:

                                    coords_vehicle.append(coordinate)
                                    scores_vehicle.append(scores[coordinate_index])
                                    label = "Active Vehicle"

                                    put_text(predicted_image_cv2, label,
                                             (int(coordinate[0]), max(int(coordinate[1]) - 5, 0)),
                                             font, fontScale, (0, 0, 255), thickness)

                                    cv2.rectangle(predicted_image_cv2, (int(coordinate[0]), int(coordinate[1])),
                                                  (int(coordinate[2]), int(coordinate[3])), color_box, thickness)
                                else:
                                    color_box = (255, 255, 255)
                                    coords_vehicle_inactive.append(coordinate)
                                    scores_vehicle_inactive.append(scores[coordinate_index])


                    else:
                        color_box = (255, 0, 255)
                    # cv2.rectangle(predicted_image_cv2, start_point, end_point, color_box, thickness)

                output_image = predicted_image_cv2 * 255



                # Changes image back to a cv2 friendly format
                # frame = predicted_image.permute(1,2,0).contiguous().numpy()
                # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # frame = output_image
                

                data = []
                classes_found = []

                # if "MULTI" in image_path:
                #     cam_index = image_path.split("MULTI_")[1].split("/")[0]
                #     cam_name = "MULTI_{}".format(cam_index)
                # else:
                #     cam_name = "Unknown_Camera"
                cam_name = "Unknown_Camera"

                for index, coord_person in enumerate(coords_person):
                    data.append({
                        "coordinate": coords_person[index],
                        "score": scores_person[index],
                        "label": "Active_Person",
                        "camera_name": cam_name
                    })

                    classes_found.append("Person")

                for index, coord_vehicle in enumerate(coords_vehicle):
                    data.append({
                        "coordinate": coords_vehicle[index],  # Can change to coord_vehicle to make more efficient.
                        "score": scores_vehicle[index],
                        "label": "Active_Vehicle",
                        "camera_name": cam_name
                    })

                    classes_found.append("Vehicle")

                data_w_inactive = data.copy()

                for index, coord_person in enumerate(coords_person_inactive):
                    data_w_inactive.append({
                        "coordinate": coords_person_inactive[index],
                        # Can change to coord_person to make more efficient.
                        "score": scores_person_inactive[index],
                        "label": "Inactive_Person",
                        "camera_name": cam_name
                    })

                for index, coord_vehicle in enumerate(coords_vehicle_inactive):
                    data_w_inactive.append({
                        "coordinate": coords_vehicle_inactive[index],
                        # Can change to coord_vehicle to make more efficient.
                        "score": scores_vehicle_inactive[index],
                        "label": "Inactive_Vehicle",
                        "camera_name": cam_name
                    })

                max_log_entries = 60 * 2  # TIME ON JETSON IS CONFUZED! THIS IS TEMP SOLN. DELEEEEETE LATEEEERRR ONNNNNNN

                # Updates the log entries of past detections
                results_log.append(data_w_inactive)
                if len(results_log) > max_log_entries:
                    while len(results_log) > max_log_entries:
                        results_log.pop(0)
                with open("results_log-1.json", 'w') as f:
                    json.dump(results_log, f, indent=4)
            # -------------------------------------------------------------------------
            else:
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
                    cv2.putText(frame, labels_found[coordinate_index],
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
                                                    int(min(coordinates_tensor[:, 1])):int(max(coordinates_tensor[:, 3])),
                                                    int(min(coordinates_tensor[:, 0])):int(max(coordinates_tensor[:, 2]))
                                                    ]
                                                )
                                elif email_frame_index <= 2:
                                    cv2.imwrite("./Images/Screenshot_Images/{}.jpg".format(img_2_name),
                                                frame_unedited[
                                                    int(min(coordinates_tensor[:, 1])):int(max(coordinates_tensor[:, 3])),
                                                    int(min(coordinates_tensor[:, 0])):int(max(coordinates_tensor[:, 2]))
                                                    ]
                                                )
                                elif email_frame_index <= 3:
                                    cv2.imwrite("./Images/Screenshot_Images/{}.jpg".format(img_3_name),
                                                frame_unedited[
                                                    int(min(coordinates_tensor[:, 1])):int(max(coordinates_tensor[:, 3])),
                                                    int(min(coordinates_tensor[:, 0])):int(max(coordinates_tensor[:, 2]))
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