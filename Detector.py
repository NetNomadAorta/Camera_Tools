# TODO turn this into a class. Then use it the same way in main.py

import os
import sys
import torch
import torchvision.transforms
from torchvision import models
import cv2
import json
import argparse
from datetime import datetime
import numpy as np
import time
from typing import Type, Tuple, List


# User parameters
class Detector:
    def __init__(self, configuration):
        self.configuration = configuration
        self.device = "cpu"
        self.model_1 = None
        self.classes_1 = None
        print("started")

    def setup(self):
        f = open(os.path.join(self.configuration.DATASET_PATH, "train", "_annotations.coco.json"))
        data = json.load(f)
        n_classes_1 = len(data['categories'])
        self.classes_1 = [i['name'] for i in data["categories"]]

        if os.path.isfile("results_log.json"):
            f = open("results_log.json")
            self.results_log = json.load(f)
        else:
            print("NO_DATA")
            self.results_log = []

        if os.path.isfile("exclusion_zones_test.json"):
            f = open("exclusion_zones_test.json")
            self.exclusion_zones = json.load(f)
        else:
            print("NO_DATA for exclusion zone!")
            self.exclusion_zones = []

        if "ResNet" in self.configuration.model_name:
            '''
            Load model, find devices available, etc. Should only be called once
            '''

            # lets load the faster rcnn model
            self.model_1 = models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                    min_size=self.configuration.MIN_IMAGE_SIZE,
                                                                    max_size=self.configuration.MIN_IMAGE_SIZE * 2
                                                                    )
            in_features = self.model_1.roi_heads.box_predictor.cls_score.in_features  # we need to change the head
            self.model_1.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features,
                                                                                                  n_classes_1)

            # Loads last saved checkpoint
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            print("MODEL IN USE:" + str(self.configuration.MODEL_IN_USE))
            if os.path.isfile(self.configuration.MODEL_IN_USE):
                checkpoint = torch.load(self.configuration.MODEL_IN_USE, map_location=self.device)
                self.model_1.load_state_dict(checkpoint)
            else:
                print("MODEL NOT FOUND! Maybe typo?")

            self.model_1 = self.model_1.to(self.device)

            self.model_1.eval()
            torch.cuda.empty_cache()

            self.transforms_1 = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        elif self.configuration.model_name == "YOLOv7":
            import numpy as np
            import time
            import sys
            import argparse
            from numpy import random
            from models.experimental import attempt_load
            from utils.datasets import LoadStreams, LoadImages
            from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
                apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
            from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

            self.non_max_suppression = non_max_suppression
            self.time_synchronized = time_synchronized
            self.scale_coords = scale_coords

            parser = argparse.ArgumentParser()
            parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
            parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
            parser.add_argument('--view-img', action='store_true', help='display results')
            parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
            parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
            parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
            parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
            parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
            parser.add_argument('--augment', action='store_true', help='augmented inference')
            parser.add_argument('--update', action='store_true', help='update all models')
            parser.add_argument('--project', default='runs/detect', help='save results to project/name')
            parser.add_argument('--name', default='exp', help='save results to project/name')
            parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
            parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
            self.opt = parser.parse_args()

            set_logging()
            self.device = select_device(self.opt.device)
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            self.model = attempt_load(self.configuration.MODEL_IN_USE, map_location=self.device)  # load FP32 model
            stride = int(self.model.stride.max())  # model stride
            self.imgsz = self.configuration.MIN_IMAGE_SIZE  # check img_size

            if self.half:
                self.model.half()  # to FP16

            # Set Dataloader
            self.dataset = LoadImages("test.jpg", img_size=self.imgsz, stride=stride)

            # Get names and colors
            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # Run inference
            if self.device.type != 'cpu':
                with torch.no_grad():
                    self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                        next(self.model.parameters())))  # run once

    def put_text(self, image, label, start_point, font, fontScale, color, thickness):
        cv2.putText(image, label, start_point, font, fontScale, (0, 0, 0), thickness + 2)
        cv2.putText(image, label, start_point, font, fontScale, color, thickness)

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    # This will check if a vehicle has moved and label it as active
    def check_if_active(self, results_log, coordinate_current, object):
        is_active = True

        if len(results_log) == 0:
            return is_active

        # Sets how much object can move
        if object == "Vehicle":
            fraction_box_needed_to_move = self.configuration.FRACTION_BOX_NEEDED_TO_MOVE
        elif object == "Person":
            fraction_box_needed_to_move = self.configuration.FRACTION_BOX_NEEDED_TO_MOVE_PERSON_ONLY

        for index, data_entries in reversed(list(enumerate(results_log))):
            for data in data_entries:

                # Only a limited number of detections for active people will be examined for comparison purposes
                if object == "Person":
                    if "Vehicle" not in data['label']:
                        if (len(list(results_log)) - index) > 2 * 5:
                            continue

                coordinate_past = data['coordinate']  # Gets coordinates of previous detection's object

                fraction = fraction_box_needed_to_move  # Percentage of change of coordinates needed to claim if object is active or has moved
                width = coordinate_past[2] - coordinate_past[0]
                width_fract = width * fraction
                height = coordinate_past[3] - coordinate_past[1]
                height_fract = height * fraction

                x1_min_current = min(coordinate_current[0], coordinate_current[2])
                y1_min_current = min(coordinate_current[1], coordinate_current[3])
                x2_min_current = max(coordinate_current[0], coordinate_current[2])
                y2_min_current = max(coordinate_current[1], coordinate_current[3])

                # Getting the min and max of coordinates might be redundant as coorinates[0] is normally the top left coordinate
                x1_min_past = min(coordinate_past[0], coordinate_past[2])
                y1_min_past = min(coordinate_past[1], coordinate_past[3])
                x2_min_past = max(coordinate_past[0], coordinate_past[2])
                y2_min_past = max(coordinate_past[1], coordinate_past[3])

                x1_diff = abs(x1_min_current - x1_min_past)
                y1_diff = abs(y1_min_current - y1_min_past)
                x2_diff = abs(x2_min_current - x2_min_past)
                y2_diff = abs(y2_min_current - y2_min_past)

                if x1_diff < width_fract and y1_diff < height_fract and x2_diff < width_fract and y2_diff < height_fract:
                    is_active = False
                    return is_active

        return is_active  # Do I need this line, because I already have return above?

    def dim_polygon_section(self, image, alpha=0.2):

        return image

        if len(self.exclusion_zones) == 0:
            print("dim polygon section \u0394 time: "+str(time.time() - start))
            return image

        polygon_coordinates = self.exclusion_zones["Coordinates"]

        for polygon_coordinate in polygon_coordinates:

            if len(polygon_coordinate) > 2:
                pts = np.array(polygon_coordinate, np.int32)
                pts = pts.reshape((-1, 1, 2))

                overlay = image.copy()
                cv2.fillPoly(overlay, [pts], (0, 0, 255, 255))

                image[:] = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        return image

    # Tests to see if coordinates are within a certain zone. If so, then exclude it.
    def excludes_zone(self, coordinate, exclusion_zones):
        in_excluded_zone = False
        return in_excluded_zone

        polygon_coordinates = exclusion_zones["Coordinates"]

        point_to_check = (int((coordinate[2] + coordinate[0]) / 2), int(coordinate[3]))  # Bottom center of bounding box

        for polygon_coordinate in polygon_coordinates:
            contour = np.array(polygon_coordinate, dtype=np.int32)
            result = cv2.pointPolygonTest(contour, point_to_check, False)
            if result >= 0:
                in_excluded_zone = True

        return in_excluded_zone

    def detect(self, frame_rgb):
        # Setting up for safety and security in their respected methods/function
        self.image_width = frame_rgb.shape[1]
        self.image_height = frame_rgb.shape[0]

        if "ResNet" in self.configuration.model_name:
            # Preprocess the frame
            transformed_image = self.transforms_1(frame_rgb)

            # Pass the transformed frame through the model
            with torch.no_grad():
                prediction_1 = self.model_1([transformed_image.to(self.device)])
                pred_1 = prediction_1[0]

            # Extract the predicted boxes, scores, and class labels
            coordinates = pred_1['boxes'][pred_1['scores'] > self.configuration.MIN_SCORE].cpu().numpy()
            class_indexes = pred_1['labels'][pred_1['scores'] > self.configuration.MIN_SCORE].cpu().numpy()
            scores = pred_1['scores'][pred_1['scores'] > self.configuration.MIN_SCORE].cpu().numpy()

            return coordinates, scores, class_indexes

        elif self.configuration.model_name == "YOLOv7":
            with torch.no_grad():
                frame_image_rgv = frame_rgb

                t0 = time.time()
                # STEPPING THROUGH IMAGES
                img0 = frame_image_rgv  # BGR # INSERT IMAGE HERE!!!
                im0s = img0
                # Padded resize
                img = self.letterbox(img0, self.imgsz, stride=32)[0]

                # Convert
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)

                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                old_img_w = old_img_h = self.imgsz
                old_img_b = 1

                # Warmupdet
                if self.device.type != 'cpu' and (
                        old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                    old_img_b = img.shape[0]
                    old_img_h = img.shape[2]
                    old_img_w = img.shape[3]
                    for i in range(3):
                        self.model(img, augment=self.opt.augment)[0]

                # Inference
                t1 = self.time_synchronized()
                pred = self.model(img, augment=self.opt.augment)[0]
                t2 = self.time_synchronized()
                print("Inference Time:", (t2-t1) )

                # Apply NMS
                pred = self.non_max_suppression(pred, self.configuration.MIN_SCORE, self.opt.iou_thres,
                                                classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
                t3 = self.time_synchronized()

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    s, im0, frame_useless = '', im0s, getattr(self.dataset, 'frame', 0)

                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = self.scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # return det

                    coordinates = []
                    scores = []
                    class_indexes = []
                    for *xyxy, conf, cls in reversed(det):
                        x1 = int(xyxy[0])
                        y1 = int(xyxy[1])
                        x2 = int(xyxy[2])
                        y2 = int(xyxy[3])
                        if conf > self.configuration.MIN_SCORE:
                            coordinates.append([x1, y1, x2, y2])
                            scores.append(conf.item())
                            class_indexes.append(int(cls) + 1)
                            # print([x1, y1, x2, y2], conf.item(), int(cls))

                    return np.array(coordinates), np.array(scores), np.array(class_indexes)

    def safety(self, image, coordinates, scores, class_indexes, is_det_frame=True):
        # Setting up for text and rectangle in cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        thickness = 1

        # Gets minimum object size in pixels
        min_height_vehicle = self.image_height * self.configuration.MIN_OBJ_FRACT_VEHICLE
        min_height_person = self.image_height * self.configuration.MIN_OBJ_FRACT_PERSON

        # Widens bounding box around people by pixel amount
        widens_bbox_by = 5

        # Setting up list info
        coords_person_active = []
        scores_person_active = []
        violations_text = []
        coords_helmet = []
        coords_safety_vest = []
        coords_glasses = []
        coords_no_glasses = []
        coords_person_inactive = []
        scores_person_inactive = []
        coords_vehicle_inactive = []
        scores_vehicle_inactive = []

        if is_det_frame:  # If the detection is being made in this frame, run it. (This false option is if we want to skip frames)

            self.red_labels = []
            self.red_coordinates = []
            self.yellow_coordinates = []
            self.green_coordinates = []
            self.white_coordinates = []

            # Places text and bounding boxes around objects
            for coordinate_index, coordinate in enumerate(coordinates):
                height_coordinate = abs(coordinate[3] - coordinate[1])
                if "Person" in self.classes_1[class_indexes[coordinate_index]]:
                    # if scores[coordinate_index] < 0.70:  # Minimum confidence level Person has to have
                    #     continue

                    in_excluded_zone = self.excludes_zone(coordinate,
                                                          self.exclusion_zones)  # Checks if coordinates are in zone of interest

                    if not in_excluded_zone:
                        # is_active = self.check_if_active(self.results_log, coordinate, "Person")
                        is_active = True

                        # If person is active and reaches minimum height then to work with it
                        if is_active and height_coordinate > min_height_person:
                            coords_person_active.append(coordinate)
                            scores_person_active.append(scores[coordinate_index])
                        else:
                            coords_person_inactive.append(coordinate)
                            scores_person_inactive.append(scores[coordinate_index])

                elif "HighViz" in self.classes_1[class_indexes[coordinate_index]]:
                    coords_safety_vest.append([((coordinate[0] + coordinate[2]) / 2), coordinate[1]])
                elif "Helmet" in self.classes_1[class_indexes[coordinate_index]]:
                    coords_helmet.append([((coordinate[0] + coordinate[2]) / 2), coordinate[3]])
                elif "Glasses" == self.classes_1[class_indexes[coordinate_index]]:
                    coords_glasses.append([((coordinate[0] + coordinate[2]) / 2), coordinate[3]])
                elif "No_Glasses" == self.classes_1[class_indexes[coordinate_index]]:
                    coords_no_glasses.append([((coordinate[0] + coordinate[2]) / 2), coordinate[3]])
                elif "Vehicle" in self.classes_1[class_indexes[coordinate_index]]:
                    in_excluded_zone = self.excludes_zone(coordinate,
                                                          self.exclusion_zones)  # Checks if coordinates are in zone of interest
                    if not in_excluded_zone:
                        coords_vehicle_inactive.append(coordinate)
                        scores_vehicle_inactive.append(scores[coordinate_index])

            # Checks if person has proper PPE
            for index, coord_person in enumerate(coords_person_active):
                label = ""
                violations = []
                has_helmet = False
                has_highVis = False
                has_glasses = False
                has_no_glasses = False

                height_person = abs(coord_person[3] - coord_person[1])

                # Checks if helmet present
                for coord_helmet in coords_helmet:
                    if (coord_person[0] < coord_helmet[0] < coord_person[2]
                            and coord_person[1] < coord_helmet[1] < coord_person[3]
                    ):
                        has_helmet = True

                # Checks if safety_vest present
                for coord_safety_vest in coords_safety_vest:
                    if ((coord_person[0] - height_person * .05) < coord_safety_vest[0] < coord_person[2]
                            and coord_person[1] < coord_safety_vest[1] < coord_person[3]
                    ):
                        has_highVis = True

                # Checks if glasses present
                for coord_glasses in coords_glasses:
                    if (coord_person[0] < coord_glasses[0] < coord_person[2]
                            and coord_person[1] < coord_glasses[1] < coord_person[3]
                    ):
                        has_glasses = True

                # Checks if glasses definitely NOT present
                for coord_no_glasses in coords_no_glasses:
                    if (coord_person[0] < coord_no_glasses[0] < coord_person[2]
                            and coord_person[1] < coord_no_glasses[1] < coord_person[3]
                    ):
                        has_no_glasses = True

                # Safety_Vest labeler
                if not has_highVis:
                    label = label + "Clothing, "
                    violations.append("Clothing")

                # No Glasses Labeler
                if has_no_glasses:
                    label = label + "No Eyewear, "
                    violations.append("No Eyewear")

                # Helmet labeler
                if not has_helmet:
                    label = label + "Head, "
                    violations.append("Head")

                violations_labels = []
                for index, violation in enumerate(violations):
                    self.put_text(image, violation,
                                  (max(int(coord_person[0]) - widens_bbox_by, 0),
                                   max(int(coord_person[1]) - 10 * (index * 2 + 1) + 5 - widens_bbox_by, 0)),
                                  font, fontScale, (0, 0, 255), thickness)
                self.red_labels.append(violations)

                label = label[:-3] + label[-3:].replace(", ", "")
                violations_text.append(label)

                if has_helmet == True and has_highVis == True and has_glasses == True:
                    if self.configuration.SHOWS_INACTIVE_TOGGGLE:
                        cv2.rectangle(image,
                                      (int(coord_person[0]) - widens_bbox_by, int(coord_person[1]) - widens_bbox_by),
                                      (int(coord_person[2]) + widens_bbox_by, int(coord_person[3]) + widens_bbox_by),
                                      (0, 255, 0), thickness)
                    self.green_coordinates.append(coord_person)

                elif has_helmet == False or has_highVis == False or has_no_glasses:
                    cv2.rectangle(image,
                                  (int(coord_person[0]) - widens_bbox_by, int(coord_person[1]) - widens_bbox_by),
                                  (int(coord_person[2]) + widens_bbox_by, int(coord_person[3]) + widens_bbox_by),
                                  (0, 0, 255),
                                  thickness)
                    self.red_coordinates.append(coord_person)

                elif has_helmet == True and has_highVis == True and has_glasses == False:
                    if self.configuration.SHOWS_INACTIVE_TOGGGLE:
                        cv2.rectangle(image,
                                      (int(coord_person[0]) - widens_bbox_by, int(coord_person[1]) - widens_bbox_by),
                                      (int(coord_person[2]) + widens_bbox_by, int(coord_person[3]) + widens_bbox_by),
                                      (0, 255, 255), thickness)
                    self.yellow_coordinates.append(coord_person)

            for index, coord_person in enumerate(coords_person_inactive):
                if self.configuration.SHOWS_INACTIVE_TOGGGLE:
                    cv2.rectangle(image,
                                  (int(coord_person[0]) - widens_bbox_by, int(coord_person[1]) - widens_bbox_by),
                                  (int(coord_person[2]) + widens_bbox_by, int(coord_person[3]) + widens_bbox_by),
                                  (255, 255, 255), thickness)
                self.white_coordinates.append(coord_person)

            # Pops out any person info with no violations
            for index, data_entries in reversed(list(enumerate(coords_person_active))):
                if len(violations_text[index]) == 0:
                    # But first, needs to place in inactive before popping
                    coords_person_inactive.append(coords_person_active[index])
                    scores_person_inactive.append(scores_person_active[index])

                    # Now pop them
                    coords_person_active.pop(index)
                    scores_person_active.pop(index)
                    violations_text.pop(index)

            # Creating JSON section
            # ==================================================================================
            data = []

            for index, coord_person in enumerate(coords_person_active):
                data.append({
                    "coordinate": coords_person_active[index].tolist(),
                    "score": str(scores_person_active[index]),
                    "label": violations_text[index]
                })

            data_w_inactive = data.copy()

            for index, coord_person in enumerate(coords_person_inactive):
                data_w_inactive.append({
                    "coordinate": coords_person_inactive[index].tolist(),
                    # Can change to coord_person to make more efficient.
                    "score": str(scores_person_inactive[index]),
                    "label": "Inactive_Person"
                })

            for index, coord_vehicle in enumerate(coords_vehicle_inactive):
                data_w_inactive.append({
                    "coordinate": coords_vehicle_inactive[index].tolist(),
                    # Can change to coord_vehicle to make more efficient.
                    "score": str(scores_vehicle_inactive[index]),
                    "label": "Inactive_Vehicle"
                })

            max_log_entries = self.configuration.MAX_LOG_ENTRIES

            # Updates the log entries of past detections
            self.results_log.append(data_w_inactive)
            if len(self.results_log) > max_log_entries:
                while len(self.results_log) > max_log_entries:
                    self.results_log.pop(0)
            with open("results_log_test.json", 'w') as f:
                json.dump(self.results_log, f, indent=4)
            # ==================================================================================================================

        else:
            if self.configuration.SHOWS_INACTIVE_TOGGGLE:
                for green_coordinate in self.green_coordinates:
                    cv2.rectangle(image,
                                  (
                                  int(green_coordinate[0]) - widens_bbox_by, int(green_coordinate[1]) - widens_bbox_by),
                                  (
                                  int(green_coordinate[2]) + widens_bbox_by, int(green_coordinate[3]) + widens_bbox_by),
                                  (0, 255, 0), thickness)

            for index_red_coordinate, red_coordinate in enumerate(self.red_coordinates):
                cv2.rectangle(image,
                              (int(red_coordinate[0]) - widens_bbox_by, int(red_coordinate[1]) - widens_bbox_by),
                              (int(red_coordinate[2]) + widens_bbox_by, int(red_coordinate[3]) + widens_bbox_by),
                              (0, 0, 255), thickness)

                for index, violation in enumerate(self.red_labels[index_red_coordinate]):
                    self.put_text(image, violation,
                                  (max(int(red_coordinate[0]) - widens_bbox_by, 0),
                                   max(int(red_coordinate[1]) - 10 * (index * 2 + 1) + 5 - widens_bbox_by, 0)),
                                  font, fontScale, (0, 0, 255), thickness)

            if self.configuration.SHOWS_INACTIVE_TOGGGLE:
                for yellow_coordinate in self.yellow_coordinates:
                    cv2.rectangle(image,
                                  (int(yellow_coordinate[0]) - widens_bbox_by,
                                   int(yellow_coordinate[1]) - widens_bbox_by),
                                  (int(yellow_coordinate[2]) + widens_bbox_by,
                                   int(yellow_coordinate[3]) + widens_bbox_by),
                                  (0, 255, 255), thickness)

            if self.configuration.SHOWS_INACTIVE_TOGGGLE:
                for white_coordinate in self.white_coordinates:
                    cv2.rectangle(image,
                                  (
                                  int(white_coordinate[0]) - widens_bbox_by, int(white_coordinate[1]) - widens_bbox_by),
                                  (
                                  int(white_coordinate[2]) + widens_bbox_by, int(white_coordinate[3]) + widens_bbox_by),
                                  (255, 255, 255), thickness)

        return image

    def security(self, image, coordinates, scores, class_indexes, is_det_frame=True):
        # Setting up for text and rectangle in cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        thickness = 1

        # Gets minimum object size in pixels
        min_height_vehicle = self.image_height * self.configuration.MIN_OBJ_FRACT_VEHICLE
        min_height_person = self.image_height * self.configuration.MIN_OBJ_FRACT_PERSON

        # Widens bounding box around people by pixel amount
        widens_bbox_by = 5

        # Setting up list info
        coords_person_active = []
        scores_person_active = []
        coords_vehicle_active = []
        scores_vehicle_active = []
        coords_person_inactive = []
        scores_person_inactive = []
        coords_vehicle_inactive = []
        scores_vehicle_inactive = []

        if is_det_frame:  # If the detection is being made in this frame, run it. (This false option is if we want to skip frames)

            self.red_labels = []
            self.red_coordinates = []
            self.white_coordinates = []
            classes_found = []

            # Places text and bounding boxes around objects
            for coordinate_index, coordinate in enumerate(coordinates):
                height_coordinate = abs(coordinate[3] - coordinate[1])
                if "Person" in self.classes_1[class_indexes[coordinate_index]]:
                    # if scores[coordinate_index] < 0.70:  # Minimum confidence level Person has to have
                    #     continue

                    in_excluded_zone = self.excludes_zone(coordinate,
                                                          self.exclusion_zones)  # Checks if coordinates are in zone of interest

                    if not in_excluded_zone:
                        is_active = self.check_if_active(self.results_log, coordinate, "Person")

                        # If person is active and reaches minimum height then to work with it
                        if is_active and height_coordinate > min_height_person:
                            classes_found.append("Person")
                            label = "Active Person"
                            self.put_text(image, label,
                                          (int(coordinate[0]), max(int(coordinate[1]) - 5 - widens_bbox_by, 0)),
                                          font, fontScale, (0, 0, 255), thickness)

                            cv2.rectangle(image,
                                          (int(coordinate[0]) - widens_bbox_by,
                                           int(coordinate[1]) - widens_bbox_by),
                                          (int(coordinate[2]) + widens_bbox_by,
                                           int(coordinate[3]) + widens_bbox_by),
                                          (0, 0, 255),
                                          thickness)

                            coords_person_active.append(coordinate)
                            scores_person_active.append(scores[coordinate_index])
                            self.red_labels.append(label)
                            self.red_coordinates.append(coordinate)
                        else:
                            if self.configuration.SHOWS_INACTIVE_TOGGGLE:
                                cv2.rectangle(image,
                                              (int(coordinate[0]) - widens_bbox_by,
                                               int(coordinate[1]) - widens_bbox_by),
                                              (int(coordinate[2]) + widens_bbox_by,
                                               int(coordinate[3]) + widens_bbox_by),
                                              (255, 255, 255),
                                              thickness)

                            coords_person_inactive.append(coordinate)
                            scores_person_inactive.append(scores[coordinate_index])
                            self.white_coordinates.append(coordinate)

                elif "Vehicle" in self.classes_1[class_indexes[coordinate_index]]:
                    in_excluded_zone = self.excludes_zone(coordinate,
                                                          self.exclusion_zones)  # Checks if coordinates are in zone of interest

                    if not in_excluded_zone:
                        is_active = self.check_if_active(self.results_log, coordinate, "Vehicle")

                        # If person is active and reaches minimum height then to work with it
                        if is_active and height_coordinate > min_height_vehicle:
                            classes_found.append("Vehicle")

                            label = "Active Vehicle"
                            self.put_text(image, label,
                                          (int(coordinate[0]), max(int(coordinate[1]) - 5, 0)),
                                          font, fontScale, (0, 0, 255), thickness)

                            cv2.rectangle(image,
                                          (int(coordinate[0]), int(coordinate[1])),
                                          (int(coordinate[2]), int(coordinate[3])),
                                          (0, 0, 255),
                                          thickness)

                            coords_vehicle_active.append(coordinate)
                            scores_vehicle_active.append(scores[coordinate_index])
                            self.red_labels.append(label)
                            self.red_coordinates.append(coordinate)
                        else:
                            if self.configuration.SHOWS_INACTIVE_TOGGGLE:
                                cv2.rectangle(image,
                                              (int(coordinate[0]), int(coordinate[1])),
                                              (int(coordinate[2]), int(coordinate[3])),
                                              (255, 255, 255),
                                              thickness)

                            coords_vehicle_inactive.append(coordinate)
                            scores_vehicle_inactive.append(scores[coordinate_index])
                            self.white_coordinates.append(coordinate)

            # Creating JSON section
            # ==================================================================================
            data = []

            for index, coord_person in enumerate(coords_person_active):
                data.append({
                    "coordinate": coords_person_active[index].tolist(),
                    "score": str(scores_person_active[index]),
                    "label": "Active_Person"
                })

            for index, coord_vehicle in enumerate(coords_vehicle_active):
                data.append({
                    "coordinate": coords_vehicle_active[index].tolist(),
                    "score": str(scores_vehicle_active[index]),
                    "label": "Active_Vehicle"
                })

            data_w_inactive = data.copy()

            for index, coord_person in enumerate(coords_person_inactive):
                data_w_inactive.append({
                    "coordinate": coords_person_inactive[index].tolist(),
                    # Can change to coord_person to make more efficient.
                    "score": str(scores_person_inactive[index]),
                    "label": "Inactive_Person"
                })

            for index, coord_vehicle in enumerate(coords_vehicle_inactive):
                data_w_inactive.append({
                    "coordinate": coords_vehicle_inactive[index].tolist(),
                    # Can change to coord_vehicle to make more efficient.
                    "score": str(scores_vehicle_inactive[index]),
                    "label": "Inactive_Vehicle"
                })

            max_log_entries = self.configuration.MAX_LOG_ENTRIES

            # Updates the log entries of past detections
            self.results_log.append(data_w_inactive)
            if len(self.results_log) > max_log_entries:
                while len(self.results_log) > max_log_entries:
                    self.results_log.pop(0)
            with open("results_log_test.json", 'w') as f:
                json.dump(self.results_log, f, indent=4)
            # ==================================================================================================================

        else:
            for index_red_coordinate, red_coordinate in enumerate(self.red_coordinates):
                cv2.rectangle(image,
                              (int(red_coordinate[0]) - widens_bbox_by, int(red_coordinate[1]) - widens_bbox_by),
                              (int(red_coordinate[2]) + widens_bbox_by, int(red_coordinate[3]) + widens_bbox_by),
                              (0, 0, 255), thickness)

                for index, violation in enumerate(self.red_labels[index_red_coordinate]):
                    self.put_text(image, violation,
                                  (max(int(red_coordinate[0]) - widens_bbox_by, 0),
                                   max(int(red_coordinate[1]) - 10 * (index * 2 + 1) + 5 - widens_bbox_by, 0)),
                                  font, fontScale, (0, 0, 255), thickness)

            if self.configuration.SHOWS_INACTIVE_TOGGGLE:
                for white_coordinate in self.white_coordinates:
                    cv2.rectangle(image,
                                  (
                                  int(white_coordinate[0]) - widens_bbox_by, int(white_coordinate[1]) - widens_bbox_by),
                                  (
                                  int(white_coordinate[2]) + widens_bbox_by, int(white_coordinate[3]) + widens_bbox_by),
                                  (255, 255, 255), thickness)

        return image, classes_found

    def combined(self, image, coordinates, scores, class_indexes, is_det_frame=True):
        # Setting up for text and rectangle in cv2
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        thickness = 1

        # Gets minimum object size in pixels
        min_height_vehicle = self.image_height * self.configuration.MIN_OBJ_FRACT_VEHICLE
        min_height_person = self.image_height * self.configuration.MIN_OBJ_FRACT_PERSON

        # Widens bounding box around people by pixel amount
        widens_bbox_by_inner = 5
        widens_bbox_by_outer = 10

        # Setting up list info
        coords_person_active = []
        scores_person_active = []
        coords_person_all = []
        violations_text = []
        coords_helmet = []
        coords_safety_vest = []
        coords_glasses = []
        coords_no_glasses = []
        coords_vehicle_active = []
        scores_vehicle_active = []
        coords_person_inactive = []
        scores_person_inactive = []
        coords_vehicle_inactive = []
        scores_vehicle_inactive = []

        if is_det_frame:  # If the detection is being made in this frame, run it. (This false option is if we want to skip frames)

            self.red_labels = []
            self.red_coordinates = []
            self.yellow_coordinates = []
            self.green_coordinates = []
            self.white_coordinates = []

            # Places text and bounding boxes around objects
            for coordinate_index, coordinate in enumerate(coordinates):
                height_coordinate = abs(coordinate[3] - coordinate[1])
                if "Person" in self.classes_1[class_indexes[coordinate_index]]:
                    # if scores[coordinate_index] < 0.70:  # Minimum confidence level Person has to have
                    #     continue

                    in_excluded_zone = self.excludes_zone(coordinate,
                                                          self.exclusion_zones)  # Checks if coordinates are in zone of interest

                    if not in_excluded_zone:
                        is_active = self.check_if_active(self.results_log, coordinate, "Person")

                        # If person is active and reaches minimum height then to work with it
                        if is_active and height_coordinate > min_height_person:
                            label = "AP"
                            self.put_text(image, label,
                                          (int(coordinate[0]), max(int(coordinate[3]) + 5 + widens_bbox_by_outer, 0)),
                                          font, fontScale, (0, 0, 255), thickness)

                            cv2.rectangle(image,
                                          (int(coordinate[0]) - widens_bbox_by_inner,
                                           int(coordinate[1]) - widens_bbox_by_inner),
                                          (int(coordinate[2]) + widens_bbox_by_inner,
                                           int(coordinate[3]) + widens_bbox_by_inner),
                                          (0, 0, 255),
                                          thickness)

                            coords_person_active.append(coordinate)
                            scores_person_active.append(scores[coordinate_index])
                            coords_person_all.append(coordinate)
                        else:
                            cv2.rectangle(image,
                                          (int(coordinate[0]) - widens_bbox_by_inner,
                                           int(coordinate[1]) - widens_bbox_by_inner),
                                          (int(coordinate[2]) + widens_bbox_by_inner,
                                           int(coordinate[3]) + widens_bbox_by_inner),
                                          (255, 255, 255),
                                          thickness)

                            coords_person_inactive.append(coordinate)
                            scores_person_inactive.append(scores[coordinate_index])
                            coords_person_all.append(coordinate)

                elif "HighViz" in self.classes_1[class_indexes[coordinate_index]]:
                    coords_safety_vest.append([((coordinate[0] + coordinate[2]) / 2), coordinate[1]])
                elif "Helmet" in self.classes_1[class_indexes[coordinate_index]]:
                    coords_helmet.append([((coordinate[0] + coordinate[2]) / 2), coordinate[3]])
                elif "Glasses" == self.classes_1[class_indexes[coordinate_index]]:
                    coords_glasses.append([((coordinate[0] + coordinate[2]) / 2), coordinate[3]])
                elif "No_Glasses" == self.classes_1[class_indexes[coordinate_index]]:
                    coords_no_glasses.append([((coordinate[0] + coordinate[2]) / 2), coordinate[3]])
                elif "Vehicle" in self.classes_1[class_indexes[coordinate_index]]:
                    in_excluded_zone = self.excludes_zone(coordinate,
                                                          self.exclusion_zones)  # Checks if coordinates are in zone of interest

                    if not in_excluded_zone:
                        is_active = self.check_if_active(self.results_log, coordinate, "Vehicle")

                        # If person is active and reaches minimum height then to work with it
                        if is_active and height_coordinate > min_height_vehicle:
                            label = "Active Vehicle"
                            self.put_text(image, label,
                                          (int(coordinate[0]), max(int(coordinate[1]) - 5, 0)),
                                          font, fontScale, (0, 0, 255), thickness)

                            cv2.rectangle(image,
                                          (int(coordinate[0]) - widens_bbox_by_inner,
                                           int(coordinate[1]) - widens_bbox_by_inner),
                                          (int(coordinate[2]) + widens_bbox_by_inner,
                                           int(coordinate[3]) + widens_bbox_by_inner),
                                          (0, 0, 255),
                                          thickness)

                            coords_vehicle_active.append(coordinate)
                            scores_vehicle_active.append(scores[coordinate_index])
                            self.red_labels.append(label)
                            self.red_coordinates.append(coordinate)
                        else:
                            if self.configuration.SHOWS_INACTIVE_TOGGGLE:
                                cv2.rectangle(image,
                                              (int(coordinate[0]) - widens_bbox_by_inner,
                                               int(coordinate[1]) - widens_bbox_by_inner),
                                              (int(coordinate[2]) + widens_bbox_by_inner,
                                               int(coordinate[3]) + widens_bbox_by_inner),
                                              (255, 255, 255),
                                              thickness)

                            coords_vehicle_inactive.append(coordinate)
                            scores_vehicle_inactive.append(scores[coordinate_index])
                            self.white_coordinates.append(coordinate)

            # Checks if person has proper PPE
            for index, coord_person in enumerate(coords_person_all):
                label = ""
                violations = []
                has_helmet = False
                has_highVis = False
                has_glasses = False
                has_no_glasses = False

                height_person = abs(coord_person[3] - coord_person[1])

                # Checks if helmet present
                for coord_helmet in coords_helmet:
                    if (coord_person[0] < coord_helmet[0] < coord_person[2]
                            and coord_person[1] < coord_helmet[1] < coord_person[3]
                    ):
                        has_helmet = True

                # Checks if safety_vest present
                for coord_safety_vest in coords_safety_vest:
                    if ((coord_person[0] - height_person * .05) < coord_safety_vest[0] < coord_person[2]
                            and coord_person[1] < coord_safety_vest[1] < coord_person[3]
                    ):
                        has_highVis = True

                # Checks if glasses present
                for coord_glasses in coords_glasses:
                    if (coord_person[0] < coord_glasses[0] < coord_person[2]
                            and coord_person[1] < coord_glasses[1] < coord_person[3]
                    ):
                        has_glasses = True

                # Checks if glasses definitely NOT present
                for coord_no_glasses in coords_no_glasses:
                    if (coord_person[0] < coord_no_glasses[0] < coord_person[2]
                            and coord_person[1] < coord_no_glasses[1] < coord_person[3]
                    ):
                        has_no_glasses = True

                # Safety_Vest labeler
                if not has_highVis:
                    label = label + "Clothing, "
                    violations.append("Clothing")

                # No Glasses Labeler
                if has_no_glasses:
                    label = label + "No Eyewear, "
                    violations.append("No Eyewear")

                # Helmet labeler
                if not has_helmet:
                    label = label + "Head, "
                    violations.append("Head")

                violations_labels = []
                for index, violation in enumerate(violations):
                    self.put_text(image, violation,
                                  (max(int(coord_person[0]) - widens_bbox_by_outer, 0),
                                   max(int(coord_person[1]) - 10 * (index * 2 + 1) + 5 - widens_bbox_by_outer, 0)),
                                  font, fontScale, (0, 0, 255), thickness)
                self.red_labels.append(violations)

                label = label[:-3] + label[-3:].replace(", ", "")
                violations_text.append(label)

                if has_helmet == True and has_highVis == True and has_glasses == True:
                    if self.configuration.SHOWS_INACTIVE_TOGGGLE:
                        cv2.rectangle(image,
                                      (int(coord_person[0]) - widens_bbox_by_outer,
                                       int(coord_person[1]) - widens_bbox_by_outer),
                                      (int(coord_person[2]) + widens_bbox_by_outer,
                                       int(coord_person[3]) + widens_bbox_by_outer), (0, 255, 0),
                                      thickness)
                    self.green_coordinates.append(coord_person)

                elif has_helmet == False or has_highVis == False or has_no_glasses:
                    cv2.rectangle(image,
                                  (int(coord_person[0]) - widens_bbox_by_outer,
                                   int(coord_person[1]) - widens_bbox_by_outer),
                                  (int(coord_person[2]) + widens_bbox_by_outer,
                                   int(coord_person[3]) + widens_bbox_by_outer), (0, 0, 255),
                                  thickness)
                    self.red_coordinates.append(coord_person)

                elif has_helmet == True and has_highVis == True and has_glasses == False:
                    if self.configuration.SHOWS_INACTIVE_TOGGGLE:
                        cv2.rectangle(image,
                                      (int(coord_person[0]) - widens_bbox_by_outer,
                                       int(coord_person[1]) - widens_bbox_by_outer),
                                      (int(coord_person[2]) + widens_bbox_by_outer,
                                       int(coord_person[3]) + widens_bbox_by_outer), (0, 255, 255),
                                      thickness)
                    self.yellow_coordinates.append(coord_person)

            for index, coord_person in enumerate(coords_person_inactive):
                self.white_coordinates.append(coord_person)

            # Pops out any person info with no violations
            # for index, data_entries in reversed(list(enumerate(coords_person_active))):
            #     if len(violations_text[index]) == 0:
            #         # But first, needs to place in inactive before popping
            #         coords_person_inactive.append(coords_person_active[index])
            #         scores_person_inactive.append(scores_person_active[index])
            #
            #         # Now pop them
            #         coords_person_active.pop(index)
            #         scores_person_active.pop(index)
            #         violations_text.pop(index)

            # Creating JSON section
            # ==================================================================================
            data = []

            for index, coord_person in enumerate(coords_person_active):
                data.append({
                    "coordinate": coords_person_active[index].tolist(),
                    "score": str(scores_person_active[index]),
                    "label": violations_text[index]
                })

            for index, coord_vehicle in enumerate(coords_vehicle_active):
                data.append({
                    "coordinate": coords_vehicle_active[index].tolist(),
                    "score": str(scores_vehicle_active[index]),
                    "label": "Active_Vehicle"
                })

            data_w_inactive = data.copy()

            for index, coord_person in enumerate(coords_person_inactive):
                data_w_inactive.append({
                    "coordinate": coords_person_inactive[index].tolist(),
                    # Can change to coord_person to make more efficient.
                    "score": str(scores_person_inactive[index]),
                    "label": "Inactive_Person"
                })

            for index, coord_vehicle in enumerate(coords_vehicle_inactive):
                data_w_inactive.append({
                    "coordinate": coords_vehicle_inactive[index].tolist(),
                    # Can change to coord_vehicle to make more efficient.
                    "score": str(scores_vehicle_inactive[index]),
                    "label": "Inactive_Vehicle"
                })

            max_log_entries = self.configuration.MAX_LOG_ENTRIES

            # Updates the log entries of past detections
            self.results_log.append(data_w_inactive)
            if len(self.results_log) > max_log_entries:
                while len(self.results_log) > max_log_entries:
                    self.results_log.pop(0)
            with open("results_log_test.json", 'w') as f:
                json.dump(self.results_log, f, indent=4)
            # ==================================================================================================================

        else:
            if self.configuration.SHOWS_INACTIVE_TOGGGLE:
                for green_coordinate in self.green_coordinates:
                    cv2.rectangle(image,
                                  (int(green_coordinate[0]) - widens_bbox_by_outer,
                                   int(green_coordinate[1]) - widens_bbox_by_outer),
                                  (int(green_coordinate[2]) + widens_bbox_by_outer,
                                   int(green_coordinate[3]) + widens_bbox_by_outer), (0, 255, 0),
                                  thickness)

            for index_red_coordinate, red_coordinate in enumerate(self.red_coordinates):
                cv2.rectangle(image,
                              (int(red_coordinate[0]) - widens_bbox_by_outer,
                               int(red_coordinate[1]) - widens_bbox_by_outer),
                              (int(red_coordinate[2]) + widens_bbox_by_outer,
                               int(red_coordinate[3]) + widens_bbox_by_outer), (0, 0, 255),
                              thickness)

                for index, violation in enumerate(self.red_labels[index_red_coordinate]):
                    self.put_text(image, violation,
                                  (max(int(red_coordinate[0]) - widens_bbox_by_outer, 0),
                                   max(int(red_coordinate[1]) - 10 * (index * 2 + 1) + 5 - widens_bbox_by_outer, 0)),
                                  font, fontScale, (0, 0, 255), thickness)

            if self.configuration.SHOWS_INACTIVE_TOGGGLE:
                for yellow_coordinate in self.yellow_coordinates:
                    cv2.rectangle(image,
                                  (int(yellow_coordinate[0]) - widens_bbox_by_outer,
                                   int(yellow_coordinate[1]) - widens_bbox_by_outer),
                                  (int(yellow_coordinate[2]) + widens_bbox_by_outer,
                                   int(yellow_coordinate[3]) + widens_bbox_by_outer), (0, 255, 255),
                                  thickness)

            if self.configuration.SHOWS_INACTIVE_TOGGGLE:
                for white_coordinate in self.white_coordinates:
                    cv2.rectangle(image,
                                  (int(white_coordinate[0]) - widens_bbox_by_outer,
                                   int(white_coordinate[1]) - widens_bbox_by_outer),
                                  (int(white_coordinate[2]) + widens_bbox_by_outer,
                                   int(white_coordinate[3]) + widens_bbox_by_outer), (255, 255, 255),
                                  thickness)

        return image

    def draw_line(self, event, x, y, flags, param):
        # global drawing, ix, iy, points

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.points = [(self.ix, self.iy)]

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # cv2.line(self.img, (self.ix, self.iy), (x, y), (0, 0, 255), 2)
                self.ix, self.iy = x, y
                self.points.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            # cv2.line(self.img, (self.ix, self.iy), (x, y), (0, 0, 255), 2)
            self.points.append((x, y))
            self.points_list.append(self.points)
            # self.fill_area(self.points)

    def fill_area(self, points, alpha=0.2):
        if len(points) > 2:
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))

            overlay = self.img.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 255, 255))

            self.img[:] = cv2.addWeighted(overlay, alpha, self.img, 1 - alpha, 0)

    def exclusion_zone_setup(self, frame):
        self.data_exclusion = {}  # Blank dictionary for json file

        # global drawing, points, points_list
        self.drawing = False
        self.ixix, self.ixiy = -1, -1
        self.points = []
        self.points_list = []

        # global img
        self.img = frame
        cv2.namedWindow('Frame')
        cv2.setMouseCallback('Frame', self.draw_line)

    def exclusion_zone_runner(self, frame):
        self.img = frame
        for points in self.points_list:
            self.fill_area(points)

            for index, point in enumerate(points):
                if index == 0:
                    prev_point = point
                    continue
                cv2.line(self.img, prev_point, point, (0, 0, 255), 2)
                prev_point = point

        for index, point in enumerate(self.points):
            if index == 0:
                prev_point = point
                continue
            cv2.line(self.img, prev_point, point, (0, 0, 255), 2)
            prev_point = point
        return self.img

    def exclusion_zone_end(self):
        self.data_exclusion.update({"Coordinates": self.points_list})

        cv2.destroyAllWindows()

        file_path = "exclusion_zones_test.json"
        with open(file_path, 'w') as file:
            json.dump(self.data_exclusion, file, indent=4)

        self.exclusion_zones = self.data_exclusion


# process single image passed as argument
if __name__ == "__main__":
    # print("args:"+str(args.image))
    args.image = "test2.jpeg"
    # cv2.imwrite(args.image, "test2.jpeg")
    image = cv2.imread("test2.jpeg")

    if os.path.isfile("results_log.json"):
        f = open("results_log.json")
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

    # jsonpath = args.image.replace(".jpeg", "_result.json")
    imgpath = args.image.replace(".jpeg", "_result.jpg")
    Detector = SafetyDetector()
    # data, image_out, data_w_inactive, image_unannotated = detect(image, results_log, imgpath, exclusion_zones)

    # with open(jsonpath, 'w') as f:
    #     json.dump(data, f, indent=4)
    #
    max_log_entries = 40  # TIME ON JETSON IS CONFUZED! THIS IS TEMP SOLN. DELEEEEETE LATEEEERRR ONNNNNNN

    # Updates the log entries of past detections
    results_log.append(data_w_inactive)
    if len(results_log) > max_log_entries:
        while len(results_log) > max_log_entries:
            results_log.pop(0)
    with open("results_log.json", 'w') as f:
        json.dump(results_log, f, indent=4)

    # print("jsonpath:"+jsonpath)

    cv2.imwrite(imgpath, image_out)

    # JUST TEMPORARY - DELETE
    if len(data) > 0:
        if "MULTI" in imgpath:
            now = datetime.now()
            now = now.strftime("%Y_%m_%d---%H_%M_%S")
            cam_index = imgpath.split("MULTI_")[1].split("/")[0]
            cv2.imwrite("./Saved_Images/cam_{}-{}.jpg".format(cam_index, now), image_unannotated)

    print("imgpath:" + imgpath)