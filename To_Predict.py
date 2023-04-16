import os
import torch
from torchvision import models
import math
import re
import cv2
import albumentations as A  # our data augmentation library
# remove arnings (optional)
import warnings
warnings.filterwarnings("ignore")
import time
from torchvision.utils import draw_bounding_boxes
from pycocotools.coco import COCO
# Now, we will define our transforms
from albumentations.pytorch import ToTensorV2

from torchvision.utils import save_image
import shutil


# User parameters
SAVE_NAME_OD = "./Models-OD/Animals-0.model"
DATASET_PATH = "./Training_Data/" + SAVE_NAME_OD.split("./Models-OD/",1)[1].split("-",1)[0] +"/"
IMAGE_SIZE              = int(re.findall(r'\d+', SAVE_NAME_OD)[-1] ) # Row and column number 
TO_PREDICT_PATH         = "./Images/Prediction_Images/To_Predict/"
# TO_PREDICT_PATH         = "//mcrtp-sftp-01/aoitool/SMiPE4-623/XDCC000109C2/"            # USE FOR XDisplay LOTS!
PREDICTED_PATH          = "./Images/Prediction_Images/Predicted_Images/"
# PREDICTED_PATH          = "//mcrtp-sftp-01/aoitool/SMiPE4-623-Cropped/XDCC000109C2/"    # USE FOR XDisplay LOTS!
# PREDICTED_PATH        = "C:/Users/troya/.spyder-py3/ML-Defect_Detection/Images/Prediction_Images/To_Predict_Images/"
SAVE_ANNOTATED_IMAGES   = True
SAVE_ORIGINAL_IMAGE     = False
SAVE_CROPPED_IMAGES     = False
DIE_SPACING_SCALE       = 0.99
MIN_SCORE               = 0.7 # Default 0.5


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )


def deleteDirContents(dir):
    # Deletes photos in path "dir"
    # # Used for deleting previous cropped photos from last run
    for f in os.listdir(dir):
        full_path = os.path.join(dir, f)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)


# Creates class folder
def makeDir(dir, classes_2):
    for classIndex, className in enumerate(classes_2):
        os.makedirs(dir + className, exist_ok=True)



# Starting stopwatch to see how long process takes
start_time = time.time()

# Deletes images already in "Predicted_Images" folder
deleteDirContents(PREDICTED_PATH)

dataset_path = DATASET_PATH



#load classes
coco = COCO(os.path.join(dataset_path, "train", "_annotations.coco.json"))
categories = coco.cats
n_classes_1 = len(categories.keys())
categories

classes_1 = [i[1]['name'] for i in categories.items()]



# lets load the faster rcnn model
model_1 = models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=500)
in_features = model_1.roi_heads.box_predictor.cls_score.in_features # we need to change the head
model_1.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, n_classes_1)


# Loads last saved checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

if os.path.isfile(SAVE_NAME_OD):
    checkpoint = torch.load(SAVE_NAME_OD, map_location=map_location)
    model_1.load_state_dict(checkpoint)

model_1 = model_1.to(device)

model_1.eval()
torch.cuda.empty_cache()

transforms_1 = A.Compose([
    # A.Resize(IMAGE_SIZE, IMAGE_SIZE), # our input size can be 600px
    # A.Rotate(limit=[90,90], always_apply=True),
    ToTensorV2()
])


# Start FPS timer
fps_start_time = time.time()

color_list =['green', 'red', 'blue', 'magenta', 'orange', 'cyan', 'lime', 'turquoise', 'yellow']
pred_dict = {}
ii = 0
for image_name in os.listdir(TO_PREDICT_PATH):
    image_path = os.path.join(TO_PREDICT_PATH, image_name)
    
    image_b4_color = cv2.imread(image_path)
    orig_image = image_b4_color
    image = cv2.cvtColor(image_b4_color, cv2.COLOR_BGR2RGB)
    
    transformed_image = transforms_1(image=image)
    transformed_image = transformed_image["image"]
    
    if ii == 0:
        line_width = max(round(transformed_image.shape[1] * 0.002), 1)
    
    with torch.no_grad():
        prediction_1 = model_1([(transformed_image/255).to(device)])
        pred_1 = prediction_1[0]
    
    dieCoordinates = pred_1['boxes'][pred_1['scores'] > MIN_SCORE]
    die_class_indexes = pred_1['labels'][pred_1['scores'] > MIN_SCORE]
    # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
    die_scores = pred_1['scores'][pred_1['scores'] > MIN_SCORE]
    
    labels_found = [str(int(die_scores[index]*100)) + "% - " + str(classes_1[class_index]) 
                    for index, class_index in enumerate(die_class_indexes)]
            
    
    if SAVE_ANNOTATED_IMAGES:
        predicted_image = draw_bounding_boxes(transformed_image,
            boxes = dieCoordinates,
            # labels = [classes_1[i] for i in die_class_indexes], 
            # labels = [str(round(i,2)) for i in die_scores], # SHOWS SCORE IN LABEL
            width = line_width,
            colors = [color_list[i] for i in die_class_indexes],
            font = "arial.ttf",
            font_size = 10
            )
        
        predicted_image_cv2 = predicted_image.permute(1,2,0).contiguous().numpy()
        predicted_image_cv2 = cv2.cvtColor(predicted_image_cv2, cv2.COLOR_RGB2BGR)
        
        for dieCoordinate_index, dieCoordinate in enumerate(dieCoordinates):
            start_point = ( int(dieCoordinate[0]), int(dieCoordinate[1]) )
            # end_point = ( int(dieCoordinate[2]), int(dieCoordinate[3]) )
            color = (255, 255, 255)
            # thickness = 3
            # cv2.rectangle(predicted_image_cv2, start_point, end_point, color, thickness)
            
            start_point_text = (start_point[0], max(start_point[1]-5,0) )
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            thickness = 1
            cv2.putText(predicted_image_cv2, labels_found[dieCoordinate_index], 
                        start_point_text, font, fontScale, color, thickness)
        
        # Saves full image with bounding boxes
        if len(die_class_indexes) != 0:
            cv2.imwrite(PREDICTED_PATH + image_name, predicted_image_cv2)
            # save_image((predicted_image/255), PREDICTED_PATH + image_name)
        
        # save_image((predicted_image/255), PREDICTED_PATH + image_name)
        
    if SAVE_ORIGINAL_IMAGE and len(die_class_indexes) != 0:
        cv2.imwrite(PREDICTED_PATH + image_name.replace(".jpg","") + "-Original.jpg", orig_image)
    
    
    if len(os.listdir(TO_PREDICT_PATH)) > 2000:
        tenScale = 1000
    elif len(os.listdir(TO_PREDICT_PATH)) > 1000:
        tenScale = 500
    else:
        tenScale = 100

    ii += 1
    if ii % tenScale == 0:
        fps_end_time = time.time()
        fps_time_lapsed = fps_end_time - fps_start_time
        print("  " + str(ii) + " of " 
              + str(len(os.listdir(TO_PREDICT_PATH))), 
              "-",  round(tenScale/fps_time_lapsed, 2), "FPS")
        fps_start_time = time.time()


print("Done!")

# Stopping stopwatch to see how long process takes
end_time = time.time()
time_lapsed = end_time - start_time
time_convert(time_lapsed)