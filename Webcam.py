import cv2
from cv2 import VideoWriter
from cv2 import VideoWriter_fourcc
import os
import torch
from torchvision import models
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


# User parameters
SAVE_NAME_OD = "./Models-OD/Animals-0.model"
DATASET_PATH = "./Training_Data/" + SAVE_NAME_OD.split("./Models-OD/",1)[1].split("-",1)[0] +"/"
MIN_SCORE               = 0.7


def time_convert(sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print("Time Lapsed = {0}h:{1}m:{2}s".format(int(hours), int(mins), round(sec) ) )



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
vc = cv2.VideoCapture(0)


if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

# For recording video
video = VideoWriter('webcam.avi', VideoWriter_fourcc(*'MJPG'), 10.0, (int(frame.shape[1]), int(frame.shape[0])) )

transforms_1 = A.Compose([
    # A.Resize(int(frame.shape[0]/2), int(frame.shape[1]/2)),
    ToTensorV2()
])

# Start FPS timer
fps_start_time = time.time()
ii = 0
tenScale = 100

while rval:
    cv2.imshow("preview", frame)
    # cv2.setWindowProperty("preview", cv2.WND_PROP_TOPMOST, 1)
    rval, frame = vc.read()
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transformed_image = transforms_1(image=image)
    transformed_image = transformed_image["image"]
    
    with torch.no_grad():
        prediction_1 = model_1([(transformed_image/255).to(device)])
        pred_1 = prediction_1[0]
    
    dieCoordinates = pred_1['boxes'][pred_1['scores'] > MIN_SCORE]
    die_class_indexes = pred_1['labels'][pred_1['scores'] > MIN_SCORE].tolist()
    # BELOW SHOWS SCORES - COMMENT OUT IF NEEDED
    die_scores = pred_1['scores'][pred_1['scores'] > MIN_SCORE].tolist()
    
    labels_found = [str(round(die_scores[index]*100)) + "% - " + str(classes_1[class_index]) 
                    for index, class_index in enumerate(die_class_indexes)]
    
    predicted_image = draw_bounding_boxes(transformed_image,
        boxes = dieCoordinates,
        # labels = [classes_1[i] for i in die_class_indexes], 
        labels = labels_found, # SHOWS SCORE AND INDEX IN LABEL
        width = 3,
        colors = [color_list[i] for i in die_class_indexes]
        )
    
    # Can comment out - this is for testing
    # save_image((predicted_image/255), "image.jpg")
    
    # Changes image back to a cv2 friendly format
    frame = predicted_image.permute(1,2,0).contiguous().numpy()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Write frame to the video file
    video.write(frame)
    
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    
    ii += 1
    if ii % tenScale == 0:
        fps_end_time = time.time()
        fps_time_lapsed = fps_end_time - fps_start_time
        print("  ", round(tenScale/fps_time_lapsed, 2), "FPS")
        fps_start_time = time.time()


# Release web camera stream
vc.release()
cv2.destroyWindow("preview")

# Release video output file stream
video.release()