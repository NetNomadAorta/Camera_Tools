import json
import os


class DetectionConfiguration:
    '''
    Helper class that allows us to easily get the detection configuration settings for our database anywhere in the app.
    TODO rename periscope_pytest -> detection_scripts or something
    '''

    def __init__(self, path="configurations/detectconfiguration.json", model_name="YOLOv7"):
        # self.PTZ_IP = 'rtsp://admin:Orbital227@192.168.24.151:3001/profile2/media.smp'  # Local IP to PTZ camera
        # self.SCALAR = 1
        # self.PTZ_IP    = 'rtsp://admin:Orbital227@166.163.152.216:3001/profile2/media.smp'
        # self.SCALAR      = 0.4
        self.PTZ_IP = 0
        self.SCALAR = 2

        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        self.file_path = os.path.join(current_directory, path)
        with open(self.file_path, 'r') as config_file:
            json_config = json.load(config_file)

        self.model_name = model_name

        self.WEIGHTS = json_config[self.model_name]["WEIGHTS"]
        self.MODEL_IN_USE = self.WEIGHTS
        self.DATASET_PATH = "./Training_Data/" + \
                            json_config["ResNet50"]["WEIGHTS"].split("./Modelo/", 1)[1].split(".model", 1)[0] + "/"
        self.MIN_IMAGE_SIZE = int(json_config[self.model_name][
                                      "MIN_IMAGE_SIZE"])  # Minimum size of image (ASPECT RATIO IS KEPT SO DONT WORRY).
        self.MIN_SCORE = float(json_config[self.model_name]["MIN_SCORE"])
        self.MIN_OBJ_FRACT_VEHICLE = float(json_config[self.model_name]["MIN_OBJ_FRACT_VEHICLE"])
        self.MIN_OBJ_FRACT_PERSON = float(json_config[self.model_name]["MIN_OBJ_FRACT_PERSON"])
        self.FRACTION_BOX_NEEDED_TO_MOVE = float(json_config[self.model_name]["FRACTION_BOX_NEEDED_TO_MOVE"])
        self.FRACTION_BOX_NEEDED_TO_MOVE_PERSON_ONLY = float(
            json_config[self.model_name]["FRACTION_BOX_NEEDED_TO_MOVE_PERSON_ONLY"])
        self.SHOWS_INACTIVE_TOGGGLE = True if (
                    json_config[self.model_name]["SHOWS_INACTIVE_TOGGGLE"] == "True") else False
        self.IMSHOW_TOGGLE = True if (json_config[self.model_name]["IMSHOW_TOGGLE"] == "True") else False
        self.MAX_LOG_ENTRIES = int(json_config[self.model_name]["MAX_LOG_ENTRIES"])

    def printSettings(self):
        '''
        Prints key detection configuration settings to the terminal
        '''
        print(f"Save Name \'OD\': {self.MODEL_IN_USE}")
        print(f"Dataset Path: {self.DATASET_PATH}")
        print(f"Min. Image Size: {str(self.MIN_IMAGE_SIZE)}")
        print(f"Min Score: {str(self.MIN_SCORE)}")
        print(f"Min Obj. Fract. Vehicle: {str(self.MIN_OBJ_FRACT_VEHICLE)}")
        print(f"Min Obj. Fract. Person: {str(self.MIN_OBJ_FRACT_PERSON)}")
        print(f"Fraction Box Needed to Move: {str(self.FRACTION_BOX_NEEDED_TO_MOVE)}")
        print(f"Fraction Box Needed to Move Person Only: {str(self.FRACTION_BOX_NEEDED_TO_MOVE_PERSON_ONLY)}")
        print(f"Shows inactive objects: {self.SHOWS_INACTIVE_TOGGGLE}")
        print(f"Shows frame in GUI: {self.IMSHOW_TOGGLE}")


if (__name__ == "__main__"):
    DBConn = DBConfiguration()
    DBConn.printSettings()
    DetectConn = DetectionConfiguration()
    DetectConn.printSettings()
