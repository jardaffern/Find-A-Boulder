from ultralytics import YOLO
import os

if __name__ == '__main__':

    TRAIN = True
    VALID = False

    DATA_LOCATION = 'C:/Users/jlomb/Documents/PersonalProjects/MPExtensions/modeling/yolo/'
    #os.system('yolo mode=checks')
    os.system(f'yolo task=detect mode=train model=yolov8m.pt data={DATA_LOCATION}/data.yaml epochs=100 imgsz=640')

    #model = YOLO("yolov8n.pt")
    #

#see https://docs.ultralytics.com/modes/train/
# and https://www.youtube.com/watch?v=LNwODJXcvt4