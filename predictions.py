from ultralytics import YOLO
import torch
from IPython.display import display, Image

my_device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')

model = YOLO("../runs/segment/SEG_yolov8l_SEGData_V12Best_V1/weights/last.pt")

model.to(my_device)

results = model.predict(source="datasets/cat-laptop-chair_V1/test/images",conf=0.7,save=True, name="p3_SEG_yolov8l_SEGData_V12Best_V1")


