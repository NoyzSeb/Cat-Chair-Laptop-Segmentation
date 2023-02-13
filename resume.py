from multiprocessing import freeze_support
from ultralytics import YOLO
import torch

my_device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')



model = YOLO("runs/detect/yolov8l_GSData_V1/weights/last.pt")

model.to(my_device)

if (__name__ == '__main__'):
    freeze_support()
    model.train(resume=True)