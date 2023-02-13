from multiprocessing import freeze_support

from ultralytics import YOLO
import wandb
import torch




my_device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')

model = YOLO("runs/segment/SEG_yolov8l_SEGData_V22/weights/last.pt")

model.to(my_device)


if (__name__ == '__main__'):
    freeze_support()
    model.train(data="../datasets/cat-chair-laptop_Seg_V2/data.yaml", epochs=50,batch=10, name="SEG_yolov8l_SEGData_V2",verbose=True, patience=0,save=True, pretrained=True,workers=12,resume=True)
    model.val()  # It'll automatically evaluate the data you traine
    

