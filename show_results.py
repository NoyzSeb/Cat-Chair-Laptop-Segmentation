import glob
from IPython.display import Image, display

for image_path in glob.glob('C://Users/mberk/Codes/Python/YoloV8/ultralytics/ultralytics/runs/detect/predict3/*.jpg')[:3]:
      display(Image(filename=image_path, width=600))
      print("\n")