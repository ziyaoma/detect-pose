from ultralytics import YOLO
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
#model = YOLO(current_dir+'/coco-pose.yaml').load(current_dir+'/yolov8s-pose.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data=current_dir+'/coco-pose.yaml',
                      epochs=100, imgsz=224,workers=1,batch=1)
