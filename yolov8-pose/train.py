from ultralytics import YOLO
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model = YOLO('yolov8s-pose.yaml').load('yolov8s-pose.pt')
# Train the model
results = model.train(data=current_dir+'/lsp-pose.yaml',
                      epochs=300, imgsz=224,workers=1,batch=16)
