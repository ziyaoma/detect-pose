from ultralytics import YOLO
import os

model = YOLO('weights/best.pt')
metrics = model.val(data='lsp-pose10.yaml',imgsz=224)  # no arguments needed, dataset and settings remembered
# metrics.box.map    # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps   # a list contains map50-95 of each category


