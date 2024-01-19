# detect-pose
1、dataset：数据集部分
    Convert_yolov8pose.py：包含LSP、FLIC数据集的处理，转化为yolov8-pose可训练的txt格式的数据集
2、yolov8-pose
    官方地址：https://github.com/ultralytics/ultralytics
    直接 pip install ultralytics==8.1.1（我使用的版本）
    train.py、val.py都比较简单
    predict.py：可视化验证的一个demo，需要根据数据集修改skeleton
    
