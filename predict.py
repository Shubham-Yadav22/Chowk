from ultralytics import YOLO

model = YOLO("yolo_custom.pt")


model.predict(source = "video.mp4",show = True,save = True,conf=0.6)