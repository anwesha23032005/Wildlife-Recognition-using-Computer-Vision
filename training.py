from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=50, imgsz=640, save_period=1, project="animal_runs", name="animal_model")