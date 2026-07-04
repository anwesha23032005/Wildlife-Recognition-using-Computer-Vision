from ultralytics import YOLO
from pathlib import Path


model = YOLO("animal_runs/animal_model5/weights/best.pt")


video_path = "vid2.mp4"


results = model.predict(
    source=video_path,
    imgsz=640,
    conf=0.25,
    save=True,                 
    project="animal_runs",
    name="animal_video_results"
)

print("Video inference completed.")
print("Output saved in animal_runs/animal_video_results")