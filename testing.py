from ultralytics import YOLO
from pathlib import Path


model = YOLO("animal_runs/animal_model5/weights/best.pt")


test_dir = Path("test")


results = model.predict(
    source=test_dir,
    imgsz=640,
    conf=0.25, 
    # ues to tell the model if the confident score is less than 0.25 , dont detect the object means dont draw the rectangle.
    save=True,        
    save_txt=True,    
    project="animal_runs",
    name="animal_test_results"
)

print("Inference completed. Results saved in animal_runs/animal_test_results")