from ultralytics import YOLO
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image
import io

app = Flask(__name__)

model = YOLO("animal_runs/animal_model5/weights/best.pt")

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Image not provided"}), 400

    file = request.files["image"]

    
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    results = model.predict(
        source=image,
        imgsz=640,
        conf=0.25,
        verbose=False
    )

    detections = []

    for r in results:
        for box in r.boxes:
            detections.append({
                "object": model.names[int(box.cls)],
                "confidence": round(float(box.conf), 4)
            })

    return jsonify({
        "detections": detections,
        "count": len(detections)
    })

if __name__ == "__main__":
    app.run(debug=True)