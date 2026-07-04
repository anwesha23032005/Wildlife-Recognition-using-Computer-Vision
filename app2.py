from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import os
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)

model = YOLO("animal_runs/animal_model5/weights/best.pt")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route("/detect-multiple", methods=["POST"])
def detect_multiple():
    if "images" not in request.files:
        return jsonify({"error": "No images provided"}), 400

    files = request.files.getlist("images")

    if len(files) == 0:
        return jsonify({"error": "Empty image list"}), 400

    summary_counts = defaultdict(int)
    image_results = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx, file in enumerate(files):
        try:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
        except Exception:
            return jsonify({"error": f"Invalid image file: {file.filename}"}), 400

        results = model.predict(
            source=image,
            imgsz=640,
            conf=0.25,
            verbose=False
        )

        detections = []

        for r in results:
            for box in r.boxes:
                class_name = model.names[int(box.cls)]
                confidence = round(float(box.conf), 4)

                summary_counts[class_name] += 1

                detections.append({
                    "object": class_name,
                    "confidence": confidence,
                    "bbox": [round(x, 2) for x in box.xyxy[0].tolist()]
                })

            
            output_path = os.path.join(
                OUTPUT_DIR,
                f"{timestamp}_image_{idx + 1}.jpg"
            )
            r.save(filename=output_path)

        image_results.append({
            "image_name": file.filename,
            "detections": detections,
            "saved_as": output_path
        })

    return jsonify({
        "timestamp": timestamp,
        "total_images": len(files),
        "summary": dict(summary_counts),
        "results": image_results
    })

if __name__ == "__main__":
    app.run(debug=True)