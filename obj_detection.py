from ultralytics import YOLO
# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs


try:
    results = model.track(source=0, show=True)
    print(results)
except Exception as e:
    print(f"An error occurred during prediction: {e}")