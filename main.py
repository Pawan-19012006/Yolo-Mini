from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("https://ultralytics.com/images/bus.jpg")

result = results[0]

print("Bounding boxes:")
print(result.boxes.xyxy)

print("\nConfidence:")
print(result.boxes.conf)

print("\nClass IDs:")
print(result.boxes.cls)