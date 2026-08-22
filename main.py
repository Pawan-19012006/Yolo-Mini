from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("https://ultralytics.com/images/bus.jpg")

result = results[0]

annotated_image = result.plot()

from PIL import Image

Image.fromarray(annotated_image).save("detected.jpg")

print("Saved detection image as detected.jpg")