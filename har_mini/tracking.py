from ultralytics import YOLO
import cv2

# OUTPUT
# 0: 384x640 1 person, 2 bottles, 26.8ms
# Speed: 0.8ms preprocess, 26.8ms inference, 0.4ms postprocess per image at shape (1, 3, 384, 640)
# ID: 1 | Object: person | Confidence: 0.90 | Box: (497, 237, 1632, 1072)
# ID: 2 | Object: bottle | Confidence: 0.90 | Box: (1266, 236, 1599, 909)
# ID: 4 | Object: bottle | Confidence: 0.29 | Box: (151, 332, 404, 1070)

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    results = model.track(frame, persist=True) # tracking is different from detection, because, detection shows that there is an object that has been found from the dataset, whereas tracking will identify whether the same object is been detected in every frame

    boxes = results[0].boxes

    #tracking id is different from class id, even if the class id is same, say two bottles appear in the image, then the tracking id will be different, because it will be unique for every object to keep track of each and every objecgt throughout across every frames

    if boxes.id is not None:
        for box, track_id, class_id, confidence in zip(
            boxes.xyxy,
            boxes.id,
            boxes.cls,
            boxes.conf
        ):
            x1, y1, x2, y2 = box

            class_name = model.names[int(class_id)]

            print(
                f"ID: {int(track_id)} | "
                f"Object: {class_name} | "
                f"Confidence: {confidence:.2f} | "
                f"Box: ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
            )

    annotated_frame = results[0].plot()

    cv2.imshow("Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()