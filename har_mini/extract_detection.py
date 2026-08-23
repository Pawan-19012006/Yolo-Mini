from ultralytics import YOLO
import cv2

# from this experiment we can identify the extraction of output from the detection

#OUTPUT EXAMPLE

#0: 384x640 1 person, 1 book, 26.8ms
# Speed: 0.8ms preprocess, 26.8ms inference, 0.3ms postprocess per image at shape (1, 3, 384, 640)
# Object: person | Confidence: 0.93 | Box: (533, 331, 1513, 1071)
# Object: book | Confidence: 0.26 | Box: (1538, 95, 1715, 248)

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    results = model(frame)

    boxes = results[0].boxes

    for box, confidence, class_id in zip(
        boxes.xyxy,
        boxes.conf,
        boxes.cls
    ):
        x1, y1, x2, y2 = box

        class_name = model.names[int(class_id)]

        print(
            f"Object: {class_name} | "
            f"Confidence: {confidence:.2f} | " #confidence score of each object detected
            f"Box: ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})" #coordinates of the object that has been detected, it changes from frame to frame becoz of the movement in the object and its detected boundary along with it
        )

    annotated_frame = results[0].plot()

    cv2.imshow("Detection Data", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()