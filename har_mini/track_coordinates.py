from ultralytics import YOLO
import cv2

#OUTPUT
#Initial object position (x_center,y_center)
# 0: 384x640 1 person, 25.8ms
# Speed: 0.9ms preprocess, 25.8ms inference, 0.4ms postprocess per image at shape (1, 3, 384, 640)
# ID: 1 | Object: person | Center: (1011, 661)

#Object moving towards right so the x_center coordinate increased
# 0: 384x640 1 person, 28.0ms
# Speed: 0.7ms preprocess, 28.0ms inference, 0.6ms postprocess per image at shape (1, 3, 384, 640)
# ID: 1 | Object: person | Center: (1354, 715)

# Object moved towards left, so the x_center coordinate drastically decreased
# 0: 384x640 1 person, 28.2ms
# Speed: 1.2ms preprocess, 28.2ms inference, 0.4ms postprocess per image at shape (1, 3, 384, 640)
# ID: 1 | Object: person | Center: (649, 703)


model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    results = model.track(frame, persist=True)

    boxes = results[0].boxes

    if boxes.id is not None:
        for box, track_id, class_id in zip(
            boxes.xyxy,
            boxes.id,
            boxes.cls
        ):
            x1, y1, x2, y2 = box

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            class_name = model.names[int(class_id)]

            print(
                f"ID: {int(track_id)} | "
                f"Object: {class_name} | "
                f"Center: ({center_x}, {center_y})"
            )

    annotated_frame = results[0].plot()

    cv2.imshow("SIH Track Coordinates", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()