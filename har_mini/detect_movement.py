from ultralytics import YOLO
import cv2

#OUTPUT
# 0: 384x640 1 person, 26.3ms
# Speed: 0.7ms preprocess, 26.3ms inference, 0.3ms postprocess per image at shape (1, 3, 384, 640)
# ID: 1 | Object: person | STATIONARY | Movement: 5.0px

# 0: 384x640 1 person, 25.8ms
# Speed: 0.8ms preprocess, 25.8ms inference, 0.4ms postprocess per image at shape (1, 3, 384, 640)
# ID: 1 | Object: person | MOVING | Movement: 7.1px

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

previous_positions = {} #we need to keep track of the previous positions in order to track whether the object is moving or not

#also we do the movement detection using the track id, so that we can measure whether the same object is moving or different object is moving

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

            track_id = int(track_id)
            class_name = model.names[int(class_id)]

            current_position = (center_x, center_y)

            if track_id in previous_positions:
                previous_x, previous_y = previous_positions[track_id]

                distance = (
                    (center_x - previous_x) ** 2
                    + (center_y - previous_y) ** 2
                ) ** 0.5

                if distance > 5:
                    status = "MOVING"
                else:
                    status = "STATIONARY"

                print(
                    f"ID: {track_id} | "
                    f"Object: {class_name} | "
                    f"{status} | "
                    f"Movement: {distance:.1f}px"
                )

            previous_positions[track_id] = current_position

    annotated_frame = results[0].plot()

    cv2.imshow("Movement Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()