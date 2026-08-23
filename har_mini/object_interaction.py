from ultralytics import YOLO
import cv2

#OUTPUT
# 0: 384x640 1 person, 1 bottle, 23.7ms
# Speed: 0.7ms preprocess, 23.7ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
# INTERACTION: person ↔ bottle

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)


def is_inside(inner_box, outer_box):
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box

    center_x = (ix1 + ix2) / 2
    center_y = (iy1 + iy2) / 2

    return (
        ox1 <= center_x <= ox2
        and
        oy1 <= center_y <= oy2
    )


while True:
    success, frame = cap.read()

    if not success:
        break

    results = model.track(frame, persist=True)

    boxes = results[0].boxes

    persons = []
    objects = [] # we seperate the person detection boundary and the object detection boundary, since we need to check whether the person and object are interacting with each other

    # If the boundary of the object is inside the boundary of person, then they are interacting with each other, but this is not the correct interaction find, coz we still dont know whether the object is picked up or not by the person, for that we need to perform temporal detection

    if boxes.id is not None:

        for box, track_id, class_id in zip(
            boxes.xyxy,
            boxes.id,
            boxes.cls
        ):
            box = box.tolist()
            class_name = model.names[int(class_id)]

            if class_name == "person":
                persons.append(box)

            else:
                objects.append((class_name, box))

    for person_box in persons:

        for object_name, object_box in objects:

            if is_inside(object_box, person_box):

                print(
                    f"INTERACTION: person ↔ {object_name}"
                )

    annotated_frame = results[0].plot()

    cv2.imshow("Object Interaction", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()