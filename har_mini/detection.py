from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt") #loading the model

cap = cv2.VideoCapture(0) #opening the camera

while True:
    success, frame = cap.read()

    if not success:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("SIH Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()