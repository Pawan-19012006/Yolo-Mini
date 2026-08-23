from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read() #read the frames one by one from the camera - it is used through opencv

    if not success:
        print("Failed to read frame")
        break

    results = model(frame) #run the inference on that frame to detect the object - yolo

    annotated_frame = results[0].plot() #draw the results lively on the video itself

    cv2.imshow("YOLO Camera", annotated_frame) # display that image along with the resultant(boundary drawn)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()