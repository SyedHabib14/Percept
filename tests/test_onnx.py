import cv2
import time
from ultralytics import YOLO

model = YOLO(r"D:\PDCv5\models\dpose_engine.onnx", task="pose")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    start = time.perf_counter()
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.5, verbose=False)
    annotated = results[0].plot()

    fps = 1 / (time.perf_counter() - start)

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLO Pose ONNX", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()