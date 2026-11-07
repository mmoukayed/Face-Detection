import cv2
import time
from picamera2 import Picamera2

MODEL = "face_detection_yunet_2023mar.onnx"

detector = cv2.FaceDetectorYN.create( MODEL, "", (320, 240), 0.8, 0.3, 5000 )

cam = Picamera2()
config = cam.create_still_configuration(lores={"size": (320, 240)}, display="lores")
cam.align_configuration(config)
cam.start()

while True:
    start = time.perf_counter()
    raw_frame = cam.capture_array()
    frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
    small = cv2.resize(frame, (320, 240))
    detector.setInputSize((320, 240))
    _, faces = detector.detect(small)

    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)

            x *= 2
            y *= 2
            w *= 2
            h *= 2

            cv2.rectangle( frame, (x, y), (x + w, y + h), (0, 255, 0), 2 )

    fps = 1 / (time.perf_counter() - start)

    cv2.putText( frame, f"FPS: {fps:.0f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2 )

    cv2.imshow("YuNet Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.stop()
cv2.destroyAllWindows()