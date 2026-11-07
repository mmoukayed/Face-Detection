import cv2
from picamera2 import Picamera2
import os
import time

face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))

cam = Picamera2()
cam.start()

print("Camera running. Press 'esc' to quit.")
prev_time = time.perf_counter()
while True:
    # Capture a frame as a NumPy array
    frame = cam.capture_array()
    # Convert RGB (Picamera2 default) to BGR (OpenCV default)
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(bgr_frame, (x,y),(x+w,y+h),(255,0,0),2)
    
    current_time = time.perf_counter()
    delta_time = current_time - prev_time
    prev_time = current_time
    
    # Avoid DivisionByZero if frame processing is instantaneous
    fps = 1 / delta_time if delta_time > 0 else 0
    
    # Overlay FPS on the video stream
    cv2.putText(bgr_frame, f"FPS: {int(fps)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Face Detection", bgr_frame)
    if(cv2.waitKey(1) & 0xFF == 27):
        break

cam.stop()
cv2.destroyAllWindows()