import cv2

camera_id = 1
delay = 1
window_name = 'camerascreen'

bd = cv2.barcode.BarcodeDetector("./opencv_3rdparty/detect.prototxt", "./opencv_3rdparty/detect.caffemodel")
cap = cv2.VideoCapture(camera_id)
import numpy as np


focus_step_value = 5

def setFocus(focus_value):
    global cap
    focus_value *= focus_step_value
    if focus_value%5==0:
        cap.set(28, focus_value) 
    else:
        print(f"focus value should be multiple of 5, found {focus_value}")

cv2.namedWindow(window_name)
cv2.createTrackbar('Focus', window_name, 0, 100, setFocus)

focus = 0  # min: 0, max: 255, increment:5
cap.set(28, focus) 

ret, frame = cap.read()
# print(frame.shape)
while True:
    ret, frame = cap.read()


    if ret:
        frame = cv2.resize(frame, (1080,720))
        ret_bc, decoded_info, points, _ = bd.detectAndDecodeMulti(frame)

        if ret_bc:
            # print(points)
            frame = cv2.polylines(frame, points.astype(int), True, (0, 255, 0), 3)
            for s, p in zip(decoded_info, points):
                if s:
                    # print(s)
                    frame = cv2.putText(frame, s, p[1].astype(int),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cv2.destroyWindow(window_name)