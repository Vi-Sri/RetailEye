import numpy as np
import tensorflow as tf
import json
import cv2
from pyzbar.pyzbar import decode
import time

# Loading product database
with open('product_database.json', 'r') as f:
    product_database = json.load(f)

# Loading TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="C:\\Users\\zz881\\Documents\\CAP - 6411\\Class Project\\Demo_3_Frieburg\\custom_model_lite\\saved_model\\detect.tflite")
interpreter.allocate_tensors()

# Getting input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Loading class names
with open('C:\\Users\\zz881\\Documents\\CAP - 6411\\Class Project\\Demo_3_Frieburg\\custom_model_lite\\labelmap_new.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Opening the camera
cap = cv2.VideoCapture(0)

def scan_barcode(frame):
    barcodes = decode(frame)
    for barcode in barcodes:
        barcode_data = barcode.data.decode("utf-8")
        barcode_polygon = barcode.polygon
        return barcode_data, np.array(barcode_polygon)
    return None, None

reset_wait_time = 3
last_detection_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    
    if time.time() - last_detection_time < reset_wait_time:
        continue

    frame_resized = cv2.resize(frame, (320, 320))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb / 255.0
    input_data = np.expand_dims(frame_normalized, axis=0).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get the detection scores, classes, and boxes
    detection_scores = interpreter.get_tensor(output_details[0]['index'])[0]  # StatefulPartitionedCall:1 (Detection Scores)
    detection_boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # StatefulPartitionedCall:3 (Bounding Boxes)
    detection_classes = interpreter.get_tensor(output_details[3]['index'])[0]  # StatefulPartitionedCall:2 (Classes)

    for i in range(len(detection_scores)):
        score = detection_scores[i]
        if score > 0.6:
            ymin, xmin, ymax, xmax = detection_boxes[i]
            ymin = int(ymin * h)
            ymax = int(ymax * h)
            xmin = int(xmin * w)
            xmax = int(xmax * w)
            detected_label = class_names[int(detection_classes[i])]

            barcode_data, barcode_polygon = scan_barcode(frame)
            product_info = product_database.get(barcode_data, None)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            text = f'{detected_label}: {score:.2f}'
            cv2.putText(frame, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (173, 216, 230), 2)

            if score > 0.6 and product_info and detected_label == product_info['product_category']:
                print("Product and Label Matched")
                print(f"Price of the item: ${product_info['price']}")
                print(f"Item Scanned: {product_info['name']}")
                cv2.putText(frame, "Valid Transaction", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                last_detection_time = time.time()

    cv2.imshow('Detected Objects and Barcode', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
