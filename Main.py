from utils.BarcodDetection import BarcodDetection
from utils.OpenCVCameraHandler import OpenCVCameraHandler
from utils.ObjectDetector import ObjectDetector
from utils.BarcodeSwitchDetector import BarcodeSwitchDetector
from utils.Config import Config

import cv2
import os
import uuid

Save_Dir = "output"
os.makedirs(Save_Dir,exist_ok=True)

if __name__=="__main__":

    bd = BarcodDetection(prototxt=Config.BARCODE_PROTOTYPE_FILE, caffemodel=Config.BARCODE_CAFFE_MODEL_FILE)
    barcode_switch_detector = BarcodeSwitchDetector(db_file_path=Config.BARCODE_DB_FILE)
    obj_det = ObjectDetector(model_path=Config.OBJECT_DETECTION_MODEL_FILE, labels_path=Config.OBJECT_DETECTION_LABEL_FILE)

    cv_cap = OpenCVCameraHandler(camera_id=Config.CAMERA_ID)
    cv_cap.setFocus(focus_value=Config.CAMERA_FOCUS)

    while True:

        ret, frame = cv_cap.getFrame()

        if ret:
            # barcode_number,draw_image = bd.detect(frame)
            barcode_number, barcode_polygon = bd.scan_barcode(frame)
            draw_image = frame

            if barcode_number is not None:

                """
                Save barcode detected product image
                """
                product_data = None
                product_name = None
                try:
                    product_data = barcode_switch_detector.product_database.get(barcode_number, None)
                except KeyError:
                    pass
                
                if product_data is not None:
                    product_category = product_data['product_category']
                    product_price = product_data['price']
                    product_name = product_data['name']
                
                if product_name is not None:
                    barcode_number = product_name
                
                image_file_name = f"{Save_Dir}/{barcode_number}_{str(uuid.uuid4())[:5]}.jpg"

                draw_image = cv2.polylines(frame, [barcode_polygon], True, (0, 255, 0), 2)

                cv2.imwrite(image_file_name, frame)

            # detected_boxes, detected_labels, confidences = obj_det.detect_object(input=frame,confidence_threshold=0.6)

            # for i in range(len(detected_boxes)):
            #     xmin, ymin, xmax, ymax = detected_boxes[i]
            #     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)


            # IS_TICKET_SWITCH = barcode_switch_detector.detect_barcode_switch(detected_labels, confidences, barcode_number=barcode_number,take_top_n=3)
            # if IS_TICKET_SWITCH:
            #     # TODO Raise Alert
            #     print("Ticket switch")
            #     pass


            
            
            cv2.imshow(cv_cap.window_name, draw_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break