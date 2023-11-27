from utils.BarcodDetection import BarcodDetection
from utils.OpenCVCameraHandler import OpenCVCameraHandler
from utils.ObjectDetector import ObjectDetector
from utils.BarcodeSwitchDetector import BarcodeSwitchDetector
from utils.Config import Config

import cv2

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
            # barcode_number, barcode_polygon = bd.scan_barcode(frame)

            # if barcode_number is not None:

            barcode_number = "232142543463535"

            detected_boxes, detected_labels, confidences = obj_det.detect_object(input=frame,confidence_threshold=0.6)

            for i in range(len(detected_boxes)):
                xmin, ymin, xmax, ymax = detected_boxes[i]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)


            IS_TICKET_SWITCH = barcode_switch_detector.detect_barcode_switch(detected_labels, confidences, barcode_number=barcode_number,take_top_n=3)
            if IS_TICKET_SWITCH:
                # TODO Raise Alert
                print("Ticket switch")
                pass
            
            cv2.imshow(cv_cap.window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break