from utils.BarcodDetection import BarcodDetection
from utils.OpenCVCameraHandler import OpenCVCameraHandler
from utils.ObjectDetector import ObjectDetector
from utils.BarcodeSwitchDetector import BarcodeSwitchDetector
from utils.Config import Config
from utils.ClassifyLabel import ClassifyLabel
from utils.StateMachine import FSM, STATES
from flask import Flask, render_template, Response, make_response, jsonify
from flask import request
import zmq
from threading import Thread
import cv2
import os
import uuid
from PIL import Image
import time


app = Flask(__name__)
host_name = "127.0.0.1"
port = 3000

prediction_data = None

def main():
    bd = BarcodDetection(prototxt=Config.BARCODE_PROTOTYPE_FILE, caffemodel=Config.BARCODE_CAFFE_MODEL_FILE)
    barcode_switch_detector = BarcodeSwitchDetector(db_file_path=Config.BARCODE_DB_FILE)
    obj_det = ObjectDetector(model_path=Config.OBJECT_DETECTION_MODEL_FILE, labels_path=Config.OBJECT_DETECTION_LABEL_FILE)

    classifier = ClassifyLabel(model_path="./weights/model_unquant.tflite", labels_path="./weights/labels.txt")

    StateMachine_handler = FSM()

    cv_cap = OpenCVCameraHandler(camera_id=Config.SCANNER_CAMERA_ID)
    cv_cap.setFocus(focus_value=Config.SCANNER_CAMERA_FOCUS)

    os.makedirs(Config.OUTPUT_DIR,exist_ok=True)


    FramesToSkipAfterScan = 2
    FrameCounterForScan = 0
    IsScanHappened= False

    context = zmq.Context()
    zmq_socket = context.socket(zmq.SUB)
    zmq_socket.connect("tcp://localhost:5555")
    zmq_socket.setsockopt_string(zmq.SUBSCRIBE, '')
    global prediction_data

    while cv_cap.isCapOpen():

        ret, frame = cv_cap.getFrame()

        if prediction_data is not None:
            person_state = max(prediction_data['prediction'], key=lambda x: x['probability'])

        try:
            data = zmq_socket.recv_json(flags=zmq.NOBLOCK)  
            print("Received data:", data)
        except zmq.Again:
            pass

        if ret:

            # draw_image = frame.copy()

            if IsScanHappened:
                if FrameCounterForScan<FramesToSkipAfterScan:
                    FrameCounterForScan +=1
                    continue
                else:
                    IsScanHappened = False
                    FrameCounterForScan = 0
            else:
                # barcode_number,draw_image = bd.detect(frame)
                draw_image = frame.copy()

                """
                Barcode Detection
                """
                barcode_number, barcode_polygon = bd.scan_barcode(frame)

                if barcode_number is not None:

                    IsScanHappened = True

                    draw_image = cv2.polylines(draw_image, [barcode_polygon], True, (0, 255, 0), 2)

                    # """
                    # Save barcode detected product image
                    # """
                    # product_data = None
                    # product_name = None
                    # try:
                    #     product_data = barcode_switch_detector.product_database.get(barcode_number, None)
                    # except KeyError:
                    #     pass
                    
                    # if product_data is not None:
                    #     product_category = product_data['product_category']
                    #     product_price = product_data['price']
                    #     product_name = product_data['name']
                    
                    # if product_name is not None:
                    #     barcode_number = product_name
                    
                    if Config.SAVE_BARCODE_IMAGES:
                        image_file_name = f"{Config.OUTPUT_DIR}/{barcode_number}_{str(uuid.uuid4())[:5]}.jpg"
                        cv2.imwrite(image_file_name, frame)

                
                    """
                    Object Classfication
                    """
                    predLabels, predConfidences = classifier.predict(frame, isOpenCVImage=True, top=3)
                    # print(barcode_number, predLabels, predConfidences)
                    
                    """
                    Object Detection
                    """
                    # detected_boxes, detected_labels, confidences = obj_det.detect_object(input=frame,confidence_threshold=0.6)
                    # for i in range(len(detected_boxes)):
                    #     xmin, ymin, xmax, ymax = detected_boxes[i]
                    #     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                    """
                    Ticket Switch Detection
                    """
                    IS_TICKET_SWITCH = barcode_switch_detector.detect_barcode_switch(predLabels, predConfidences, barcode_number=barcode_number,take_top_n=2)
                    if IS_TICKET_SWITCH:
                        # print("Ticket switch")
                        
                        h,w,c = draw_image.shape
                        posx, posy = int(w/2), int(h/2)
                        cv2.putText(draw_image, "Ticket Switch", (posy,posx), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
                        cv2.imshow(cv_cap.window_name, draw_image)

                        if IS_TICKET_SWITCH:
                            cv2.waitKey(2000)

                    
                        
                
        cv2.imshow(cv_cap.window_name, draw_image)

        # """
        # TODO 
        # Wait for person Entry (get it from pose estimation module)
        # """
        # stateUpdated = FSM_handler.update_state(STATE=STATES.PERSON_ENTRY)

        # """
        # Wait for person to Scan the object, or when the bard code is detected update this state
        # """
        # stateUpdated = FSM_handler.update_state(STATE=STATES.SCANNING)


        # """
        # Wait for person to make a payment (wither by UI action or from the pose estimation)
        # """
        # stateUpdated = FSM_handler.update_state(STATE=STATES.PAYING)


        # """
        # Wait for person to exit (from the pose estimation)
        # Upon receiving exit, check if person has scanned the objects, and have performed the payment.
        # Reset all flags

        # FSM_handler.reset_FSM()

        # LAST_3_STATES = FSM_handler.get_last_3_states()

        # """
        # stateUpdated = FSM_handler.update_state(STATE=STATES.PERSON_EXIT)
        if IsScanHappened: cv2.waitKey(2000)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/receive_state', methods = ['POST'])
def receive_state():
    global prediction_data
    prediction_data = request.get_json()
    data = {'message': 'Done', 'code': 'SUCCESS'}
    return make_response(jsonify(data), 201)


if __name__ == "__main__":
    # main()
    flask_thread = Thread(target=lambda: app.run(host=host_name, port=port, debug=True, use_reloader=False))
    flask_thread.start()
    # main()
    flask_thread.join()
