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
import base64
from PIL import Image
from io import BytesIO

from flask_socketio import SocketIO

app = Flask(__name__)
host_name = "127.0.0.1"
port = 3000
# app = Flask(__name__)
# app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# socketio = SocketIO(app)
# socketio.init_app(app, cors_allowed_origins="*")
# host_name = "127.0.0.1"
# port = 3000



prediction_data = None
last_scanned_product = None
ScannedItems = []
TicketSwitchDetected = False
SuccessTransationCount = 0
TheftTransactionCount = 0

def getBase64Image(arry):
    """
    Converts opencv type of image to web based image to display on web
    """
    pil_img = Image.fromarray(arry)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    img = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img

def main():

    global TicketSwitchDetected, SuccessTransationCount, TheftTransactionCount, ScannedItems
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

    PaymentCounter = 0
    PaymentCounterMax = 5
    PaymentDone = False

    global prediction_data

    while cv_cap.isCapOpen():

        ret, frame = cv_cap.getFrame()
        person_state = None

        if prediction_data is not None:
            # print(prediction_data)
            person_state = max(prediction_data['prediction_global'], key=lambda x: x['probability'])
            # "labels":["No person","Person Entry","Scanning","Weighing","Payment","Person exit"]

        # try:
        #     data = zmq_socket.recv_json(flags=zmq.NOBLOCK)  
        #     print("Received data:", data)
        # except zmq.Again:
        #     pass
        # print(StateMachine_handler.get_current_state())
        # print(ScannedItems)

        if ret:
            draw_image = frame.copy()

            # person_state = {"className": "Scanning", "probability": 0.99}

            if person_state is not None:
                personStateConf = person_state["probability"]
                if person_state["className"]=="No person":
                    if personStateConf>0.99:
                        last3States = StateMachine_handler.get_last_3_states()
                        if STATES.SCANNING in last3States:
                            if STATES.PAYING in last3States:
                                SuccessTransationCount +=1
                            else:
                                TheftTransactionCount +=1
                        StateMachine_handler.update_state(STATE=STATES.PERSON_EXIT, confidence=personStateConf)
                        StateMachine_handler.reset_FSM()
                        TicketSwitchDetected = False
                        PaymentCounter = 0
                    # pass
                
                elif person_state["className"]=="Person Entry":
                    if personStateConf>=1:
                        if StateMachine_handler.get_current_state()!=STATES.PERSON_ENTRY:
                            if StateMachine_handler.get_current_state()==STATES.PERSON_EXIT:
                                stateUpdated = StateMachine_handler.update_state(STATE=STATES.PERSON_ENTRY, confidence=personStateConf)
                                if not stateUpdated:
                                    print(f"Error upadting state: {STATES.PERSON_ENTRY}")

                elif person_state["className"]=="Scanning":
                    # TODO Add scanner code here to ensure scanning has happened
                    # if StateMachine_handler.get_current_state()!=STATES.SCANNING:
                    if personStateConf>0.90:
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
                            product_data = None
                            product_name = None
                            product_price = None
                            try:
                                product_data = barcode_switch_detector.product_database.get(barcode_number, None)
                            except KeyError:
                                pass
                            
                            if product_data is not None:
                                product_category = product_data['product_category']
                                product_price = product_data['price']
                                product_name = product_data['name']


                            else:
                                print("Product data is None")

                            
                            # if product_name is not None:
                            #     barcode_number = product_name
                            
                            if Config.SAVE_BARCODE_IMAGES:
                                image_file_name = f"{Config.OUTPUT_DIR}/{barcode_number}_{str(uuid.uuid4())[:5]}.jpg"
                                cv2.imwrite(image_file_name, frame)

                            """
                            Object Classfication
                            """
                            predLabels, predConfidences = classifier.predict(frame, isOpenCVImage=True, top=3)
                            print(barcode_number, predLabels, predConfidences)
                            
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
                                TicketSwitchDetected = True

                                # if IS_TICKET_SWITCH:
                                #     cv2.waitKey(2000)

                            ScannedItem = {
                                    "product_name" : product_name, 
                                    "product_img" : getBase64Image(draw_image),
                                    "product_price" : f"{product_price}$"
                            }

                            # socketio.emit('scanned_list',ScannedItem)
                            # socketio.emit('ticket_switch',TicketSwitchDetected)
                            is_present = False  
                            for elem in ScannedItems:
                                if elem["product_name"] == product_name:
                                    is_present = True
                            
                            if not is_present:
                                ScannedItems.append(ScannedItem)

                            stateUpdated = StateMachine_handler.update_state(STATE=STATES.SCANNING, confidence=personStateConf)
                            if not stateUpdated:
                                print(f"Error upadting state: {STATES.SCANNING}")

                elif person_state["className"]=="Weighing":
                    if StateMachine_handler.get_current_state()!=STATES.WEGHING:
                        stateUpdated = StateMachine_handler.update_state(STATE=STATES.WEGHING, confidence=personStateConf)
                        if not stateUpdated:
                            print(f"Error upadting state: {STATES.SCANNING}")

                elif person_state["className"]=="Payment":
                    if person_state["probability"]>=1:
                        if PaymentCounter<PaymentCounterMax:
                            PaymentCounter +=1
                        else:
                            stateUpdated = StateMachine_handler.update_state(STATE=STATES.PAYING, confidence=personStateConf)
                            if not stateUpdated:
                                print(f"Error upadting state: {STATES.PAYING}")
                    pass

                elif person_state["className"]=="Person exit":
                    if personStateConf>=1:
                        last3States = StateMachine_handler.get_last_3_states()
                        if STATES.SCANNING in last3States:
                            if STATES.PAYING in last3States:
                                SuccessTransationCount +=1
                            else:
                                TheftTransactionCount +=1
                        
                        
                        StateMachine_handler.reset_FSM()
                        TicketSwitchDetected = False
                        PaymentCounter = 0

                
                # socketio.emit('Statemachine',StateMachine_handler.get_current_state().name)  


                # print(StateMachine_handler.get_last_3_states(), f"Success: {SuccessTransationCount} Theft: {TheftTransactionCount}")
                    



            # draw_image = frame.copy()
            # if IsScanHappened:
            #     if FrameCounterForScan<FramesToSkipAfterScan:
            #         FrameCounterForScan +=1
            #         continue
            #     else:
            #         IsScanHappened = False
            #         FrameCounterForScan = 0
            # else:
            #     # barcode_number,draw_image = bd.detect(frame)
            #     pass   
                
            cv2.imshow(cv_cap.window_name, draw_image)

        # """
        # stateUpdated = FSM_handler.update_state(STATE=STATES.PERSON_EXIT)
        # if IsScanHappened: cv2.waitKey(2000)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# @socketio.on('prediction_state')
# def handle_prediction_state(json):
#     global prediction_data
#     prediction_data = json
#     print('received json: ' + str(json))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/receive_state', methods = ['POST'])
def receive_state():
    global prediction_data
    prediction_data = request.get_json()
    data = {'message': 'Done', 'code': 'SUCCESS'}
    return make_response(jsonify(data), 201)

@app.route('/get_scanned_items', methods = ['GET'])
def get_scanned_items():
    global ScannedItems
    data = {'message': 'Done', 'code': 'SUCCESS', 'data': ScannedItems}
    ScannedItems = []
    return make_response(jsonify(data), 201)


@app.route('/get_transaction_counts', methods = ['GET'])
def get_transaction_counts():
    global SuccessTransationCount, TheftTransactionCount
    data = {'message': 'Done', 'code': 'SUCCESS', 'data': {"Success": SuccessTransationCount, "Theft": TheftTransactionCount}}
    return make_response(jsonify(data), 201)


if __name__ == "__main__":
    flask_thread = Thread(target=lambda: app.run(host=host_name, port=port, debug=False, use_reloader=False))
    # flask_thread = Thread(target=lambda: socketio.run(app, host=host_name, port=port, debug=False, use_reloader=False))
    flask_thread.start()

    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)


    main()
    flask_thread.join()
