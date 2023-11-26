import os

class Config:
    ROOT_DIR = ".\\"
    # BARCODE_PROTOTYPE_FILE = os.path.join(ROOT_DIR, "opencv_3rdparty\\detect.prototxt")
    # BARCODE_CAFFE_MODEL_FILE = os.path.join(ROOT_DIR, "opencv_3rdparty\\detect.caffemodel")
    BARCODE_PROTOTYPE_FILE =  "./opencv_3rdparty/detect.prototxt"
    BARCODE_CAFFE_MODEL_FILE = "./opencv_3rdparty/detect.caffemodel"
    # BARCODE_DB_FILE = "./bardcodedb/barcodedb.csv"
    BARCODE_DB_FILE = "./FrieburgPackage/product_database.json"


    OBJECT_DETECTION_MODEL_FILE = "./FrieburgPackage/custom_model_lite/saved_model/detect.tflite"
    OBJECT_DETECTION_LABEL_FILE = "./FrieburgPackage/custom_model_lite/labelmap_new.txt"
    
    CAMERA_ID = 0
    CAMERA_FOCUS = 40

    TOP_VIEW_CAMERA_ID = 1
    TOP_VIEW_CAMERA_FOCUS = 40