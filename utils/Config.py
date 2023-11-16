import os

class Config:
    ROOT_DIR = ".\\"
    # BARCODE_PROTOTYPE_FILE = os.path.join(ROOT_DIR, "opencv_3rdparty\\detect.prototxt")
    # BARCODE_CAFFE_MODEL_FILE = os.path.join(ROOT_DIR, "opencv_3rdparty\\detect.caffemodel")
    BARCODE_PROTOTYPE_FILE =  "./opencv_3rdparty/detect.prototxt"
    BARCODE_CAFFE_MODEL_FILE = "./opencv_3rdparty/detect.caffemodel"
    BARCODE_DB_FILE = "./bardcodedb/barcodedb.csv"

    CAMERA_ID = 1
    CAMERA_FOCUS = 40

    