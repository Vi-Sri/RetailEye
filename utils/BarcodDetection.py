import cv2
import numpy as np
from pyzbar.pyzbar import decode


class BarcodDetection:

    def __init__(self, prototxt, caffemodel) -> None:
        print(prototxt, caffemodel)
        self.__barcode_detector = cv2.barcode.BarcodeDetector(prototxt, caffemodel)

    def detect(self, image) -> (int, np.ndarray):

        draw_image = image.copy()
        barcode = None

        ret_bc, decoded_info, points, _ = self.__barcode_detector.detectAndDecodeMulti(image)

        if ret_bc:
            # print(points)
            draw_image = cv2.polylines(draw_image, points.astype(int), True, (0, 255, 0), 3)
            for s, p in zip(decoded_info, points):
                if s:
                    barcode = s
                    # print(barcode)
                    draw_image = cv2.putText(draw_image, s, p[1].astype(int),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

        return barcode,draw_image
    

    def scan_barcode(self, frame):
        barcodes = decode(frame)
        for barcode in barcodes:
            barcode_data = barcode.data.decode("utf-8")
            barcode_polygon = barcode.polygon
            # print(barcode_data)
            return barcode_data, np.array(barcode_polygon)
        return None, None