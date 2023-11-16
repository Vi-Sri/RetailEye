class ObjectDetector:

    def __init__(self) -> None:
        """
        TODO Initialize object detector
        """
        self.detector = None


    def detect_object(self, image):
        """
        Detects objects
        """
        
        detected_boxes, detected_labels, confidences = self.detector(image)

        # return detected_boxes, detected_labels, confidences
        return NotImplementedError()