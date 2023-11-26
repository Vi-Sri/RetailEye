import cv2
import time
import numpy as np
import tensorflow as tf

class ObjectDetector:

    def __init__(self, model_path, labels_path) -> None:
        """
        TODO Initialize object detector
        """
        self.detector = None

        # Loading TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Loading class names
        with open(labels_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]


        # Getting input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def pre_process_image(self, input_image):
        frame_resized = cv2.resize(input_image, (320, 320))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb / 255.0
        input_data = np.expand_dims(frame_normalized, axis=0).astype(np.float32)
        return input_data


    def detect_object(self, input, confidence_threshold=0.6):
        """
        Detects objects and gets top N predictions
        """

        input_data = self.pre_process_image(input)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Get the detection scores, classes, and boxes
        detection_scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # StatefulPartitionedCall:1 (Detection Scores)
        detection_boxes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]  # StatefulPartitionedCall:3 (Bounding Boxes)
        detection_classes = self.interpreter.get_tensor(self.output_details[3]['index'])[0]  # StatefulPartitionedCall:2 (Classes)


        detected_boxes,confidences,detected_labels = self.post_process_detections(input, detection_boxes,detection_scores,detection_classes,threshold=confidence_threshold, sortDetections=True)

        return detected_boxes, detected_labels, confidences
    

    def post_process_detections(self,input,detection_boxes,detection_scores,detection_classes, threshold=0.6, sortDetections=True):
        h, w, _ = input.shape
        boxes = []
        scores = []
        classes = []

        for i in range(len(detection_scores)):
            score = detection_scores[i]
            if score > threshold:
                ymin, xmin, ymax, xmax = detection_boxes[i]
                ymin = int(ymin * h)
                ymax = int(ymax * h)
                xmin = int(xmin * w)
                xmax = int(xmax * w)
                detected_label = self.class_names[int(detection_classes[i])]

                boxes.append([xmin, ymin, xmax, ymax])
                scores.append(score)
                classes.append(detected_label)

        if sortDetections:
            zipped_detections = zip(boxes, scores, classes)
            zipped_detections.sort()
            boxes,scores,classes = zip(*zipped_detections)

        return boxes,scores,classes