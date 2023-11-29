import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import enum
import cv2

class Labels(enum.Enum):
    FiberGummies = 1
    SDCard = 2
    Gum = 3
    Unknown = 4
    

class ClassifyLabel(object):
    def __init__(self, model_path, labels_path) -> None:
        self.detector = None

        # Loading TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Loading class names
        with open(labels_path, 'r') as f:
            self.class_names = [line.strip().split(" ")[-1] for line in f.readlines()]
            print(self.class_names)

        # Getting input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def pre_process_image(self, input_image):
        # frame_resized = ImageOps.fit(input_image, (224,224), Image.Resampling.LANCZOS)
        frame_resized = input_image.resize((224,224))
        frame_rgb = frame_resized.convert('RGB')
        frame_normalized = (np.array(frame_rgb, dtype=np.float32) / 127.5) - 1
        input_data = np.expand_dims(frame_normalized, axis=0).astype(np.float32)
        return input_data
    
    def pre_process_opencv_image(self, input_image):
        input_image = cv2.resize(input_image, (224, 224))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        frame_normalized = (input_image / 127.5) - 1
        input_data = np.expand_dims(frame_normalized, axis=0).astype(np.float32)
        return input_data
    
    def predict(self, input, isOpenCVImage=False, top=3):
        processingFn = self.pre_process_image
        if isOpenCVImage:
            processingFn = self.pre_process_opencv_image

        input_data = processingFn(input)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        if len(prediction)<top:
            top = len(prediction)

        ind = np.argpartition(prediction, -top)[-top:]
        topPreds = prediction[ind]

        predLabels, predConfidences = [], []

        for i in range(len(topPreds)):
            predConfidences.append(topPreds[i])
            predLabels.append(self.class_names[ind[i]])

        zipped_detections = zip(predLabels, predConfidences)
        zipped_detections = sorted(zipped_detections, key=lambda x: x[1], reverse=True) # get high confidence first
        predLabels,predConfidences  = zip(*zipped_detections)
        return predLabels, predConfidences
        


    