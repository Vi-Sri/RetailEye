import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import enum

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
            self.class_names = [line.strip() for line in f.readlines()]
            print(self.class_names)

        # Getting input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def pre_process_image(self, input_image):
        frame_resized = ImageOps.fit(input_image, (224,224), Image.Resampling.LANCZOS)
        frame_rgb = frame_resized.convert('RGB')
        frame_normalized = (np.array(frame_rgb, dtype=np.float32) / 127.5) - 1
        input_data = np.expand_dims(frame_normalized, axis=0).astype(np.float32)
        return input_data
    
    def predict(self, input):
        input_data = self.pre_process_image(input)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        prediction_arg = np.argmax(prediction)
        return self.class_names[prediction_arg], prediction[prediction_arg], prediction
        


    