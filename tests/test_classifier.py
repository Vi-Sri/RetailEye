from utils.ClassifyLabel import ClassifyLabel
from PIL import Image

classifier = ClassifyLabel(model_path="model_unquant.tflite", labels_path="labels.txt")
print(classifier.predict(Image.open("4.jpg")))