!pip install transformers torch pillow torchvision
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
url = "https://www.parents.com/thmb/VK_eMsHSWaYAAAuFnyO88r_mh0=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/GettyImages-901208614-2000-9d4cdf4d1ad94fcb97ca78d67836a9d8.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(image, return_tensors="pt")
caption = model.generate(**inputs)
print(processor.decode(caption[0], skip_special_tokens=True))

!pip install deepface opencv-python
import cv2
from deepface import DeepFace
image_path = "/content/faces.jpg"
image = cv2.imread(image_path)
result = DeepFace.analyze(image, actions=['emotion'])
print("Detected Emotion:", result[0]['dominant_emotion'])
