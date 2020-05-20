import os
import io
import json
import cv2
import numpy as np
from PIL import Image

from model import get_model, preprocess
import torch
import torch.nn.functional as F
from torchvision import models, transforms

with open("dog_names.json", "r") as f:
    names = json.load(f)

model = get_model()

cat_to_name = dict(enumerate(names, 1))
print(cat_to_name)


def detect_face(image_bytes):
    face_cascade = cv2.CascadeClassifier(
        "haarcascades/haarcascade_frontalface_alt.xml"
    )
    try:
        # Preprocess the image in bytes data (from web app) using OpenCV
        img_stream = io.BytesIO(image_bytes)
        image = cv2.imdecode(np.fromstring(img_stream.read(), np.uint8), 1)

        # Convert BGR to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # find faces in the image
        faces = face_cascade.detectMultiScale(gray)

        return len(faces) > 0, faces

    except:
        return 0, "error"


def detect_dog(image_bytes):
    resnet50_model = models.resnet50(pretrained=True)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = transform(image).unsqueeze(0)
        resnet50_model.eval()
        outputs = resnet50_model(image)
    except:
        return 0, "error"

    outputs.cpu()
    # calculating the output class
    _, output_class = torch.max(outputs, 1)
    class_index = output_class.numpy()[0]  # predicted class index

    if 151 <= class_index <= 268:
        return True, class_index
    else:
        return False, class_index


def get_best_match(image_bytes):
    try:
        image_tensor = preprocess(image_bytes=image_bytes)
        outputs = model.forward(image_tensor)
    except:
        return 0, "error"

    # probas = F.softmax(outputs, 1).to("cpu")
    print(outputs.detach().numpy())
    index = torch.argmax(outputs, dim=1).to("cpu")
    index = index.detach().numpy()[0] + 1
    name = cat_to_name[index]
    return index, name
