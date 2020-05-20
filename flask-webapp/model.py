import io

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def get_model():
    # Path of the saved model
    saved_model_path = "model_transfer.pt"

    # Total number of classes
    num_classes = 133

    # Loading the pretrained resnet50 model
    model = models.resnet50(pretrained=True)

    # Performing feature extraction; changing the last fc layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(
        torch.load(saved_model_path, map_location="cpu"), strict=False
    )

    # Putting the model to evaluation mode
    model.eval()
    return model


def preprocess(image_bytes):
    # Normalizing the images with ImageNet dataset
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Transforms for eval mode
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    image = Image.open(io.BytesIO(image_bytes))
    image = transform(image).unsqueeze(0)
    return image
