import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Define the model
def resnet_model(use_cuda=False):
    """
    Function to load the pretrained models from torchvision, move them to GPU
    if available.

    Arguments:
    ==========
    :param use_cuda: bool (use GPU if CUDA is available)
    :return: downloaded model
    """
    # define resnet model
    resnet50 = models.resnet50(pretrained=True)

    # move model to GPU
    if use_cuda:
        resnet50 = resnet50.cuda()

    return resnet50

# Calculate the model's prediction
def model_predict(img_path, model, use_cuda=False):
    """
    Use pre-trained ResNet-50 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path.
    
    Arguments:
    ==========
    :param img_path: str (path to an image)
    :param model: pretrained model
    :param use_cuda: bool (use GPU if CUDA is available)
    :return: Index corresponding to ResNet-50 model's prediction
    """

    # defining various image transforms
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])

    # applying the transforms on the input image
    image = Image.open(img_path)
    image = transform(image).float()
    image = image.unsqueeze(0)      # Creating a 4-D tensor with batch_size=1

    # moving the image to GPU if available
    if use_cuda:
        image.cuda()
    
    # putting the model to evaluation mode
    model.eval()

    # forward pass
    output = model(image)
    
    # moving the image to CPU
    output = output.cpu()

    # calculating the output class
    _, output_class = torch.max(output, 1)

    return output_class.numpy()[0]      # predicted class index

# Defining a baseline model to compare with the ResNet model
# Using an architecture of our choice
class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        
        ## Defining the layers of CNN
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)     
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3)

        self.fc1 = nn.Linear(128*5*5, 168)
        self.fc2 = nn.Linear(168, 133)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.25)
    
    def forward(self, x):
        ## Define forward behavior
        x = F.relu(self.conv1(x))    # shape: 3x224x224 --> 16x222x222
        x = self.maxpool(x)          # shape: 16x222x222 --> 16x111x111
        x = F.relu(self.conv2(x))    # shape: 16x111x111 --> 32x109x109
        x = self.maxpool(x)          # shape: 32x109x109 --> 32x54x54
        x = F.relu(self.conv3(x))    # shape: 32x54x54 --> 64x52x52
        x = self.maxpool(x)          # shape: 64x52x52 --> 64x26x26
        x = F.relu(self.conv4(x))    # shape: 64x26x26 --> 128x24x24
        x = self.maxpool(x)          # shape: 128x24x24 --> 128x12x12
        x = F.relu(self.conv5(x))    # shape: 128x12x12 --> 128x10x10
        x = self.maxpool(x)          # shape: 128x10x10 --> 128x5x5

        
        # flatten the layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
    