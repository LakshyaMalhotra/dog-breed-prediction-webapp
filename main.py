import os
from glob import glob
import numpy as np
from detector.detector import show_image, detect_face, dog_detector
from pretrained_models import model
from tqdm import tqdm
import torch

# Counting all the images 
dog_files = np.array(glob("dog_images/*/*/*"))
human_files = np.array(glob("lfw/*/*"))

print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))

# Un-decorating the original function to just use the face detector
face_detector = detect_face.__wrapped__

# Creating mini-datasets to test the performance of the detectors
human_files_short = human_files[:100]
dog_files_short = dog_files[:100]

def test_performance(detector, human_images, dog_images):
    """
    Function to test the performance of dog and human detectors.

    Arguments:
    ==========
    :param detector: function (type of detector (face or dog) to be used)
    :param human_images: (ndarray) image dataset containing human 
    :param dog_images: (ndarray) image dataset containing dogs
    """
    use_cuda = torch.cuda.is_available()
    detected = 'dog' if (detector.__name__.split('_')[0] == 'dog') else 'human'
    human_files_pct = sum([detector(img, use_cuda)[0] for img in tqdm(human_images)]) / len(human_images)
    dog_files_pct = sum([detector(img, use_cuda)[0] for img in tqdm(dog_images)]) / len(dog_images)

    human_files_pct *= 100
    dog_files_pct *= 100

    print("First 100 images in human_files have {0}% {1} detected.".format(human_files_pct, detected))
    print("First 100 images in dog_files have {0}% {1} detected.".format(dog_files_pct, detected))


if __name__ == '__main__':
    
    # detect_face(human_files[15])      # original function which shows the BBox
    _, _, faces = face_detector(human_files[23])
    print("Number of faces detected: {}".format(len(faces)))

    # Assessing the performance of face detector
    test_performance(face_detector, human_files_short, dog_files_short)

    # Assessing the performance of dog detector
    test_performance(dog_detector, human_files_short, dog_files_short)
