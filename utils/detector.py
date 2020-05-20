import cv2
import matplotlib.pyplot as plt
from functools import wraps

from model import resnet_model, model_predict


def show_image(func):
    """
    Function decorator that shows the image with a bounding box if a face is detected,
    else just shows the image.

    Arguments:
    ==========
    :func: Function which detects a face
    :return: returns a wrapper which includes the bounding box in the image.
    """

    @wraps(func)  # un-decorates the fucntion if needed
    def wrapper(*args, **kwargs):
        # get bounding boxes for each detected face
        """
        :params args: the image path of the input image.
        and columns=coordinates of bbox)
        """
        face_detected, img, faces = func(*args, **kwargs)
        print("Number of faces detected:", len(faces))
        if face_detected:
            for (x, y, w, h) in faces:
                # add bounding box to the color image
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # convert BGR image to RGB image for plotting
        cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.imshow(cv_rgb)
        # display the image, along with the bounding box
        plt.show()
        # return cv_rgb

    return wrapper


@show_image
def detect_face(img_path, *args):
    """
    Function to read an image and detect the face using the Haarcascade model,
    returns True if a face is detected in the image.

    Arguments:
    ==========
    :param img_path: Path of the image to be read.
    :return: Boolean
    """

    # extract pre-trained face detector from OpenCV
    ### Add a pre-trained face detector of your choice.
    face_cascade = cv2.CascadeClassifier(
        "/opt/intel/openvino/opencv/etc/haarcascades/haarcascade_frontalface_alt.xml"
    )

    # use this if using GCP
    # face_cascade = cv2.CascadeClassifier('/home/Shrushti/anaconda3/envs/pt/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt.xml')

    # load the color (BGR) image
    img = cv2.imread(img_path)  # by default, cv2 reads images in BGR format.

    # convert BGR to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find faces in the image
    faces = face_cascade.detectMultiScale(gray)

    # "faces" will be a 2-D array whose number of rows tell you the number of
    # faces detected. The different columns will give you the coordinates of
    # bounding boxes.

    return len(faces) > 0, img, faces


def dog_detector(img_path, use_cuda):
    """
    Function to detect dog in the image. We need only check if the pre-trained
    model predicts an index between 151 and 268 (inclusive) since in the actual
    ImageNet dataset these indices are for the dogs.

    Arguments:
    ==========
    :param img_path: str (Path to the input image).
    :return: bool.
    """
    # loading the model and calulating the output
    model = resnet_model(use_cuda)
    idx = model_predict(img_path, model, use_cuda)

    if 151 <= idx <= 268:
        return True, idx
    else:
        return False, idx
