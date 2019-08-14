import cv2
from keras.models import load_model
import matplotlib.pyplot as plt

def prepare(filepath):
    IMG_SIZE = 28
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
