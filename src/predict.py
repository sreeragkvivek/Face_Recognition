import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from PIL import Image


def predict_img(path):
    img = np.asarray(Image.open(path))
    model=tf.keras.models.load_model('../cnn.h5')

    resized_img = cv2.resize(img, (64, 64))
    resized_img = resized_img / 255.0
    resized_img = resized_img.reshape(-1, 64, 64, 1)

    return(np.argmax(model.predict(resized_img)))
