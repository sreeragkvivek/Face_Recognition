import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import pickle
from PIL import Image


def predict_img(path):
    image = cv2.imread(path)
    cv2.imwrite("./static/working-dir/final_output.png", image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    print("Found {0} Faces!".format(len(faces)))


    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # model=tf.keras.models.load_model('./models/cnn.h5')
    model = pickle.load(open('./models/NB.pickle','rb'))
    # print(img.reshape(-1,64,64))

    resized_img = cv2.resize(img, (64, 64))
    resized_img = resized_img / 255.0
    print(resized_img.shape)
    resized_img=resized_img.reshape(1, 64*64)
    return(model.predict(resized_img))



# print(predict_img("./src/index.png"))
