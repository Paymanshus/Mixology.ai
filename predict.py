import os
import numpy as np
import pandas as pd
import cv2

from keras.models import load_model

IMG_WIDTH = 200
IMG_HEIGHT = 200


def get_model():
    global model_path

    model = load_model(model_path)
    return model


def process_image(image_path):

    image = cv2.imread(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),
                       interpolation=cv2.INTER_AREA)
    print(image)
    image = np.array(image)

    image = image.astype('float32')
    image /= 255
    print(image)

    return image


def predict_image(img_path):
    image = process_image(img_path)
    image = np.array(tf.expand_dims(image, axis=0))

    pred = np.argmax(model.predict(image), axis=-1)[0]

    id = pd.read_csv('data/id_df.csv')
    pred_name = id[id['ID'] == pred]

    return pred_name
