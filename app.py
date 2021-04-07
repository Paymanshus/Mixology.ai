import os
from flask import Flask, request, render_template, flash, redirect
from werkzeug.utils import secure_filename
import requests
import pickle as pk
from bs4 import BeautifulSoup

import cv2

import pandas as pd
import numpy as np

# from predict import predict_image

import urllib.request as urllib
from urllib.request import urlopen

import tensorflow as tf
from keras.models import load_model


# ----------------
# Global Variables
# ----------------
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

codePath = os.path.dirname(os.path.abspath('app.py'))
model_path = os.path.join(codePath, 'models/custCNN1.h5')
model = load_model(model_path)

df = pd.read_csv('data/all_drinks.csv')
df.apply(lambda x: x.astype(str).str.upper())
df['strDrinkLower'] = df['strDrink'].apply(lambda x: x.lower())

IMG_WIDTH = 200
IMG_HEIGHT = 200


# --------------------
# Function Definitions
# --------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(image):

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
    global model
    image = process_image(img_path)
    image = np.array(tf.expand_dims(image, axis=0))

    pred = np.argmax(model.predict(image), axis=-1)[0]

    id = pd.read_csv('data/id_df.csv')
    pred_name = id[id['ID'] == pred]

    return pred_name


def return_details(pred):
    pred = pred.lower()

    ing_df = df[df['strDrinkLower'].str.contains(pred)]
    ing_list = ing_df.iloc[:, 9:24].dropna(axis=1).values.tolist()[0]

    recipe = ing_df['strInstructions'].values[0]

    return ing_list, recipe


app = Flask(__name__, template_folder='templates')

app.config['UPLOADS'] = 'uploads'
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route('/')
def home():
    return render_template("index.html")


# --------------------------------------
# Reciever And Processor Output Function
# --------------------------------------
@app.route('/', methods=['POST'])
def upload_image():

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print('upload_image filename: ' + filename)
        # flash('Image successfully uploaded and displayed below')

        pred = predict_image(filepath)

        ing_list, recipe = return_details(pred)
        # recipe = return_recipe(pred)

        return render_template('index.html', user_img=filename, pred=pred, scroll='scrollable', ingredients_list=ing_list, recipe=recipe)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

    # return render_template("index.html", pred=pred, scroll="scrollable", )


# ---------------------------------------------
# Tabular Information Page (for later possibly)
# ---------------------------------------------
@app.route('/explore')
def explore():
    return render_template(".html")


if __name__ == '__main__':
    app.run(debug=True)
