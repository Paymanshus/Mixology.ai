import os
from flask import Flask, request, render_template
import requests
import pickle as pk
from bs4 import BeautifulSoup

from preprocessing import preprocess_text

import urllib.request as urllib
from urllib.request import urlopen


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__, template_folder='templates')


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
		return render_template('upload.html', filename=filename, pred=pred)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

    return render_template("index.html", pred=output, scroll="scrollable")


# ---------------------------------------------
# Tabular Information Page (for later possibly)
# ---------------------------------------------
@app.route('/explore')
def explore():
    return render_template(".html")


if __name__ == '__main__':
    app.run(debug=True)
