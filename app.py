# -*- coding: utf-8 -*-
"""
 17:06:01 2019
@author: Ali
"""
import flask
from flask import Flask, request , jsonify, render_template, url_for, render_template
import jinja2
import tensorflow_text
import tensorflow as tf
import numpy as np

# loading the model from disk
model = tf.saved_model.load("CRM__bert_en_uncased_L-12_H-768_A-12/")
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index_2.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
            message  = str(request.form['message'])
            data = [message]
            results = model(data)
            my_prediction = np.argmax(tf.sigmoid(results))

            if my_prediction == int(1):
                output = 1
            else:
                output = 0
    return render_template('after.html',data = output)

if __name__ == "__main__":
    app.run(debug=True)
