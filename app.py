# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 17:06:01 2019

@author: prithvi
"""
import flask
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer as wnl
import re
# load the model from disk
filename = 'NLP_model2.pkl'
clf = pickle.load(open(filename, 'rb'))
cvv=pickle.load(open('vectorizer.pkl','rb'))

# Defining the function for cleaning
stpwrds = set(stopwords.words('english'))
stpwrds.discard('not')
stpwrds.update(['intrstd','endnan','nan','end','r','im','bt','tmrw'])
def clean_text1(text):  
    #no_punctuation = [char for char in text if char not in string.punctuation]
    #print(no_punctuation)
    #no_punctuation1 = ''.join(no_punctuation)
    #print(no_punctuation1)
    text = str(text)
    words = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", text )  
    pattren = r"[\d]"
    words = re.sub(pattren, '', words)
    words = words.lower()
    final_words =  [wnl().lemmatize(word) for word in words.split() if word not in stpwrds]
    final_words = ' '.join(final_words)
    #return(list(ngrams(final_words,2)))
    return(final_words)
app = Flask(__name__)

@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data1 = request.form['cust_name']
    data2 = request.form['location']
    data3 = request.form['message']
    text_actual = str(data2) + ' ' + str(data3)
    text_actual = clean_text1(text_actual)
    text_actual = [text_actual]
    bag = cvv.transform(text_actual).toarray()
    
    
    pred = clf.predict(bag)
    return render_template('after.html',data=pred)


if __name__ == "__main__":
    app.run(debug=True)
