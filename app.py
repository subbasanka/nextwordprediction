#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from flask import Flask,render_template,session,url_for,redirect
from flask_wtf import FlaskForm
from wtforms import TextField, SubmitField
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd


app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


def get_words(tokenizer,model,data):
    input_text = data['text'].strip().lower()
    encoded_text = tokenizer.texts_to_sequences([input_text])[0]
    pad_encoded = pad_sequences([encoded_text],maxlen=3, truncating='pre')
    predicted_words = [] 
    for i in (model.predict(pad_encoded)[0]).argsort()[-5:][::-1]:
        pred_word = tokenizer.index_word[i]
        predicted_words.append(pred_word)

    return predicted_words
    del predicted_words


class SearchBar(FlaskForm):
    
    text = TextField('Search Bar')
    
    submit = SubmitField('Analyse')


@app.route('/',methods=['GET','POST'])
def index():
 search = SearchBar()
 
 if search.validate_on_submit():
     
     session['text'] = search.text.data
     
     return redirect(url_for('prediction'))
 
 return render_template('home.html',form=search)
     


#user_model = load_model('text_classifier.h5')
user_model = load_model('model2.h5')

with open('tokenizer.pickle','rb') as handle:
    user_tokenizer = pickle.load(handle)
    
@app.route('/prediction',methods=['POST','GET'])
def prediction():
    
    content = {}
    
    content['text'] = session['text']
    results = get_words(user_tokenizer,user_model,content)
    
    return render_template('prediction.html',results=results)


if __name__ == "__main__":
    app.run(debug=True)