# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 00:06:19 2021

@author: abdal
"""
from tensorflow.python import keras
from tensorflow.python.keras.preprocessing import image
from PIL import Image
from flask_cors import CORS,cross_origin
import numpy as np
from flask import Flask, request, jsonify, render_template,redirect
from joblib import load
from tensorflow.python.keras.models import load_model
app=Flask(__name__,template_folder='template')
foot_model = keras.models.load_model('diabetic_foot.h5')
model=load('diabetic_classfication.joblib')
model2 = load("food_cluster.joblib")

app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'

cors = CORS(app, resources={r"/foo": {"origins": "http://localhost:port"}})



# Load your own trained model
@app.route('/')
def home():
    return render_template('front.html')



@app.route('/predict_api',methods=['POST','GET'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data1=float(request.form['data1'])
    data2=float(request.form['data2'])
    data3=float(request.form['data3'])
    data4=float(request.form['data4'])
    data5=float(request.form['data5'])
    arr=np.array([[data1,data2,data3,data4,data5]])
    prediction=model.predict(arr)
    output =str(prediction[0])
    return output

@app.route('/predict_cluster',methods=['POST','GET'])
def predict_cluster():

    '''
    For direct API calls trought request
    '''
    data1=float(request.form['data1'])
    data2=float(request.form['data2'])
    data3=float(request.form['data3'])
    data4=float(request.form['data4'])
    data5=float(request.form['data5'])
    data6=float(request.form['data6'])
    data7=float(request.form['data7'])
    data8=float(request.form['data8'])
    arr=np.array([[data1,data2,data3,data4,data5,data6,data7,data8]])
    prediction=model2.predict(arr)
    output =str(prediction[0])
   
    return jsonify(output)






@app.route('/diabetic_foot', methods=["POST"])
def foot_prediction():
    img = request.files['img']
    img.save('img.jpg')

    img = image.load_img("img.jpg", target_size=(64,64))
    x=image.img_to_array(img) / 255
    resized_img_np = np.expand_dims(x,axis=0)
    prediction = foot_model.predict(resized_img_np)
    

    p1=1-prediction
    return 'predicted raio' +str(p1)

if __name__ == "__main__":
    app.run(host="localhost", port=5000)