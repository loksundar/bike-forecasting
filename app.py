from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np
app = Flask(__name__)
with open('dtree.pkl', 'rb') as file:  
    model = pickle.load(file)
cols = ['season', 'holiday','workingday', 'weather', 'temp', 'humidity','windspeed', 'casual', 'registered',  'hour', 'month']

@app.route('/')
def home():
	return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
	featu = [int(x) for x in request.form.values()]
	final = np.array(featu)
	data_unseen = pd.DataFrame([final], columns = cols)
	prediction = model.predict(data_unseen)
	prediction = np.exp(int(prediction[0]))
	return render_template('home.html',pred="Hourly rate count will be : {}".format(int(prediction)))
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = model.predict(final)
    output = prediction.Label[0]
    return jsonify(output)
if __name__ == '__main__':
    app.run(debug=True)
