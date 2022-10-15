import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('IRIS.mdl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_width = float(request.form['petal_width'])
    petal_length = float(request.form['petal_length'])
    finalFeatures = pd.DataFrame(np.array([[sepal_length,sepal_width,petal_length,petal_width]]) , columns=['sepal.length','sepal.width','petal.length','petal.width']) 
    prediction = model.predict(finalFeatures)

    

    return render_template('index.html', prediction_text= " IRIS variery is:"+prediction[0])


if __name__ == "__main__":
    app.run(debug=True)