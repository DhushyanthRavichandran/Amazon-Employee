import pickle
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd
app = Flask(__name__, template_folder='template')

model=pickle.load(open('catboostModel.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(np.array(list(data.values())).reshape(1,-1))
    result = int(output[0])  # Convert int64 to int
    return jsonify({'result': result})
if __name__ == '__main__':
    app.run(debug=True)