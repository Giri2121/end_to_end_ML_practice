import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__) #starting point of app
model = pickle.load(open('reg_model.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data'] #input is given in json format
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    transformed_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(transformed_data)
    print(output[0])
    return jsonify(output[0])


@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()] #getting values from the front end form through request
    final_data = scalar.transform(np.array(data).reshape(1,-1))
    print(final_data)
    output = model.predict(final_data)[0]
    return render_template("home.html",prediction_text=f"Predicted price is {output}")

if __name__ == "__main__":
    app.run(debug=True)



