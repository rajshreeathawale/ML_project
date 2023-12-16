from flask import Flask, render_template,request

import pickle
import numpy as np

model=pickle.load(open('lr_yield.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict_yield():
	data1=float(request.form.get('Soil_Quality'))
	data2=int(request.form.get('Seed_Variety'))
	data3=float(request.form.get('Fertilizer_Amount_kg_per_hectare'))
	data4=float(request.form.get('Sunny_Days'))
	data5=float(request.form.get('Rainfall_mm'))
	data6=int(request.form.get('Irrigation_Schedule'))

	result=model.predict(np.array([data1,data2,data3,data4,data5,data6]).reshape(1,6))
	return render_template('result.html',data=result)

if __name__ == "__main__":
    app.run(debug=True)

