import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    sex = request.form['sex']
    age = request.form['age']
    intubed = request.form['intubed']
    pneumonia = request.form['pneumonia']
    pregnancy = request.form['pregnancy']
    diabetes = request.form['diabetes']
    copd = request.form['copd']
    asthma = request.form['asthma']
    hypertension = request.form['hypertension']
    cardiovascular = request.form['cardiovascular']
    obesity = request.form['obesity']
    renal_chronic = request.form['renal_chronic']
    tobacco = request.form['tobacco']
    contact_other_covid = request.form['contact_other_covid']

    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    prediction = model.predict([[sex,age,intubed,pneumonia,pregnancy,diabetes,copd,asthma,hypertension,cardiovascular,obesity,renal_chronic,tobacco,contact_other_covid]])

    output = round(prediction[0], 2)
    if output==1.0:
        return render_template('index.html', prediction_text='Prediksi Terpapar Covid : Positif')
    elif output==2.0:
        return render_template('index.html', prediction_text='Prediksi Terpapar Covid : Negatif')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
