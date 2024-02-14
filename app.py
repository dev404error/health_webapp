from flask import Flask, request, render_template, flash, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import pickle
import imutils
import numpy as np
from tensorflow.keras.models import load_model
from flask_cors import CORS
import base64  
import pandas as pd  
import traceback 

# Loading Models
covid_model = load_model('models/covid.h5')
diabetes_model = pickle.load(open('models/diabetes.sav', 'rb'))
heart_model = pickle.load(open('models/heart_disease.pickle.dat', "rb"))
pneumonia_model = load_model('models/pneumonia_model.h5')



# Configuring Flask
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"
CORS(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



########################### Routing Functions ########################################

@app.route('/')
def home():
    return render_template('homepage.html')


@app.route('/covid')
def covid():
    return render_template('covid.html')


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')


@app.route('/heartdisease')
def heartdisease():
    return render_template('heartdisease.html')


########################### Result Functions ########################################

@app.route('/resultc', methods=['POST'])
def resultc():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = cv2.resize(img, (224, 224))
            img = img.reshape(1, 224, 224, 3)
            img = img / 255.0
            pred = covid_model.predict(img)
            pred_label = 'NEGATIVE' if pred < 0.5 else 'POSITIVE'
            return render_template('resultc.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred_label, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)



@app.route('/resultd', methods=['POST'])
def resultd():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        pregnancies = request.form['pregnancies']
        glucose = request.form['glucose']
        bloodpressure = request.form['bloodpressure']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        diabetespedigree = request.form['diabetespedigree']
        age = request.form['age']
        skinthickness = request.form['skin']
        pred = diabetes_model.predict([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age]])
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resultd.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

@app.route('/resultp', methods=['POST'])
def resultp():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (150, 150))
            img = img.reshape(1, 150, 150, 3)
            img = img/255.0
            pred = pneumonia_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour COVID-19 test results are ready.\nRESULT: {}'.format(firstname,['POSITIVE','NEGATIVE'][pred]))
            return render_template('resultp.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)


@app.route('/resulth', methods=['POST'])
def resulth():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        nmv = float(request.form['nmv'])
        tcp = float(request.form['tcp'])
        eia = float(request.form['eia'])
        thal = float(request.form['thal'])
        op = float(request.form['op'])
        mhra = float(request.form['mhra'])
        age = float(request.form['age'])
        print(np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
        pred = heart_model.predict(
            np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
       
        return render_template('resulth.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

# Add a new route to handle screenshot saving
@app.route('/save_screenshot', methods=['POST'])
def save_screenshot():
    if request.method == 'POST':
        data = request.get_json()
        screenshot_data = data.get('screenshotData')
        filename = data.get('filename')

        # Save the screenshot on the server
        screenshot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'screenshot_' + filename)
        with open(screenshot_path, 'wb') as f:
            f.write(base64.b64decode(screenshot_data.split(',')[1]))

        return jsonify({'message': 'Screenshot saved successfully.'})


@app.route('/save_to_csv', methods=['POST'])
def save_to_csv():
    if request.method == 'POST':
        result_type = request.form.get('result_type')
        data = request.form.to_dict()

        csv_file_path = 'static/results_data.csv'

        # Check if the CSV file exists, create it if not
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, 'w') as csv_file:
                header = ['Result Type'] + list(data.keys())
                csv_file.write(','.join(header) + '\n')

        # Append data to the CSV file
        with open(csv_file_path, 'a') as csv_file:
            row = [result_type] + list(data.values())
            csv_file.write(','.join(row) + '\n')

        return jsonify({'message': 'Data saved to CSV successfully.'})



# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(debug=True)
