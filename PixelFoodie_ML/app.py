from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

from flask_cors import CORS
app = Flask(__name__)
CORS(app) # Idhai add pannunga

# Namma create panna AI model-ah load panrom
with open('pixel_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    # User first time varum pothu index.html kaatum
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Website form-la irundhu data-ve edukuroam
    spicy = int(request.form['spicy'])
    time = int(request.form['time'])
    veg = int(request.form['veg'])
    meal = int(request.form['meal'])
    cuisine = int(request.form['cuisine'])

    # ML model prediction logic
    prediction = model.predict([[spicy, time, veg, meal, cuisine]])
    
    # Result-oda thirumba index.html-ke pogum
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)