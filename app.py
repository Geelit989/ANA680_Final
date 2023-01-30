from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
app = Flask(__name__)
filename = 'ridge.pkl'
#model = pickle.load(open(filename, 'rb'))
model = joblib.load(filename)
#model = joblib.load(filename)
@app.route('/')
def index(): 
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    MedianIncome = request.form['MedInc']
    HouseAge = request.form['HouseAge']
    AveRooms = request.form['AveRooms']
    AveBedrms = request.form['AveBedrms']
    Population = request.form['Population']
    AveOccup = request.form['AveOccup']
    Latitude = request.form['Latitude']
    Longitude = request.form['Longitude']

    
      
    pred = model.predict(np.array([[MedianIncome,
                                    HouseAge,
                                    AveRooms,
                                    AveBedrms,
                                    Population,
                                    AveOccup,
                                    Latitude,
                                    Longitude,
                                    ]]))
    print(pred)
    return render_template('index.html', predict=str(pred))


if __name__ == '__main__':
    app.run
