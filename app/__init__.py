from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

data = pd.read_csv('health_care_diabetes.csv')

with open('random_forest_model.pkl', 'rb') as file:
    random_forest_model = joblib.load(file)

@app.route('/', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            float_features = [float(x) for x in request.form.values()]
            mean = [3.845052083,120.8945313,69.10546875,20.53645833,79.79947917,31.99257813,0.471876302,33.24088542,0.348958333]
            std_dev = [3.369578063,31.9726182,19.35580717,15.95221757,115.2440024,7.88416032,0.331328595,11.76023154,0.476951377]
    
            scaled_features = []
            if len(float_features) == 0:
                raise ValueError("No features provided. Please make sure you input valid feature values.")

            scaled_features = [(x - m) / s for x, m, s in zip(float_features, mean, std_dev)]

            features = [np.array(scaled_features)]
            prediction = random_forest_model.predict(features)

            return render_template('C://Users/Tarayan/OneDrive/Desktop/RanPythonCode/app/templates/base.html', pred='The predicted value is {}'.format(prediction[0]))

    except Exception as e:
        return render_template('C://Users/Tarayan/OneDrive/Desktop/RanPythonCode/app/templates/base.html', error_message=str(e))

    return render_template('C://Users/Tarayan/OneDrive/Desktop/RanPythonCode/app/templates/base.html', pred='')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
