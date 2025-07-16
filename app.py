from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load trained model
model = joblib.load("xgb_best_model.pkl")

# Load dataset and fit scaler
df = pd.read_csv("fetal_health.csv")
X = df.drop('fetal_health', axis=1)
scaler = StandardScaler()
scaler.fit(X)

# Feature names
features = list(X.columns)

@app.route('/')
def index():
    return render_template("index.html", feature_names=features)

@app.route('/predict', methods=['POST'])
def predict():
    # Read inputs
    input_values = [float(request.form[feature]) for feature in features]
    input_array = np.array(input_values).reshape(1, -1)

    # Scale the input
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prediction_map = {0: "Normal", 1: "Suspect", 2: "Pathological"}
    result = prediction_map[int(prediction)]

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)