from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Załaduj modele
regression_model = load_model("../models/regression_model.h5")
classification_model = load_model("../models/classification_model.h5")

classification_model.summary()

@app.route("/")
def index():
    return render_template("index.html")  # Strona główna z formularzem

@app.route("/predict/regression", methods=["POST"])
def predict_regression():
    data = request.json  # Dane wejściowe w formacie JSON
    input_data = np.array(data["features"]).reshape(1, -1)  # Dopasowanie do modelu
    # Dodaj linię, aby sprawdzić kształt danych
    print("Kształt danych wejściowych:", input_data.shape)
    prediction = regression_model.predict(input_data)
    return jsonify({"prediction": float(prediction[0])})

@app.route("/predict/classification", methods=["POST"])
def predict_classification():
    data = request.json  # Dane wejściowe w formacie JSON
    input_data = np.array(data["features"]).reshape(1, -1)
    # Dodaj linię, aby sprawdzić kształt danych
    print("Kształt danych wejściowych:", input_data.shape)
    prediction = classification_model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)  # Klasa z największym prawdopodobieństwem
    return jsonify({"prediction": int(predicted_class[0])})

if __name__ == "__main__":
    app.run(debug=True)
