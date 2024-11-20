from pathlib import Path

from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS


base_dir = Path(__file__).parent
model_path = base_dir / "wemlAi.joblib"
model = joblib.load(model_path)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://localhost:7176"}})
@app.route('/search_symptoms', methods=['GET'])
def search_symptoms():

    query = request.args.get('term', '').lower()
    if not query:
        return jsonify([])

    matching_symptoms = [feature for feature in model.feature_names_in_ if query in feature.lower()]
    return jsonify(matching_symptoms[:10])

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    provided_symptoms = input_data.get('symptoms')

    if not provided_symptoms:
        return jsonify({"error": "No symptoms provided"}), 400


    expected_features = model.feature_names_in_


    full_symptom_data = {feature: 0 for feature in expected_features}

    for symptom in provided_symptoms:
        if symptom in full_symptom_data:
            full_symptom_data[symptom] = 1
        else:
            return jsonify({"error": f"Unknown symptom: {symptom}"}), 400

    input_df = pd.DataFrame([full_symptom_data])

    try:
        prediction = model.predict(input_df)
        predicted_disease = prediction[0]
        return jsonify({"predicted_disease": predicted_disease})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
