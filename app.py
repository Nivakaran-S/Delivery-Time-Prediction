from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np 
import pandas as pd 
from src.logging.logger import logging
import os 

app = Flask(__name__, template_folder='templates', static_folder='static')

TOP_12_FEATURES = [
    'multiple_deliveries', 'Road_traffic_density', 'Vehicle_condition', 'Delivery_person_Ratings',
    'distance_deliveries', 'Weather_conditions', 'Festival', 'distance_traffic', 'distance',
    'Delivery_person_Age', 'prep_traffic', 'City'
]

# Mapping dictionaries for dropdowns
WEATHER_CONDITIONS = {'Sunny': 0, 'Cloudy': 1, 'Fog': 2, 'Sandstorms': 3, 'Stormy': 4, 'Windy': 5}
TRAFFIC_DENSITY = {'Low': 0, 'Medium': 1, 'High': 2, 'Jam': 3}
VEHICLE_CONDITION = {'Poor': 0, 'Good': 1, 'Excellent': 2}
FESTIVAL = {'No': 0, 'Yes': 1}
CITY = {'Urban': 0, 'Semi-Urban': 1, 'Metropolitan': 2}

# Load model and scaler
try:
    model = joblib.load('final_model/model.pkl')  # This is an instance of DeliveryPredictionModel
    

    preprocessor = joblib.load('final_model/preprocessor.pkl')
    scaler = preprocessor['scaler']
    expected_features = preprocessor['feature_names']  # Ensure the same feature order

    logging.info("Model and scaler loaded successfully")
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}")
    raise e

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        logging.info(f"Received form data: {data}")

        def validate_dropdown(value, mapping, field_name):
            if value not in mapping:
                raise ValueError(f"Invalid value '{value}' for {field_name}. Valid options: {list(mapping.keys())}")
            return mapping[value]
        
        input_data = {
            'multiple_deliveries': float(data['multiple_deliveries']),
            'Road_traffic_density': validate_dropdown(data['Road_traffic_density'], TRAFFIC_DENSITY, 'Road_traffic_density'),
            'Vehicle_condition': validate_dropdown(data['Vehicle_condition'], VEHICLE_CONDITION, 'Vehicle_condition'),
            'Delivery_person_Ratings': float(data['Delivery_person_Ratings']),
            'distance_deliveries': float(data['distance_deliveries']),
            'Weather_conditions': validate_dropdown(data['Weather_conditions'], WEATHER_CONDITIONS, 'Weather_conditions'),
            'Festival': validate_dropdown(data['Festival'], FESTIVAL, 'Festival'),
            'distance_traffic': float(data['distance_traffic']),
            'distance': float(data['distance']),
            'Delivery_person_Age': float(data['Delivery_person_Age']),
            'prep_traffic': float(data['prep_traffic']),
            'City': validate_dropdown(data['City'], CITY, 'City')
        }

        df = pd.DataFrame([input_data])
        df = df[expected_features]  # enforce correct order

        features_scaled = scaler.transform(df)
        prediction = float(model.predict(features_scaled)[0])

        logging.info(f"Prediction: {prediction}")
        return jsonify({'prediction': round(prediction, 2)})

    except KeyError as e:
        logging.error(f"Missing field: {e}")
        return jsonify({'error': f"Missing required field: {e}"}), 400
    except ValueError as e:
        logging.error(f"Invalid input: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
