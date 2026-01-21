"""
House Price Prediction Web Application
Flask-based web GUI for predicting house prices
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model and preprocessing objects
MODEL_PATH = 'model/house_price_model.pkl'
SCALER_PATH = 'model/scaler.pkl'
ENCODER_PATH = 'model/label_encoder.pkl'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("✅ Model and preprocessing objects loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None
    label_encoder = None

# Get list of neighborhoods from the encoder
if label_encoder:
    neighborhoods = sorted(label_encoder.classes_.tolist())
else:
    neighborhoods = []


@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', neighborhoods=neighborhoods)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None or scaler is None or label_encoder is None:
            return jsonify({
                'error': 'Model not loaded. Please ensure model files are in the model folder.'
            }), 500

        # Get form data
        overall_qual = int(request.form['overall_qual'])
        gr_liv_area = float(request.form['gr_liv_area'])
        total_bsmt_sf = float(request.form['total_bsmt_sf'])
        garage_cars = int(request.form['garage_cars'])
        year_built = int(request.form['year_built'])
        neighborhood = request.form['neighborhood']

        # Validate inputs
        if overall_qual < 1 or overall_qual > 10:
            return jsonify({'error': 'Overall Quality must be between 1 and 10'}), 400

        if gr_liv_area < 0 or total_bsmt_sf < 0:
            return jsonify({'error': 'Area values cannot be negative'}), 400

        if garage_cars < 0 or garage_cars > 5:
            return jsonify({'error': 'Garage Cars must be between 0 and 5'}), 400

        if year_built < 1800 or year_built > 2026:
            return jsonify({'error': 'Year Built must be between 1800 and 2026'}), 400

        # Create input dataframe
        input_data = pd.DataFrame({
            'OverallQual': [overall_qual],
            'GrLivArea': [gr_liv_area],
            'TotalBsmtSF': [total_bsmt_sf],
            'GarageCars': [garage_cars],
            'YearBuilt': [year_built],
            'Neighborhood': [neighborhood]
        })

        # Encode neighborhood
        try:
            neighborhood_encoded = label_encoder.transform([neighborhood])[0]
        except ValueError:
            return jsonify({'error': f'Invalid neighborhood: {neighborhood}'}), 400

        # Prepare features
        features = np.array([[
            overall_qual,
            gr_liv_area,
            total_bsmt_sf,
            garage_cars,
            year_built,
            neighborhood_encoded
        ]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        # Format the prediction
        predicted_price = f"${prediction:,.2f}"

        return jsonify({
            'success': True,
            'predicted_price': predicted_price,
            'prediction_value': float(prediction),
            'input_data': {
                'Overall Quality': overall_qual,
                'Living Area (sq ft)': gr_liv_area,
                'Basement Area (sq ft)': total_bsmt_sf,
                'Garage Cars': garage_cars,
                'Year Built': year_built,
                'Neighborhood': neighborhood
            }
        })

    except ValueError as e:
        return jsonify({'error': f'Invalid input values: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'encoder_loaded': label_encoder is not None
    }
    return jsonify(status)


if __name__ == '__main__':
    # For local development
    app.run(debug=True, host='0.0.0.0', port=5000)
