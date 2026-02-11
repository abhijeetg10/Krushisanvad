from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os

# Import custom modules
from pdf_parser import extract_data_from_pdf
from utils.crop_predictor import crop_model, crop_encoder
from utils.fertilizer_predictor import get_fertilizer_recommendation
from run_XAI_pipline import explain_prediction

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/*": {"origins": "*"}})

# --- CROP MAPPING ---
CROP_MAPPING = {
    'rice': 'Paddy', 'maize': 'Maize', 'chickpea': 'Pulses', 'kidneybeans': 'Pulses',
    'pigeonpeas': 'Pulses', 'mothbeans': 'Pulses', 'mungbean': 'Pulses', 'blackgram': 'Pulses',
    'lentil': 'Pulses', 'pomegranate': 'Pomegranate', 'banana': 'Sugarcane', 'mango': 'Sugarcane',
    'grapes': 'Sugarcane', 'watermelon': 'Tobacco', 'muskmelon': 'Tobacco', 'apple': 'Tobacco',
    'orange': 'Tobacco', 'papaya': 'Tobacco', 'coconut': 'Tobacco', 'cotton': 'Cotton',
    'jute': 'Cotton', 'coffee': 'Tobacco'
}

def get_weather_data():
    return { 'Temperature': 25.5, 'Humidity': 82.0, 'Rainfall': 180.0, 'Moisture': 45.0 }

def clean_nans(value):
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value): return None
    if isinstance(value, dict): return {k: clean_nans(v) for k, v in value.items()}
    if isinstance(value, list): return [clean_nans(v) for v in value]
    return value

# --- 1. SUSTAINABILITY & GOVERNANCE ENGINE (Restored) ---
def check_governance(crop_name, rainfall, ph, organic_carbon):
    alerts = []
    
    # Water Audit
    water_intensive = ['rice', 'paddy', 'sugarcane', 'jute', 'banana']
    if crop_name.lower() in water_intensive and rainfall < 700:
        alerts.append("⚠️ WATER AUDIT: High Risk. This crop requires heavy irrigation (>700mm). Growing it here may deplete groundwater.")
    elif crop_name.lower() in water_intensive and rainfall >= 700:
        alerts.append("✅ WATER AUDIT: Pass. Rainfall is sufficient for this water-intensive crop.")
    else:
        alerts.append("✅ WATER AUDIT: Pass. This crop has a low water footprint.")

    # Soil Health Audit
    heavy_feeders = ['cotton', 'sugarcane', 'maize', 'tobacco']
    if crop_name.lower() in heavy_feeders and organic_carbon < 0.5:
        alerts.append("⚠️ SOIL AUDIT: Critical. Organic Carbon is low (<0.5%). Growing this heavy feeder will degrade soil health.")
    
    # pH Audit
    if ph < 5.5 and crop_name.lower() not in ['tea', 'coffee', 'rice']:
        alerts.append("⚠️ PH AUDIT: Soil is too acidic. Liming is mandatory.")

    return alerts

# --- 2. FERTILIZER EXPLANATION LOGIC ---
def generate_fertilizer_explanation(fert_name, n, p, k, crop_name):
    reasons = []
    fert_lower = fert_name.lower()
    has_N = any(x in fert_lower for x in ['urea', 'n', 'dap', 'npk'])
    has_P = any(x in fert_lower for x in ['dap', 'p', 'npk', 'ssp'])
    
    if n < 80:
        if has_N: reasons.append(f"⚠️ Nitrogen is low ({n} kg/ha). {fert_name} restores greenery.")
        else: reasons.append(f"⚠️ Nitrogen is low. Consider adding Urea alongside {fert_name}.")
    elif n > 150: reasons.append(f"✅ Nitrogen is sufficient.")

    if p < 40:
        if has_P: reasons.append(f"⚠️ Phosphorus is low ({p} kg/ha). {fert_name} aids root development.")
        else: reasons.append(f"⚠️ Phosphorus is low. Consider adding SSP.")
    
    if not reasons:
        reasons.append(f"✅ {fert_name} is well-suited for {crop_name}.")
    return reasons

@app.route('/analyze', methods=['POST'])
def analyze_soil():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    extracted_data = extract_data_from_pdf(file)
    if not extracted_data:
        return jsonify({"error": "Failed to extract data"}), 400

    weather = get_weather_data()
    full_data = {**extracted_data, **weather}

    try:
        # Crop Prediction
        crop_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        crop_input = pd.DataFrame([[
            full_data['N'], full_data['P'], full_data['K'],
            full_data['Temperature'], full_data['Humidity'],
            full_data['ph'], full_data['Rainfall']
        ]], columns=crop_features)

        crop_probs = crop_model.predict_proba(crop_input)[0]
        crop_idx = np.argmax(crop_probs)
        predicted_crop = crop_encoder.inverse_transform([crop_idx])[0]
        crop_xai = explain_prediction(crop_model, crop_input, crop_idx, predicted_crop, crop_features)

        # Fertilizer Prediction
        mapped_crop = CROP_MAPPING.get(predicted_crop.lower(), 'Maize')
        soil_type = full_data.get('Soil Type', 'Clayey') 
        fert_input_data = full_data.copy()
        fert_input_data['Crop Type'] = mapped_crop
        fert_input_data['Soil Type'] = soil_type
        fert_name, fert_conf = get_fertilizer_recommendation(fert_input_data)
        
        fert_reasons = generate_fertilizer_explanation(
            fert_name, full_data['N'], full_data['P'], full_data['K'], predicted_crop
        )

        # --- 3. RUN GOVERNANCE CHECK (Added Back) ---
        governance_report = check_governance(
            predicted_crop, 
            full_data['Rainfall'], 
            full_data['ph'], 
            full_data.get('OC', 0.5)
        )

        response = {
            "crop": {
                "prediction": predicted_crop,
                "confidence": round(float(crop_probs[crop_idx]) * 100, 2),
                "reasons": crop_xai['text_explanation'],
                "plot": crop_xai['plot_url']
            },
            "fertilizer": {
                "prediction": fert_name,
                "confidence": fert_conf,
                "reasons": fert_reasons
            },
            "governance": {  # This sends the data to the frontend
                "alerts": governance_report
            },
            "soil_data": extracted_data
        }

        return jsonify(clean_nans(response))

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('static/xai_plots'):
        os.makedirs('static/xai_plots')
    app.run(debug=True, port=5000)