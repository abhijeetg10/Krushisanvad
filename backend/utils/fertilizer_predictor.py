import joblib
import pandas as pd
import os
import numpy as np

# Define Model Paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Load Models
try:
    fert_model = joblib.load(os.path.join(MODEL_DIR, "best_fertilizer_model.pkl"))
    le_soil = joblib.load(os.path.join(MODEL_DIR, "le_soil.pkl"))
    le_crop = joblib.load(os.path.join(MODEL_DIR, "le_crop.pkl"))
    le_fert = joblib.load(os.path.join(MODEL_DIR, "le_fertilizer.pkl"))
    fert_scaler = joblib.load(os.path.join(MODEL_DIR, "fertilizer_scaler.pkl"))
    print("✅ Fertilizer Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading fertilizer model: {e}")

def get_fertilizer_recommendation(data):
    """
    Input: Dictionary with clean keys [Temperature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Phosphorous, Potassium]
    """
    try:
        # 1. Encode Categorical Values (Soil & Crop)
        try:
            # We use .get() to handle capitalization differences safely
            soil_input = data.get('Soil Type', 'Clayey')
            crop_input = data.get('Crop Type', 'Maize')
            
            soil_encoded = le_soil.transform([soil_input])[0]
            crop_encoded = le_crop.transform([crop_input])[0]
        except Exception as e:
            # Fallback for unknown categories
            print(f"⚠️ Encoder warning: {e}. Using defaults.")
            soil_encoded = 0
            crop_encoded = 0

        # 2. Create DataFrame with STANDARD names first
        input_dict = {
            'Temperature': data.get('Temperature', 25),
            'Humidity': data.get('Humidity', 60),
            'Moisture': data.get('Moisture', 40),
            'Soil Type': soil_encoded,
            'Crop Type': crop_encoded,
            'Nitrogen': data.get('Nitrogen', 50) or data.get('N', 50),
            'Potassium': data.get('Potassium', 40) or data.get('K', 40),
            'Phosphorus': data.get('Phosphorous', 30) or data.get('P', 30) # Use correct spelling here initially
        }
        
        df = pd.DataFrame([input_dict])

        # 3. --- FIX: RENAME COLUMNS TO MATCH MODEL TYPOS ---
        # The error said: "Feature names seen at fit time, yet now missing: - Phosphorous"
        # So we must rename our correct "Phosphorus" to the model's misspelled "Phosphorous".
        df.rename(columns={
            'Temperature': 'Temparature',   # Model typo 1
            'Humidity': 'Humidity ',        # Model typo 2 (trailing space)
            'Phosphorus': 'Phosphorous'     # Model typo 3 (extra 'o')
        }, inplace=True)

        # 4. Reorder Columns
        # Ensure columns are in the EXACT order the model expects. 
        try:
            if hasattr(fert_model, 'feature_names_in_'):
                df = df[fert_model.feature_names_in_]
            elif hasattr(fert_model, 'n_features_in_'):
                # Fallback to standard order if we can't read names
                # Note: We use the RENAMED (typo) versions here
                expected_cols = ['Temparature', 'Humidity ', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
                
                # Filter to keep only columns that actually exist
                present_cols = [c for c in expected_cols if c in df.columns]
                df = df[present_cols]
        except:
            pass # Use DF as is if reordering fails

        # 5. Predict
        probs = fert_model.predict_proba(df)[0]
        max_prob_idx = np.argmax(probs)
        
        predicted_fert = le_fert.inverse_transform([max_prob_idx])[0]
        confidence = round(probs[max_prob_idx] * 100, 2)
        
        return predicted_fert, confidence

    except Exception as e:
        print(f"❌ Fert Prediction Error: {e}")
        return f"Error: {str(e)}", 0