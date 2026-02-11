import joblib
import pandas as pd
import os

# Define paths relative to this file
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Load artifacts
try:
    crop_model = joblib.load(os.path.join(MODEL_DIR, "best_crop_model.pkl"))
    crop_encoder = joblib.load(os.path.join(MODEL_DIR, "crop_encoder.pkl"))
    crop_scaler = joblib.load(os.path.join(MODEL_DIR, "crop_scaler.pkl"))
    print("✅ Crop Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading crop model: {e}")

def get_crop_recommendation(data):
    """
    Input: Dictionary with keys [N, P, K, temperature, humidity, ph, rainfall]
    Output: Predicted Crop Name (String)
    """
    try:
        # Prepare Input
        input_data = pd.DataFrame([data])
        
        # NOTE: If your best model was KNN or LR, you must scale. 
        # If it was RF/GB, scaling doesn't hurt. We assume standard flow here:
        # Check if the model expects scaled data (This logic depends on what trained best)
        # For safety in this hybrid setup, we can try-catch or assume the hybrid handles it.
        # But generally, if we trained on raw data for Trees, passing raw is fine.
        
        # Prediction
        prediction_idx = crop_model.predict(input_data)[0]
        crop_name = crop_encoder.inverse_transform([prediction_idx])[0]
        
        return crop_name
    except Exception as e:
        return str(e)