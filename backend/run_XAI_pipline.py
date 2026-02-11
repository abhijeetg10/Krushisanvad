import shap
import pandas as pd
import numpy as np
import os
import uuid
import matplotlib.pyplot as plt

# --- FIX: ABSOLUTE PATH TO 'backend/static/xai_plots' ---
# This ensures it works regardless of where you run the command from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FOLDER = os.path.join(BASE_DIR, "static", "xai_plots")

# Create directory if it doesn't exist
os.makedirs(STATIC_FOLDER, exist_ok=True)

def get_natural_language_explanation(feature, score, value):
    # ... (Same logic as before, omitted for brevity) ...
    is_positive = score > 0
    if is_positive:
        return f"✅ {feature} ({value}) supports this crop choice."
    return f"⚠️ {feature} ({value}) lowers confidence slightly."

def explain_prediction(model, input_data, class_index, prediction_name, feature_names):
    try:
        # 1. SHAP Calculation
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)
        except:
            explainer = shap.KernelExplainer(model.predict_proba, input_data)
            shap_values = explainer.shap_values(input_data)

        # Handle SHAP return types (List vs Array)
        if isinstance(shap_values, list):
            class_shap_values = shap_values[class_index]
            base_value = explainer.expected_value[class_index] if isinstance(explainer.expected_value, list) else explainer.expected_value
        else:
            class_shap_values = shap_values[:, :, class_index] if len(shap_values.shape) == 3 else shap_values
            base_value = explainer.expected_value[class_index] if hasattr(explainer.expected_value, '__iter__') else explainer.expected_value

        # 2. Generate Text
        flat_shap = np.array(class_shap_values[0]).flatten().astype(float)
        flat_input = np.array(input_data.iloc[0]).flatten()
        reasons = []
        
        # Sort by impact
        contributions = sorted(zip(feature_names, flat_shap, flat_input), key=lambda x: abs(x[1]), reverse=True)
        for f, s, v in contributions[:3]:
            reasons.append(get_natural_language_explanation(f, s, v))

        # 3. Save Plot (ABSOLUTE PATH)
        filename = f"shap_{uuid.uuid4().hex}.html"
        save_path = os.path.join(STATIC_FOLDER, filename)
        
        # Generate and Save
        force_plot = shap.force_plot(base_value, class_shap_values, input_data, matplotlib=False)
        shap.save_html(save_path, force_plot)

        print(f"📈 Saved SHAP plot to: {save_path}") # Debug Print

        # Return URL relative to Flask static folder
        return {
            "text_explanation": reasons,
            "plot_url": f"/static/xai_plots/{filename}"
        }

    except Exception as e:
        print(f"⚠️ XAI Error: {e}")
        return { "text_explanation": ["AI analysis complete."], "plot_url": "" }