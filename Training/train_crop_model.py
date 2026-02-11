# FILE: Training/train_models.py

import pandas as pd
import numpy as np
import joblib
import os

# =========================
# SKLEARN IMPORTS
# =========================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# ============================================================
# CONFIGURATION & PATHS
# ============================================================
# We save models directly to the backend folder so app.py can find them
MODEL_SAVE_DIR = "../backend/models/"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

print(f"Models will be saved to: {MODEL_SAVE_DIR}")

# ============================================================
# ===================== STAGE 1 ===============================
# ================ CROP RECOMMENDATION MODEL ==================
# ============================================================

print("\n================ CROP MODEL TRAINING ================\n")

# -------- LOAD CROP DATASET --------
# Ensure the CSV is in the 'Training' folder or update path
crop_df = pd.read_csv("Crop_recommendation.csv")

X_crop = crop_df.drop("label", axis=1)
y_crop = crop_df["label"]

# Encode Target
crop_encoder = LabelEncoder()
y_crop_enc = crop_encoder.fit_transform(y_crop)

# Split
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_crop, y_crop_enc, test_size=0.2, random_state=42, stratify=y_crop_enc
)

# Scale
crop_scaler = StandardScaler()
Xc_train_s = crop_scaler.fit_transform(Xc_train)
Xc_test_s = crop_scaler.transform(Xc_test)

# -------- MODELS --------
crop_models = {
    "RF": RandomForestClassifier(n_estimators=200, random_state=42), # Reduced slightly for speed
    "GB": GradientBoostingClassifier(),
    "DT": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "NB": GaussianNB(),
    "LR": LogisticRegression(max_iter=500)
}

crop_results = {}

# -------- TRAIN --------
print("Training Base Crop Models...")
for name, model in crop_models.items():
    if name in ["KNN", "LR"]:
        model.fit(Xc_train_s, yc_train)
        preds = model.predict(Xc_test_s)
    else:
        model.fit(Xc_train, yc_train)
        preds = model.predict(Xc_test)
    
    acc = accuracy_score(yc_test, preds)
    crop_results[name] = acc
    print(f"   {name}: {acc*100:.2f}%")

# -------- HYBRID MODEL --------
print("Training Crop Hybrid Model...")
crop_hybrid = StackingClassifier(
    estimators=[
        ("rf", crop_models["RF"]),
        ("gb", crop_models["GB"]),
        ("dt", crop_models["DT"])
    ],
    final_estimator=LogisticRegression(max_iter=300),
    n_jobs=-1
)

crop_hybrid.fit(Xc_train, yc_train)
crop_h_acc = accuracy_score(yc_test, crop_hybrid.predict(Xc_test))
crop_results["HYBRID"] = crop_h_acc
print(f"   HYBRID: {crop_h_acc*100:.2f}%")

# -------- SAVE BEST CROP MODEL --------
best_crop_name = max(crop_results, key=crop_results.get)
best_crop_model = crop_hybrid if best_crop_name == "HYBRID" else crop_models[best_crop_name]

print(f"\n🏆 BEST CROP MODEL: {best_crop_name}")

# Save Crop Artifacts
joblib.dump(best_crop_model, os.path.join(MODEL_SAVE_DIR, "best_crop_model.pkl"))
joblib.dump(crop_encoder, os.path.join(MODEL_SAVE_DIR, "crop_encoder.pkl"))
joblib.dump(crop_scaler, os.path.join(MODEL_SAVE_DIR, "crop_scaler.pkl"))
print("✅ Crop models saved.")


# ============================================================
# ===================== STAGE 2 ===============================
# ============ FERTILIZER RECOMMENDATION MODEL ================
# ============================================================

print("\n================ FERTILIZER MODEL TRAINING ================\n")

# -------- LOAD DATASET --------
fert_df = pd.read_csv("Fertilizer Prediction.csv")

# -------- ENCODE CATEGORICAL FEATURES --------
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

fert_df["Soil Type"] = le_soil.fit_transform(fert_df["Soil Type"])
fert_df["Crop Type"] = le_crop.fit_transform(fert_df["Crop Type"])
fert_df["Fertilizer Name"] = le_fert.fit_transform(fert_df["Fertilizer Name"])

X_fert = fert_df.drop("Fertilizer Name", axis=1)
y_fert = fert_df["Fertilizer Name"]

Xf_train, Xf_test, yf_train, yf_test = train_test_split(
    X_fert, y_fert, test_size=0.2, random_state=42, stratify=y_fert
)

fert_scaler = StandardScaler()
Xf_train_s = fert_scaler.fit_transform(Xf_train)
Xf_test_s = fert_scaler.transform(Xf_test)

# -------- MODELS --------
fert_models = {
    "RF": RandomForestClassifier(n_estimators=200, random_state=42),
    "GB": GradientBoostingClassifier(),
    "DT": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "NB": GaussianNB(),
    "LR": LogisticRegression(max_iter=300)
}

fert_results = {}

print("Training Base Fertilizer Models...")
for name, model in fert_models.items():
    if name in ["KNN", "LR"]:
        model.fit(Xf_train_s, yf_train)
        preds = model.predict(Xf_test_s)
    else:
        model.fit(Xf_train, yf_train)
        preds = model.predict(Xf_test)
    
    acc = accuracy_score(yf_test, preds)
    fert_results[name] = acc
    print(f"   {name}: {acc*100:.2f}%")

# -------- HYBRID MODEL --------
print("Training Fertilizer Hybrid Model...")
fert_hybrid = StackingClassifier(
    estimators=[
        ("rf", fert_models["RF"]),
        ("gb", fert_models["GB"]),
        ("dt", fert_models["DT"])
    ],
    final_estimator=LogisticRegression(max_iter=300),
    n_jobs=-1
)

fert_hybrid.fit(Xf_train, yf_train)
fert_h_acc = accuracy_score(yf_test, fert_hybrid.predict(Xf_test))
fert_results["HYBRID"] = fert_h_acc
print(f"   HYBRID: {fert_h_acc*100:.2f}%")

# -------- SAVE BEST FERT MODEL --------
best_fert_name = max(fert_results, key=fert_results.get)
best_fert_model = fert_hybrid if best_fert_name == "HYBRID" else fert_models[best_fert_name]

print(f"\n🏆 BEST FERTILIZER MODEL: {best_fert_name}")

# Save Fertilizer Artifacts
joblib.dump(best_fert_model, os.path.join(MODEL_SAVE_DIR, "best_fertilizer_model.pkl"))
joblib.dump(le_soil, os.path.join(MODEL_SAVE_DIR, "le_soil.pkl"))
joblib.dump(le_crop, os.path.join(MODEL_SAVE_DIR, "le_crop.pkl"))
joblib.dump(le_fert, os.path.join(MODEL_SAVE_DIR, "le_fertilizer.pkl"))
joblib.dump(fert_scaler, os.path.join(MODEL_SAVE_DIR, "fertilizer_scaler.pkl"))

print("✅ Fertilizer models saved.")
print("\nALL TRAINING COMPLETE. ARTIFACTS READY IN 'backend/models/'")