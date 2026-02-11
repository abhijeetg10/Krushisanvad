
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

results = {
    "Crop": {},
    "Fertilizer": {}
}

# ================= CROP =================
try:
    crop_df = pd.read_csv("Crop_recommendation.csv")
    X_crop = crop_df.drop("label", axis=1)
    y_crop = crop_df["label"]
    
    crop_encoder = LabelEncoder()
    y_crop_enc = crop_encoder.fit_transform(y_crop)
    
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_crop, y_crop_enc, test_size=0.2, random_state=42, stratify=y_crop_enc
    )
    
    crop_scaler = StandardScaler()
    Xc_train_s = crop_scaler.fit_transform(Xc_train)
    Xc_test_s = crop_scaler.transform(Xc_test)
    
    crop_models = {
        "RF": RandomForestClassifier(n_estimators=100, random_state=42), 
        "GB": GradientBoostingClassifier(),
        "DT": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "NB": GaussianNB(),
        "LR": LogisticRegression(max_iter=500)
    }
    
    for name, model in crop_models.items():
        if name in ["KNN", "LR"]:
            model.fit(Xc_train_s, yc_train)
            preds = model.predict(Xc_test_s)
        else:
            model.fit(Xc_train, yc_train)
            preds = model.predict(Xc_test)
        results["Crop"][name] = accuracy_score(yc_test, preds) * 100

    # Hybrid
    crop_hybrid = StackingClassifier(
        estimators=[("rf", crop_models["RF"]), ("gb", crop_models["GB"]), ("dt", crop_models["DT"])],
        final_estimator=LogisticRegression(max_iter=300),
        n_jobs=1
    )
    crop_hybrid.fit(Xc_train, yc_train)
    results["Crop"]["HYBRID"] = accuracy_score(yc_test, crop_hybrid.predict(Xc_test)) * 100

except Exception as e:
    results["Crop"]["Error"] = str(e)

# ================= FERTILIZER =================
try:
    fert_df = pd.read_csv("Fertilizer Prediction.csv")
    
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
    
    fert_models = {
        "RF": RandomForestClassifier(n_estimators=100, random_state=42),
        "GB": GradientBoostingClassifier(),
        "DT": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "NB": GaussianNB(),
        "LR": LogisticRegression(max_iter=300)
    }
    
    for name, model in fert_models.items():
        if name in ["KNN", "LR"]:
            model.fit(Xf_train_s, yf_train)
            preds = model.predict(Xf_test_s)
        else:
            model.fit(Xf_train, yf_train)
            preds = model.predict(Xf_test)
        results["Fertilizer"][name] = accuracy_score(yf_test, preds) * 100

    # Hybrid
    fert_hybrid = StackingClassifier(
        estimators=[("rf", fert_models["RF"]), ("gb", fert_models["GB"]), ("dt", fert_models["DT"])],
        final_estimator=LogisticRegression(max_iter=300),
        n_jobs=1
    )
    fert_hybrid.fit(Xf_train, yf_train)
    results["Fertilizer"]["HYBRID"] = accuracy_score(yf_test, fert_hybrid.predict(Xf_test)) * 100

except Exception as e:
    results["Fertilizer"]["Error"] = str(e)

print("JSON_START")
print(json.dumps(results, indent=2))
print("JSON_END")
