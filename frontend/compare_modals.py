import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 1. Load Data
# Ensure you have your dataset file 'Crop_recommendation.csv' in the backend folder
try:
    df = pd.read_csv('Crop_recommendation.csv')
    print("✅ Dataset Loaded")
except:
    print("❌ Error: 'Crop_recommendation.csv' not found. Please add it to the backend folder.")
    exit()

X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define Models to Compare
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM (Support Vector)": SVC(),
    "Random Forest (Ours)": RandomForestClassifier(n_estimators=20, random_state=42)
}

# 4. Run Comparison
print("\n📊 MODEL COMPARISON RESULTS (For Research Paper)\n" + "="*50)
print(f"{'Model Name':<25} | {'Accuracy':<10}")
print("-" * 40)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name:<25} | {acc*100:.2f}%")

print("="*50)
print("\n✅ Copy these results into the 'Methodology & Results' section of your paper.")