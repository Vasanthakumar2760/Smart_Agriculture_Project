import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("crops.csv")

print("SMART AGRICULTURE - MODEL EVALUATION")
print("-----------------------------------")

print("Total Records:", len(data))
print("Total Crops:", data["Crop"].nunique())

# ==========================================
# INPUT AND OUTPUT
# ==========================================

X = data[
    [
        "Soil_Type",
        "Temperature",
        "Rainfall",
        "Humidity"
    ]
]

y = data["Crop"]

# ==========================================
# TRAINING AND TESTING SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# ==========================================
# PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "soil",
            OneHotEncoder(handle_unknown="ignore"),
            ["Soil_Type"]
        )
    ],
    remainder="passthrough"
)

# ==========================================
# RANDOM FOREST MODEL
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

# ==========================================
# TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# MODEL ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY")
print("--------------")

print(f"Accuracy: {accuracy * 100:.2f}%")

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

print("\nCONFUSION MATRIX")
print("----------------")

print(cm)

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nCLASSIFICATION REPORT")
print("--------------------")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nEvaluation Completed Successfully!")