import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("crops.csv")

X = data[
    ["Soil_Type", "Temperature", "Rainfall", "Humidity"]
]

y = data["Crop"]

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

model.fit(X, y)

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, "crop_model.pkl")

print("================================")
print("SMART AGRICULTURE MODEL")
print("================================")
print("Model trained successfully!")
print("Model saved as crop_model.pkl")
print("================================")