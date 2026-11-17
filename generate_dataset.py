import pandas as pd
import random

random.seed(42)

crop_data = [
    ("Rice", "Clay", 28, 1200, 80),
    ("Wheat", "Loamy", 22, 600, 60),
    ("Cotton", "Black", 30, 800, 50),
    ("Sugarcane", "Clay", 32, 1500, 75),
    ("Maize", "Loamy", 25, 700, 65),
    ("Groundnut", "Sandy", 27, 500, 55),
    ("Millets", "Sandy", 29, 400, 45),
    ("Potato", "Loamy", 20, 650, 70)
]

rows = []

for crop, soil, temperature, rainfall, humidity in crop_data:

    for i in range(15):

        rows.append({
            "Crop": crop,
            "Soil_Type": soil,
            "Temperature": round(
                temperature + random.uniform(-2.5, 2.5), 1
            ),
            "Rainfall": round(
                rainfall + random.uniform(-120, 120), 1
            ),
            "Humidity": round(
                humidity + random.uniform(-7, 7), 1
            )
        })

df = pd.DataFrame(rows)

df.insert(0, "id", range(1, len(df) + 1))

df.to_csv("crops.csv", index=False)

print("SMART AGRICULTURE DATASET CREATED")
print("Total Records:", len(df))
print("Total Crops:", df["Crop"].nunique())

print("\nCrop Distribution:")
print(df["Crop"].value_counts())

print("\nDataset saved successfully as crops.csv")