import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import pickle

# Wczytanie modelii
regression_model = load_model("models/regression_model.h5")
classification_model = load_model("models/classification_model.h5")

# Wczytanie kolumn
with open("processed_columns.txt", "r") as f:
    processed_columns = [line.strip() for line in f.readlines()]

# Wczytanie skalera
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

sample_data = {
    'BHK': 20,
    'Size': 1100,
    'Bathroom': 20,
    'Floor': 20,
    'Area Type_Super Area': 1,
    'Area Locality_Bandel': 1,
    'City_Kolkata': 1,#City_Chennai, City_Delhi, City_Hyderabad, City_Kolkata, City_Mumbai
    'Furnishing Status_Semi-Furnished': 1,
    'Tenant Preferred_Bachelors/Family': 1
}

# Dopasowanie probki
sample_df = pd.DataFrame([sample_data])
sample_df = sample_df.reindex(columns=processed_columns, fill_value=0)

# Różnice między kolumnami
missing_columns = set(processed_columns) - set(sample_df.columns)
extra_columns = set(sample_df.columns) - set(processed_columns)
print(f"Brakujące kolumny w próbce: {missing_columns}")
print(f"Dodatkowe kolumny w próbce: {extra_columns}")

# Skalowanie danych
scaled_sample = scaler.transform(sample_df)

# Predykcja dla regresji
predicted_rent = regression_model.predict(scaled_sample)
print(f"Przewidywana cena wynajmu: {predicted_rent[0][0]:.2f}")

# Predykcja dla klasyfikacji
predicted_class_probs = classification_model.predict(scaled_sample)
predicted_class = (predicted_class_probs > 0.5).astype(int)
print(f"Przewidywana klasa wynajmu: {'Drogie' if predicted_class[0][0] == 1 else 'Tanie'}")
