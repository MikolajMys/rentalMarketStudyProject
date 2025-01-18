import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Wczytanie zapisanych modeli
quantiles = [0.1, 0.5, 0.9]
models = {q: load_model(f"models/quantile_model_{int(q*100)}.h5", compile=False) for q in quantiles}

# Wczytanie zapisanych przetworzonych kolumn
with open("processed_columns.txt", "r") as f:
    processed_columns = [line.strip() for line in f.readlines()]

# Wczytanie skalera
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Przygotowanie przykładowej próbki danych
sample_data = {
    'BHK': 2,
    'Size': 1100,
    'Bathroom': 2,
    'Floor': 3,
    'Area Type_Super Area': 1,
    'Area Locality_Bandel': 1,
    'City_Kolkata': 1,
    'Furnishing Status_Unfurnished': 1,
    'Tenant Preferred_Bachelors/Family': 1
}

# Przekształcenie próbki do odpowiedniego formatu
sample_df = pd.DataFrame([sample_data])
sample_df = sample_df.reindex(columns=processed_columns, fill_value=0)

# Skalowanie próbki
scaled_sample = scaler.transform(sample_df)

# Predykcje dla każdego modelu kwantylowego
predictions = {}
for q, model in models.items():
    pred = model.predict(scaled_sample)
    predictions[f"Quantile_{q}"] = pred[0][0]

print("Predykcje dla różnych kwantyli:")
print(f"Minimalna cena w 10% najbardziej pesymistycznych scenariuszy: {predictions['Quantile_0.1']:.2f}")
print(f"Mediana, najbardziej prawdopodobna cena: {predictions['Quantile_0.5']:.2f}")
print(f"Maksymalna cena w 10% najbardziej optymistycznych scenariuszy: {predictions['Quantile_0.9']:.2f}")
