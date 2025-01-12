import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score, roc_auc_score
from tensorflow.keras.utils import to_categorical

# Wczytanie danych
df = pd.read_csv("data/House_Rent_Dataset.csv")

# **1. Analiza wstępna**
print("Podstawowe informacje o danych:")
print(df.info())
print("\nPodstawowe statystyki:")
print(df.describe())

sns.histplot(df['Rent'], kde=True)
plt.title("Rozkład cen wynajmu")
plt.show()

# **2. Obsługa braków danych**
print("\nBraki danych w kolumnach:")
print(df.isnull().sum())

for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col] = df[col].fillna(df[col].median())

# Usunięcie kolumn z >50% braków
threshold = 0.5
missing_percentage = df.isnull().mean()
columns_to_drop = missing_percentage[missing_percentage > threshold].index
df.drop(columns=columns_to_drop, inplace=True)
print("\nKolumny usunięte z powodu dużej liczby braków:", columns_to_drop.tolist())

# **3. Przetwarzanie kolumny Floor**
def preprocess_floor(floor_value):
    if 'Ground' in floor_value:
        return 0
    try:
        return int(floor_value.split(' ')[0])
    except ValueError:
        return np.nan

df['Floor'] = df['Floor'].apply(preprocess_floor)
df['Floor'] = df['Floor'].fillna(df['Floor'].median())

# **4. Usuwanie odstających wartości**
sns.boxplot(x=df['Rent'])
plt.title("Wykrywanie wartości odstających dla cen wynajmu")
plt.show()

Q1 = df['Rent'].quantile(0.25)
Q3 = df['Rent'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['Rent'] >= lower_bound) & (df['Rent'] <= upper_bound)]

# **5. Kodowanie zmiennych kategorycznych**
df = pd.get_dummies(df, columns=['Area Locality', 'Area Type', 'City', 'Furnishing Status', 'Tenant Preferred'], drop_first=True)

# Usunięcie kolumny 'Point of Contact' (nienumeryczna i nieistotna dla analizy)
df.drop(columns=['Point of Contact'], inplace=True)

# Weryfikacja, czy wszystkie kolumny są liczbowe
non_numeric_columns = df.select_dtypes(include=['object']).columns
if len(non_numeric_columns) > 0:
    print(f"\nUsuwane kolumny nienumeryczne przed standaryzacją: {non_numeric_columns.tolist()}")
    df.drop(columns=non_numeric_columns, inplace=True)

# **6. Analiza korelacji**
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
numeric_df = df[numeric_columns]

correlation_matrix = numeric_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Macierz korelacji")
plt.show()

correlation_with_rent = correlation_matrix['Rent'].sort_values(ascending=False)
print("\nKorelacja zmiennych z 'Rent':")
print(correlation_with_rent)

low_correlation_columns = correlation_with_rent[correlation_with_rent.abs() < 0.1].index
df.drop(columns=low_correlation_columns, inplace=True)
print("\nUsunięte zmienne o niskiej korelacji:", low_correlation_columns.tolist())

# **7. Redukcja wymiarowości za pomocą PCA**
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('Rent', axis=1))

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"\nLiczba komponentów zachowujących 95% wariancji: {pca.n_components_}")
explained_variance = pca.explained_variance_ratio_
print("\nWyjaśniona wariancja przez komponenty PCA:")
print(explained_variance)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance.cumsum(), marker='o', linestyle='--')
plt.title('Skumulowana wariancja wyjaśniona przez komponenty PCA')
plt.xlabel('Liczba komponentów')
plt.ylabel('Skumulowana wariancja')
plt.grid()
plt.show()

# **8. Budowa modelu regresji z Keras**
# Przygotowanie danych
X = df.drop('Rent', axis=1)
y = df['Rent']

# Skalowanie danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podział na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Budowa modelu Keras
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# Kompilacja modelu
model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['mae'])

# Trenowanie modelu
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Ewaluacja modelu
y_pred = model.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print("\nEwaluacja modelu Keras:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# Wizualizacja wyników
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Strata treningowa')
plt.plot(history.history['val_loss'], label='Strata walidacyjna')
plt.title('Krzywa uczenia modelu Keras')
plt.xlabel('Epoka')
plt.ylabel('Strata (MSE)')
plt.legend()
plt.grid()
plt.show()

# **9. Przejście do klasyfikacji**
# Określenie progu klasyfikacji
median_rent = y.median()
df['Rent_Class'] = (y > median_rent).astype(int)  # 0: Tanie mieszkanie, 1: Drogie mieszkanie

# Przygotowanie danych do klasyfikacji
X = df.drop(['Rent', 'Rent_Class'], axis=1)
y_class = df['Rent_Class']

# Podział na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

y_train = np.array(y_train).astype('float32')
y_test = np.array(y_test).astype('float32')

# Budowa modelu klasyfikacji z Keras
model_classification = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Kompilacja modelu
model_classification.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Trenowanie modelu
history_classification = model_classification.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Ewaluacja modelu klasyfikacji
y_pred_probs = model_classification.predict(X_test).flatten()
y_pred_classes = (y_pred_probs > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes)
roc_auc = roc_auc_score(y_test, y_pred_probs)

print("\nEwaluacja modelu klasyfikacji Keras:")
print(f"Accuracy: {accuracy:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"ROC-AUC: {roc_auc:.2f}")

# Wizualizacja wyników klasyfikacji
plt.figure(figsize=(10, 5))
plt.plot(history_classification.history['loss'], label='Strata treningowa')
plt.plot(history_classification.history['val_loss'], label='Strata walidacyjna')
plt.title('Krzywa uczenia modelu klasyfikacji Keras')
plt.xlabel('Epoka')
plt.ylabel('Strata (Binary Crossentropy)')
plt.legend()
plt.grid()
plt.show()
