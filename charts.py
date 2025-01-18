import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

predictions_df = pd.read_csv("quantile_predictions.csv")

# Porównanie predykcji kwantylowych z wartościami rzeczywistymi
plt.figure(figsize=(12, 6))
for quantile in [0.1, 0.5, 0.9]:
    plt.scatter(predictions_df['True Values'], predictions_df[f'Quantile_{quantile}'], alpha=0.5, label=f'Quantile {quantile}')
plt.plot([predictions_df['True Values'].min(), predictions_df['True Values'].max()],
         [predictions_df['True Values'].min(), predictions_df['True Values'].max()],
         color='red', linestyle='--', label='Ideal Prediction')
plt.title('Porównanie predykcji kwantylowych z wartościami rzeczywistymi')
plt.xlabel('Rzeczywiste wartości')
plt.ylabel('Przewidywane wartości')
plt.legend()
plt.grid()
plt.show()

# Wizualizacja przedziałów kwantylowych (10% - 90%) dla każdego przykładu
plt.figure(figsize=(12, 6))
plt.plot(predictions_df['True Values'], label='Rzeczywiste wartości', color='black', linewidth=2)
plt.fill_between(range(len(predictions_df)),
                 predictions_df['Quantile_0.1'],
                 predictions_df['Quantile_0.9'],
                 color='blue', alpha=0.2, label='Przedział (10%-90%)')
plt.plot(predictions_df['Quantile_0.5'], label='Mediana (50%)', color='orange')
plt.title('Przedziały kwantylowe predykcji')
plt.xlabel('Przykłady')
plt.ylabel('Wartości wynajmu')
plt.legend()
plt.grid()
plt.show()

# Histogram błędów predykcji dla mediany (50%)
predictions_df['Error_50%'] = predictions_df[f'Quantile_0.5'] - predictions_df['True Values']
plt.figure(figsize=(12, 6))
sns.histplot(predictions_df['Error_50%'], bins=30, kde=True, color='green')
plt.title('Histogram błędów predykcji dla mediany (50%)')
plt.xlabel('Błąd (Przewidywane - Rzeczywiste)')
plt.ylabel('Liczba przykładów')
plt.grid()
plt.show()

# Porównanie średnich błędów dla każdego kwantyla
mean_errors = {f'Mean Error Quantile {quantile}': abs(predictions_df[f'Quantile_{quantile}'] - predictions_df['True Values']).mean() for quantile in [0.1, 0.5, 0.9]}
plt.figure(figsize=(8, 5))
plt.bar(mean_errors.keys(), mean_errors.values(), color=['blue', 'orange', 'green'])
plt.title('Średni błąd predykcji dla każdego kwantyla')
plt.ylabel('Średni błąd')
plt.grid(axis='y')
plt.show()
