# Celda 2: Información Mutua (MI)

# Se importan las librerías necesarias
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuración ---
# Se define la ruta del archivo.
ruta_archivo = r'data\water_quality.csv'
df = pd.read_csv(ruta_archivo)

# Se seleccionan solo las columnas numéricas para la correlación.
df_numeric = df.drop(columns=['Timestamp'])

# --- Agregar columna de ruido aleatorio ---
np.random.seed(42)
df_numeric['noise'] = np.random.normal(loc=0.0, scale=1.0, size=len(df_numeric))

if 'df_numeric' in locals():
    columnas = df_numeric.columns
    n_cols = len(columnas)
    mi_matrix = pd.DataFrame(np.zeros((n_cols, n_cols)), columns=columnas, index=columnas)

    print("Calculando Información Mutua (esto puede tardar unos segundos)...")

    for i in columnas:
        for j in columnas:
            X_mi = df_numeric[[i]]
            y_mi = df_numeric[j]
            mi_val = mutual_info_regression(X_mi, y_mi, random_state=42)
            mi_matrix.loc[i, j] = mi_val[0]

    print("Cálculo de MI completado.")

    # --- Ocultar la diagonal principal (MI consigo misma no importa) ---
    mi_matrix.values[np.diag_indices_from(mi_matrix)] = np.nan
    mask = np.isnan(mi_matrix)  # se enmascaran las celdas NaN (la diagonal)

    # --- Visualización ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        mi_matrix,
        annot=True,
        cmap='viridis',
        fmt='.3f',
        linewidths=.5,
        mask=mask       # ocultar la diagonal
    )
    plt.title('Heatmap de Información Mutua (variables continuas)')
    plt.show()

else:
    print("Error: El DataFrame 'df_numeric' no fue definido.")
    print("Asegúrese de ejecutar la Celda 1 exitosamente primero.")
