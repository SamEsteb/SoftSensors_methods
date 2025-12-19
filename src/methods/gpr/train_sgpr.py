"""
Creé el ambiente usando:
conda create --name GPy_Env python=3.9 -y
conda install -c conda-forge gpy pandas matplotlib scikit-learn openpyxl -y
"""

import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Libería para implementar el modelo GPR
import GPy

warnings.filterwarnings('ignore')

# VARIABLES DE CONFIGURACIÓN
TIPO_DATASET = 1  # 1: Water Quality, 2: SRU2
ADD_FEATURES_TEMPORALES = True  # Agregar Features Temporales adicionales
ADD_FEATURES_LAG = False  # Agregar Features Lag adicionales
VIEW_GRAPH = True  # Visualizar gráfico de resultados
SAVE_GRAPH = True  # Guardar gráfico de resultados
VENTANA_DE_PREDICCION = 0  # Ventana de predicción (0 = sin ventana)

# Cantidad de puntos inducidos (Z)
NUM_INDUCING = 500 

# Definir rutas
data_dir = Path("data")
if TIPO_DATASET == 1:
    nombre_dataset = "water_quality"
    DATASET = data_dir / f"{nombre_dataset}.csv"
    TARGET_COLUMN = "Turbidity"
elif TIPO_DATASET == 2:
    nombre_dataset = "SRU2"
    DATASET = data_dir / f"{nombre_dataset}.csv"
    TARGET_COLUMN = "AI508"

# Crear directorio para resultados si no existe
model_dir = Path("src") / "methods" / "gpr"
results_dir = model_dir / f"results_{nombre_dataset}_sparse"
results_dir.mkdir(parents=True, exist_ok=True)

# Obtener columnas del csv
df = pd.read_csv(DATASET, nrows=0)
columnas = df.columns.tolist()
print(f"Columnas detectadas: {columnas}")

# Verificar que la TARGET_COLUMN esté en las columnas del dataset
if TARGET_COLUMN not in columnas:
    raise ValueError(f"La columna '{TARGET_COLUMN}' no se encuentra en el dataset.")
print(f"La columna '{TARGET_COLUMN}' está presente en el dataset.")

# Asegurarse que los datos estén ordenados cronológicamente
df = pd.read_csv(DATASET, parse_dates=["Timestamp"], index_col="Timestamp")
is_equal = df.index.is_monotonic_increasing
print(f"¿Los datos están ordenados cronológicamente? {is_equal}")
if not is_equal:
    df = df.sort_index()
    print("Los datos han sido ordenados cronológicamente.")

# Agregar Features temporales
if ADD_FEATURES_TEMPORALES:
    print("Agregando Features temporales...")
    df.loc[:, 'hour'] = df.index.hour
    df.loc[:, 'day_of_week'] = df.index.dayofweek
df.loc[:, 'minute'] = df.index.minute

# Agregar Lag Features
if ADD_FEATURES_LAG:
    df.loc[:, f'{TARGET_COLUMN}_lag1'] = df[TARGET_COLUMN].shift(1)

# Crear Target Futuro
if VENTANA_DE_PREDICCION > 0:
    print(f"Generando target para predecir {VENTANA_DE_PREDICCION} minutos a futuro...")
    n_steps_futuro = VENTANA_DE_PREDICCION
    FUTURE_TARGET_COLUMN = f'{TARGET_COLUMN}_future'
    df.loc[:, FUTURE_TARGET_COLUMN] = df[TARGET_COLUMN].shift(-n_steps_futuro)
    Y_COLUMN = FUTURE_TARGET_COLUMN    
else:
    print("No se usará ventana de predicción. Se predice el valor actual.")
    Y_COLUMN = TARGET_COLUMN 

# Eliminar filas con NaNs
print(f"Tamaño antes de eliminar NaNs: {len(df)}")
df = df.dropna()
print(f"Tamaño después de eliminar NaNs: {len(df)}")

# Dividir el dataset en entrenamiento y prueba (70%-30%)
print("Dividiendo el dataset en conjuntos de entrenamiento y prueba...")
train_size = int(len(df) * 0.7)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

# Verificar división
if df_train.index.max() < df_test.index.min():
    print("La división entre entrenamiento y prueba se ha realizado correctamente.")
else:
    raise ValueError("Error en la división entre entrenamiento y prueba.")

print(f"Tamaño del conjunto de entrenamiento: {len(df_train)}")
print(f"Tamaño del conjunto de prueba: {len(df_test)}")

# Crear la lista de Features
features = [col for col in df.columns if col not in [Y_COLUMN, TARGET_COLUMN, 'Timestamp']]
print(f"Características seleccionadas para el modelo: {features}")

# Separar los datos
X_train_raw = df_train[features].values
y_train_raw = df_train[Y_COLUMN].values.reshape(-1, 1)
X_test_raw = df_test[features].values
y_test_raw = df_test[Y_COLUMN].values.reshape(-1, 1)

# Escalado de datos
print("Escalando datos...")
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train_raw) # GPy acepta float64 (numpy default), no necesita float32
X_test = scaler_X.transform(X_test_raw)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train_raw)

# Sparse GPR con GPy
print(f"Configurando GPy SGPR con {NUM_INDUCING} puntos inducidos...")

# Kernel (RBF)
input_dim = X_train.shape[1]
kernel = GPy.kern.RBF(input_dim=input_dim, variance=1., lengthscale=1.)

# Modelo Sparse GP Regression
model = GPy.models.SparseGPRegression(X_train, y_train, kernel=kernel, num_inducing=NUM_INDUCING)

# Ruido gaussiano
model.Gaussian_noise.variance = 0.01

print("Optimizando el modelo con GPy (L-BFGS-B por defecto)...")
model.optimize(messages=True)

print("Entrenamiento completado.")
print(model)

# Evaluación del modelo
print("Realizando predicciones...")
y_pred_scaled, y_var_scaled = model.predict(X_test)

# Invertir escalado de la predicción
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Desviación estándar en escala original
sigma = np.sqrt(y_var_scaled) * scaler_y.scale_[0]

# Métricas (y_test_raw vs y_pred)
y_test_metrics = y_test_raw

mse = mean_squared_error(y_test_metrics, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_metrics, y_pred)
mae = mean_absolute_error(y_test_metrics, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R2 Score: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")

# Visualizar Gráficos
if VIEW_GRAPH:
    plt.figure(figsize=(15, 6))
    
    # Graficar Reales
    plt.plot(df_test.index, y_test_metrics.ravel(), label='Valores Reales', color='blue', alpha=0.4)
    
    # Graficar Predicción
    plt.plot(df_test.index, y_pred.ravel(), label='Predicción GPy SGPR', color='darkred', alpha=0.8, linewidth=1.5)
    
    # Graficar Intervalo de Confianza
    plt.fill_between(df_test.index, 
                     (y_pred - 1.96 * sigma).ravel(), 
                     (y_pred + 1.96 * sigma).ravel(), 
                     alpha=0.2, color='red', label='Intervalo Confianza 95%')
    
    plt.xlabel('Timestamp')
    plt.ylabel(TARGET_COLUMN)
    
    if VENTANA_DE_PREDICCION > 0:
        titulo_prediccion = f'Predicción de {TARGET_COLUMN} ({VENTANA_DE_PREDICCION} min. a futuro) usando GPy SGPR'
    else:
        titulo_prediccion = f'Predicción de {TARGET_COLUMN} (Actual) usando GPy SGPR'

    plt.title(titulo_prediccion)
    plt.legend(loc='upper left', title=f'RMSE={rmse:.3f}\nR2={r2:.3f}\nMAE={mae:.3f}')
    
    if SAVE_GRAPH:
        plt.savefig(results_dir / f'gpy_prediction_ventana({VENTANA_DE_PREDICCION}).png')
        print(f"Gráfico guardado en {results_dir}")
    else:
        plt.show()