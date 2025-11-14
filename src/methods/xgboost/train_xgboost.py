from pathlib import Path
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# VARIABLES DE CONFIGURACIÓN
TIPO_DATASET = 1  # 1: Water Quality, 2: SRU2
ADD_FEATURES_TEMPORALES = True  # Agregar Features Temporales adicionales
ADD_FEATURES_LAG = False  # Agregar Features Lag adicionales
VIEW_GRAPH = True  # Visualizar gráfico de resultados
SAVE_GRAPH = True  # Guardar gráfico de resultados
VENTANA_DE_PREDICCION = 0  # Ventana de predicción (0 = sin ventana)

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
model_dir = Path("src") / "methods" / "xgboost"
results_dir = model_dir / f"results_{nombre_dataset}"
results_dir.mkdir(parents=True, exist_ok=True)


# Obtener columnas del csv
df = pd.read_csv(DATASET, nrows=0)
columnas = df.columns.tolist()
print(columnas)

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

# Agregar Features temporales (Hora, Minuto y Día de la semana)
if ADD_FEATURES_TEMPORALES:
    print("Agregando Features temporales...")
    df.loc[:, 'hour'] = df.index.hour
    df.loc[:, 'day_of_week'] = df.index.dayofweek
df.loc[:, 'minute'] = df.index.minute

# Agregar Lag Features en función de la TARGET_COLUMN (Lag de 1)
if ADD_FEATURES_LAG:
    df.loc[:, f'{TARGET_COLUMN}_lag1'] = df[TARGET_COLUMN].shift(1)

# Crear Target Futuro para la predicción (Ventana de predicción)
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

# Dividir el dataset en entrenamiento y prueba ( 70%-30% )
print("Dividiendo el dataset en conjuntos de entrenamiento y prueba...")
print(f"Tamaño total del dataset: {len(df)}")
train_size = int(len(df) * 0.7)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

# Verificar que la división se haya realizado correctamente usando el índice (Timestamp)
if df_train.empty or df_test.empty:
    raise ValueError("Conjunto de entrenamiento o prueba vacío después de la división.")

if df_train.index.max() < df_test.index.min():
    print("La división entre entrenamiento y prueba se ha realizado correctamente.")
else:
    raise ValueError("Error en la división entre entrenamiento y prueba.")

print(f"Tamaño del conjunto de entrenamiento: {len(df_train)}")
print(f"Tamaño del conjunto de prueba: {len(df_test)}")

# Crear la lista de Features eliminando la columna TARGET_COLUMN y Timestamp
features = [col for col in df.columns if col not in [Y_COLUMN, TARGET_COLUMN, 'Timestamp']]
print(f"Características seleccionadas para el modelo: {features}")

# Separar los datos
X_train = df_train[features]
y_train = df_train[Y_COLUMN]
X_test = df_test[features]
y_test = df_test[Y_COLUMN]

# Entrenar el modelo XGBoost
print("Entrenando el modelo XGBoost Regressor...")
modelo = XGBRegressor(n_estimators=200, objective='reg:squarederror', random_state=None, n_jobs=-1, verbosity=1)
modelo.fit(X_train, y_train)
print("Modelo XGBoost entrenado con éxito.")

# Evaluar el modelo
print("Evaluando el modelo...")
y_pred = modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R2 Score: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")

# Visualizar Gráficos de Resultados
import matplotlib.pyplot as plt
if VIEW_GRAPH:
    # Timestamp vs Values Reales y Predichas
    plt.figure(figsize=(15, 6))
    plt.plot(df_test.index, y_test, label='Valores Reales', color='blue')
    plt.plot(df_test.index, y_pred, label='Valores Predichos', color='red', alpha=0.7)
    plt.xlabel('Timestamp')
    plt.ylabel(TARGET_COLUMN)
    if VENTANA_DE_PREDICCION > 0:
        titulo_prediccion = f'Predicción de {TARGET_COLUMN} ({VENTANA_DE_PREDICCION} min. a futuro) usando XGBoost'
    else:
        titulo_prediccion = f'Predicción de {TARGET_COLUMN} (Actual) usando XGBoost'

    plt.title(titulo_prediccion)
    plt.legend(title=f'Prediciendo t+{VENTANA_DE_PREDICCION} min\nRMSE={rmse:.3f}\nR2={r2:.3f}\nMAE={mae:.3f}')
    if SAVE_GRAPH:
        plt.savefig(results_dir / f'xgb_prediction_ventana({VENTANA_DE_PREDICCION})_Ftemp({ADD_FEATURES_TEMPORALES})_Flag({ADD_FEATURES_LAG}).png')
    else:
        plt.show()

    # Importancia de las Features
    if hasattr(modelo, "feature_importances_"):
        importances = modelo.feature_importances_
        indices = np.argsort(importances)[::-1]
        n_features = len(importances)
        positions = range(n_features)
        labels = [X_train.columns[i] for i in indices]
        plt.figure(figsize=(12, 6))
        plt.title("Importancia de las Features - XGBoost")
        plt.bar(positions, importances[indices], align="center")
        plt.xticks(positions, labels, rotation=90)
        plt.tight_layout()
        if SAVE_GRAPH:
            plt.savefig(results_dir / f'xgb_feature_importance_ventana({VENTANA_DE_PREDICCION})_Ftemp({ADD_FEATURES_TEMPORALES})_Flag({ADD_FEATURES_LAG}).png')
        else:
            plt.show()
    else:
        print("El modelo no proporciona 'feature_importances_' y no se puede mostrar la importancia de las features.")