from pathlib import Path
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# VARIABLES DE CONFIGURACIÓN
TIPO_DATASET = 1  # 1: Water Quality, 2: SRU2
ADD_FEATURES_TEMPORALES = True  # Agregar Features Temporales adicionales
ADD_FEATURES_LAG = False  # Agregar Features Lag adicionales
VIEW_GRAPH = True  # Visualizar gráfico de resultados
SAVE_GRAPH = True  # Guardar gráfico de resultados

# Se definen las rutas de los archivos según el dataset seleccionado
data_dir = Path("data")
if TIPO_DATASET == 1:
    nombre_dataset = "water_quality"
    DATASET = data_dir / f"{nombre_dataset}.csv"
    TARGET_COLUMN = "Turbidity"
elif TIPO_DATASET == 2:
    nombre_dataset = "SRU2"
    DATASET = data_dir / f"{nombre_dataset}.csv"
    TARGET_COLUMN = "AI508"

# Se crea el directorio para almacenar los resultados de Lasso
model_dir = Path("src") / "methods" / "regression_lasso"
results_dir = model_dir / f"results_{nombre_dataset}"
results_dir.mkdir(parents=True, exist_ok=True)

# Se obtienen las columnas del archivo CSV para validación preliminar
df_cols = pd.read_csv(DATASET, nrows=0)
columnas = df_cols.columns.tolist()
print(columnas)

# Se verifica la existencia de la columna objetivo en el dataset
if TARGET_COLUMN not in columnas:
    raise ValueError(f"La columna '{TARGET_COLUMN}' no se encuentra en el dataset.")
print(f"La columna '{TARGET_COLUMN}' está presente en el dataset.")

# Se cargan los datos y se establece la columna de tiempo como índice
df = pd.read_csv(DATASET, parse_dates=["Timestamp"], index_col="Timestamp")
is_equal = df.index.is_monotonic_increasing
print(f"¿Los datos están ordenados cronológicamente? {is_equal}")

# Se ordenan los datos cronológicamente si el índice no es monótono creciente
if not is_equal:
    df = df.sort_index()
    print("Los datos han sido ordenados cronológicamente.")

# Se agregan características temporales basadas en el índice de tiempo
if ADD_FEATURES_TEMPORALES:
    print("Agregando Features temporales...")
    df.loc[:, 'hour'] = df.index.hour
    df.loc[:, 'day_of_week'] = df.index.dayofweek
    df.loc[:, 'minute'] = df.index.minute

# Se generan características de rezago (Lag) si la configuración lo permite
if ADD_FEATURES_LAG:
    df.loc[:, f'{TARGET_COLUMN}_lag1'] = df[TARGET_COLUMN].shift(1)

# Se eliminan las filas que contienen valores nulos tras los desplazamientos
print(f"Tamaño antes de eliminar NaNs: {len(df)}")
df = df.dropna()
print(f"Tamaño después de eliminar NaNs: {len(df)}")

# Se divide el dataset en entrenamiento (70%) y prueba (30%) de forma secuencial
print("Dividiendo el dataset en conjuntos de entrenamiento y prueba...")
train_size = int(len(df) * 0.7)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

# Se valida que la división temporal sea correcta (sin solapamiento futuro en entrenamiento)
if df_train.empty or df_test.empty:
    raise ValueError("Conjunto de entrenamiento o prueba vacío después de la división.")

if df_train.index.max() < df_test.index.min():
    print("La división entre entrenamiento y prueba se ha realizado correctamente.")
else:
    raise ValueError("Error en la división entre entrenamiento y prueba.")

# Se seleccionan las características de entrada excluyendo los objetivos y marcas temporales
features = [col for col in df.columns if col not in [TARGET_COLUMN, 'Timestamp']]
print(f"Características seleccionadas para el modelo: {features}")

# Se preparan las matrices de datos para el entrenamiento y la evaluación
X_train = df_train[features]
y_train = df_train[TARGET_COLUMN]
X_test = df_test[features]
y_test = df_test[TARGET_COLUMN]

# Se inicializa y ajusta el escalador (StandardScaler) necesario para modelos lineales
print("Estandarizando las características...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Se instancia y entrena el modelo Lasso con parámetros fijos
print("Entrenando el modelo Lasso...")
modelo = Lasso(
    alpha=0.1,        
    random_state=None,
    max_iter=2000     
)
modelo.fit(X_train_scaled, y_train)
print("Modelo entrenado con éxito.")

# Se realizan las predicciones usando los datos escalados y se calculan las métricas
print("Evaluando el modelo...")
y_pred = modelo.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R2 Score: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")

# Se generan las visualizaciones de los resultados
if VIEW_GRAPH:
    # Se gráfica la comparación entre valores reales y predichos
    plt.figure(figsize=(15, 6))
    plt.plot(df_test.index, y_test, label='Valores Reales', color='blue')
    plt.plot(df_test.index, y_pred, label='Valores Predichos', color='red', alpha=0.7)
    plt.xlabel('Timestamp')
    plt.ylabel(TARGET_COLUMN)
    
    tipo_mdo = "Lasso Simple"
    titulo_prediccion = f'Predicción de {TARGET_COLUMN} (Actual) usando {tipo_mdo}'
    plt.title(titulo_prediccion)
    plt.legend(title=f'RMSE={rmse:.3f}\nR2={r2:.3f}\nMAE={mae:.3f}')
    
    if SAVE_GRAPH:
        plt.savefig(results_dir / f'lasso_simple_prediction_Ftemp({ADD_FEATURES_TEMPORALES})_Flag({ADD_FEATURES_LAG}).png')
    else:
        plt.show()

    # Se genera el gráfico de coeficientes
    if hasattr(modelo, "coef_"):
        coefs = modelo.coef_
        # Se ordenan los coeficientes por su magnitud absoluta
        indices = np.argsort(np.abs(coefs))[::-1]
        labels = [features[i] for i in indices]
        
        plt.figure(figsize=(12, 6))
        plt.title(f"Coeficientes del Modelo (Lasso alpha={modelo.alpha})")
        # Se grafican los valores reales de los coeficientes
        plt.bar(range(len(coefs)), coefs[indices], align="center")
        plt.xticks(range(len(coefs)), labels, rotation=90)
        plt.axhline(0, color='black', linewidth=0.8) # Línea de referencia en 0
        plt.tight_layout()
        
        if SAVE_GRAPH:
            plt.savefig(results_dir / f'lasso_simple_coefficients_Ftemp({ADD_FEATURES_TEMPORALES})_Flag({ADD_FEATURES_LAG}).png')
        else:
            plt.show()