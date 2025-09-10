import pandas as pd
import numpy as np
import json
import logging
import optuna
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def configurar_logging(log_path: Path):
    """Configura el sistema de logging para el script de optimización."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_path,
        encoding='utf-8'
    )

def objective(trial, X_train, X_test, y_train, y_test):
    """
    Define el objetivo de optimización para un 'trial' de Optuna.
    Entrena un modelo con los hiperparámetros sugeridos y devuelve su error.
    """
    # 1. Se definen los hiperparámetros a optimizar para este trial.
    n_estimators = trial.suggest_int('n_estimators', 50, 400)
    max_depth = trial.suggest_int('max_depth', 5, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_float('max_features', 0.1, 1.0)

    # 2. Se crea el modelo con los hiperparámetros del trial actual.
    modelo_rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    # 3. Se entrena y evalúa el modelo.
    modelo_rf.fit(X_train, y_train)
    predicciones = modelo_rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predicciones))
    
    # 4. Se devuelve el RMSE. Optuna intentará minimizar este valor.
    return rmse

# --- Definición de Rutas ---
model_path = Path('src') / 'methods' / 'random_forest'
logs_path = model_path / "logs"
log_file = logs_path / "optuna_optimization.log"

# Se configura el logging.
configurar_logging(log_path=log_file)
logging.info("===== INICIO DE LA OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA =====")

# --- 1. Carga de Datos Procesados ---
data_path = Path('data') / 'processed' / 'water_quality_processed.xlsx'
logging.info(f"Cargando datos desde: '{data_path}'")
try:
    df = pd.read_excel(data_path, parse_dates=['Timestamp'])
    logging.info(f"Datos cargados exitosamente. {len(df)} filas y {len(df.columns)} columnas.")
except FileNotFoundError:
    error_msg = f"Error: No se encontró el archivo procesado en '{data_path}'"
    logging.error(error_msg)
    exit()

# --- 2. Ingeniería de Características ---
logging.info("Realizando ingeniería de características.")
df['hora'] = df['Timestamp'].dt.hour
df['dia_semana'] = df['Timestamp'].dt.dayofweek
df['minuto'] = df['Timestamp'].dt.minute

# --- 3. Definición de Variables y División de Datos ---
logging.info("Definiendo y dividiendo los datos.")
y = df['Turbidity']
X = df.drop(columns=['Turbidity', 'Timestamp'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info(f"Datos divididos. Tamaño de X_train: {X_train.shape}, Tamaño de X_test: {X_test.shape}")

# --- 4. Optimización con Optuna ---
logging.info("Iniciando el estudio de optimización.")
# Se crea un 'estudio'. El objetivo es minimizar el resultado de la función 'objective'.
study = optuna.create_study(direction='minimize')

# Se ejecuta la optimización. n_trials es el número de combinaciones a probar.
# Se usa una función lambda para pasar los datos a la función objective.
study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=100)

logging.info("Optimización completada.")
print("\n✅ Optimización completada.")

# --- 5. Resultados y Exportación ---
best_params = study.best_params
best_value = study.best_value

logging.info(f"Mejor valor (RMSE): {best_value}")
logging.info(f"Mejores hiperparámetros: {best_params}")

print("\n--- Mejores Resultados Encontrados ---")
print(f"Mejor RMSE: {best_value:.4f}")
print("Mejores Hiperparámetros:")
print(json.dumps(best_params, indent=4))

# Se crea el directorio de resultados y se guarda el JSON.
results_dir = model_path / 'best_results'
results_dir.mkdir(parents=True, exist_ok=True)
output_path = results_dir / 'best_hyperparameters.json'

with open(output_path, 'w') as f:
    json.dump(best_params, f, indent=4)

logging.info(f"Mejores hiperparámetros guardados en '{output_path}'")
print(f"\n✅ Hiperparámetros guardados en '{output_path}'")
logging.info("===== FIN DEL PROCESO DE OPTIMIZACIÓN =====")