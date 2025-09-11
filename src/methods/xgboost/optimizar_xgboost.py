import pandas as pd
import numpy as np
import json
import logging
import optuna
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def configurar_logging(log_path: Path):
    """Configura el sistema de logging para el script de optimización."""
    # Se crea el directorio para los logs si no existe.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Se configura el logging básico.
    logging.basicConfig(
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_path,
        encoding='utf-8'
    )

def objective(trial, X_train, X_test, y_train, y_test):
    """
    Define el objetivo de optimización para un 'trial' de Optuna con XGBoost.
    """
    # 1. Se definen los rangos de búsqueda para los hiperparámetros de XGBoost.
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'lambda': trial.suggest_float('lambda', 1, 10), # L2 Regularization
        'alpha': trial.suggest_float('alpha', 0, 10),   # L1 Regularization
    }

    # 2. Se crea el modelo con los hiperparámetros del trial actual.
    modelo_xgb = xgb.XGBRegressor(**param, random_state=42, n_jobs=-1)

    # 3. Se entrena y evalúa el modelo.
    modelo_xgb.fit(X_train, y_train)
    predicciones = modelo_xgb.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predicciones))
    
    # 4. Se devuelve el RMSE. Optuna intentará minimizar este valor.
    return rmse

# --- Definición de Rutas (para XGBoost) ---
model_path = Path('src') / 'methods' / 'xgboost'
logs_path = model_path / "logs"
log_file = logs_path / "optuna_optimization_xgb.log"

# Se configura el logging.
configurar_logging(log_path=log_file)
logging.info("===== INICIO DE LA OPTIMIZACIÓN DE HIPERPARÁMETROS PARA XGBOOST CON OPTUNA =====")

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

# --- 2. Definición de Variables y División de Datos ---
logging.info("Definiendo y dividiendo los datos.")
y = df['Turbidity']
X = df.drop(columns=['Turbidity', 'Timestamp'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info(f"Datos divididos. Tamaño de X_train: {X_train.shape}, Tamaño de X_test: {X_test.shape}")

# --- 3. Optimización con Optuna ---
logging.info("Iniciando el estudio de optimización para XGBoost.")
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=100)

logging.info("Optimización completada.")
print("\n✅ Optimización completada.")

# --- 4. Resultados y Exportación ---
best_params = study.best_params
best_value = study.best_value

logging.info(f"Mejor valor (RMSE): {best_value}")
logging.info(f"Mejores hiperparámetros: {best_params}")

print("\n--- Mejores Resultados Encontrados para XGBoost ---")
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
logging.info("===== FIN DEL PROCESO DE OPTIMIZACIÓN DE XGBOOST =====")