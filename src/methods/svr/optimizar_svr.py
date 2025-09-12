import pandas as pd
import numpy as np
import json
import logging
import optuna
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# --- 1. CONFIGURACIÓN PRINCIPAL ---
# Elección del kernel a usar: 'linear', 'rbf' o 'poly'
KERNEL_A_USAR = 'linear'
NUM_TRIALS = 50 # Número de combinaciones a probar
# ---------------------------------

def configurar_logging(log_path: Path):
    """Configura el sistema de logging para el script de optimización."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', filename=log_path, encoding='utf-8')

def objective(trial, kernel, X_train, X_test, y_train, y_test):
    """Define el objetivo de optimización con rangos de búsqueda más ajustados."""
    
    if kernel == 'linear':
        params = {
            'C': trial.suggest_float('C', 0.1, 100, log=True),
            'epsilon': trial.suggest_float('epsilon', 0.01, 1.0, log=True),
        }
    elif kernel == 'rbf':
        params = {
            'C': trial.suggest_float('C', 0.1, 100, log=True),
            'epsilon': trial.suggest_float('epsilon', 0.01, 1.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.001, 0.1, log=True),
        }
    elif kernel == 'poly':
        params = {
            'C': trial.suggest_float('C', 0.1, 100, log=True),
            'epsilon': trial.suggest_float('epsilon', 0.01, 1.0, log=True),
            'degree': trial.suggest_int('degree', 2, 3), # Rango reducido
            'coef0': trial.suggest_float('coef0', 0.0, 10.0),
        }
    
    modelo_svr = SVR(kernel=kernel, **params)
    
    modelo_svr.fit(X_train, y_train)
    predicciones = modelo_svr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predicciones))
    
    return rmse

# --- 2. Definición de Rutas Centralizadas ---
model_path = Path('src') / 'methods' / 'svr'
results_folder_name = f"best_results_{KERNEL_A_USAR}"
results_dir = model_path / results_folder_name
results_dir.mkdir(parents=True, exist_ok=True)
log_file = model_path / "logs" / f"optuna_optimization_svr_{KERNEL_A_USAR}.log"

configurar_logging(log_path=log_file)
logging.info(f"===== INICIO DE OPTIMIZACIÓN: SVR con kernel '{KERNEL_A_USAR}' =====")
print(f"===== Optimizando SVR con kernel: '{KERNEL_A_USAR}' =====")

# --- 3. Carga y Preparación de Datos ---
data_path = Path('data') / 'processed' / 'water_quality_processed.xlsx'
df = pd.read_excel(data_path, parse_dates=['Timestamp'])
y = df['Turbidity']
X = df.drop(columns=['Turbidity', 'Timestamp'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Escalado de Datos ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Datos cargados y escalados.")

# --- 5. Optimización con Optuna ---
logging.info(f"Iniciando estudio de Optuna con {NUM_TRIALS} trials.")
study = optuna.create_study(direction='minimize')

# Se pasa el kernel y los datos escalados a la función objetivo.
study.optimize(lambda trial: objective(trial, KERNEL_A_USAR, X_train_scaled, X_test_scaled, y_train, y_test), n_trials=NUM_TRIALS)

print("\n✅ Optimización completada.")

# --- 6. Resultados y Exportación ---
best_params = study.best_params
best_value = study.best_value

logging.info(f"Mejor valor (RMSE): {best_value}")
logging.info(f"Mejores hiperparámetros: {best_params}")

print(f"\n--- Mejores Resultados para SVR (kernel={KERNEL_A_USAR}) ---")
print(f"Mejor RMSE: {best_value:.4f}")
print("Mejores Hiperparámetros:")
print(json.dumps(best_params, indent=4))

# Se guardan los mejores parámetros en la carpeta de resultados correspondiente.
results_dir.mkdir(parents=True, exist_ok=True)
output_path = results_dir / 'best_hyperparameters.json'
with open(output_path, 'w') as f:
    json.dump(best_params, f, indent=4)

logging.info(f"Mejores hiperparámetros guardados en '{output_path}'")
print(f"\n✅ Hiperparámetros guardados en '{output_path}'")
logging.info("===== FIN DEL PROCESO DE OPTIMIZACIÓN =====")