import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import xgboost as xgb  # Se importa la librería de XGBoost
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

def configurar_logging(log_path: Path):
    """Configura el sistema de logging para el script del modelo."""
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

# --- VARIABLES GLOBALES DE CONFIGURACIÓN ---
USAR_BEST_PARAMS = True
SHOW_PLOTS = False

# --- Definición de Rutas ---
model_path = Path('src') / 'methods' / 'xgboost'
logs_path = model_path / "logs"
log_name = "train_xgboost.log" if not USAR_BEST_PARAMS else "train_xgboost_best_params.log"
log_file = logs_path / log_name

# Se configura el logging antes de cualquier otra operación.
configurar_logging(log_path=log_file)
logging.info("===== INICIO DEL ENTRENAMIENTO DEL MODELO XGBOOST =====")

if USAR_BEST_PARAMS:
    # Se obtienen los hiperparámetros optimizados desde el archivo JSON.
    best_params_path = model_path / 'best_results' / 'best_hyperparameters.json'
    try:
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
        logging.info(f"Hiperparámetros optimizados cargados desde '{best_params_path}': {best_params}")
        print(f"✅ Hiperparámetros optimizados cargados desde '{best_params_path}'")
    except FileNotFoundError:
        error_msg = f"Error: No se encontró el archivo de hiperparámetros optimizados en '{best_params_path}'"
        logging.error(error_msg)
        print(f"❌ {error_msg}")
        exit()

# --- 1. Carga de Datos Procesados ---
data_path = Path('data') / 'processed' / 'water_quality_processed.xlsx'
logging.info(f"Cargando datos desde: '{data_path}'")
try:
    df = pd.read_excel(data_path, parse_dates=['Timestamp'])
    logging.info(f"Datos cargados exitosamente. {len(df)} filas y {len(df.columns)} columnas.")
except FileNotFoundError:
    error_msg = f"Error: No se encontró el archivo procesado en '{data_path}'"
    logging.error(error_msg)
    print(f"❌ {error_msg}")
    exit()

# --- 2. Definición de Variables Predictoras (X) y Objetivo (y) ---
logging.info("Definiendo variables predictoras (X) y objetivo (y).")
# Se define la variable objetivo.
y = df['Turbidity']
# Se definen las variables predictoras de forma dinámica.
X = df.drop(columns=['Turbidity', 'Timestamp'])
logging.info(f"Variable objetivo 'y' (Turbidity) definida con {len(y)} muestras.")
logging.info(f"Variables predictoras 'X' definidas con forma: {X.shape}")

# --- 3. División de Datos en Entrenamiento y Prueba ---
logging.info("Dividiendo los datos en conjuntos de entrenamiento (80%) y prueba (20%).")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info(f"Tamaño de X_train: {X_train.shape}")
logging.info(f"Tamaño de X_test: {X_test.shape}")

# --- 4. Creación y Entrenamiento del Modelo ---
logging.info("Creando y entrenando el modelo XGBoost Regressor.")
# Se usan los hiperparámetros optimizados si la variable global es True.
if USAR_BEST_PARAMS:
    modelo_xgb = xgb.XGBRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1
    )
    logging.info(f"Modelo creado con hiperparámetros optimizados: {best_params}")
    print(f"✅ Modelo creado con hiperparámetros optimizados: {best_params}")
else:
    # Se usa un modelo con parámetros por defecto si no se usan los optimizados.
    modelo_xgb = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    logging.info(f"Modelo creado con hiperparámetros por defecto: {modelo_xgb.get_params()}")
    print(f"✅ Modelo creado con hiperparámetros por defecto: {modelo_xgb.get_params()}")
    
# Se entrena el modelo.
modelo_xgb.fit(X_train, y_train)
logging.info("Modelo entrenado con éxito.")
print("✅ Modelo XGBoost entrenado con éxito.")

# --- 5. Predicción y Evaluación ---
logging.info("Realizando predicciones sobre el conjunto de prueba.")
predicciones = modelo_xgb.predict(X_test)
logging.info("Calculando métricas de rendimiento (MAE, RMSE, R²).")
mae = mean_absolute_error(y_test, predicciones)
rmse = np.sqrt(mean_squared_error(y_test, predicciones))
r2 = r2_score(y_test, predicciones)

logging.info(f"Resultados de Evaluación -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
print("\n--- Resultados de la Evaluación ---")
print(f"Error Absoluto Medio (MAE): {mae:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
print(f"Coeficiente de Determinación (R²): {r2:.4f}")

# --- 6. Exportación de Métricas ---
if USAR_BEST_PARAMS:
    results_dir = model_path / 'best_results'
else:
    results_dir = model_path / 'results'
# Se crea el directorio de resultados si no existe.
results_dir.mkdir(parents=True, exist_ok=True)
metrics = {'mae': mae, 'rmse': rmse, 'r2': r2}

logging.info(f"Exportando métricas a '{results_dir / 'model_metrics.json'}'")
with open(results_dir / 'model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"\n✅ Métricas exportadas a '{results_dir / 'model_metrics.json'}'")

# --- 7. Visualización de Resultados ---
logging.info("Generando visualizaciones de resultados.")

# Gráfico 1: Valor Real vs. Predicho (Scatter Plot)
logging.info("Generando gráfico de dispersión: Valor Real vs. Valor Predicho.")
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid", palette="pastel")
sns.scatterplot(x=y_test, y=predicciones, alpha=0.7, s=10, edgecolor='k', color="#4F8DFD")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color="#FF6F61", linewidth=2, label="Predicción Perfecta")
plt.title('Valor Real vs. Valor Predicho (XGBoost)', fontsize=18, fontweight='bold')
plt.xlabel('Valor Real de Turbidez', fontsize=14)
plt.ylabel('Valor Predicho de Turbidez', fontsize=14)
metrics_text = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}"
plt.legend(fontsize=12, loc='upper left')
plt.gca().text(0.05, 0.90, metrics_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
plt.grid(True, linestyle='--', alpha=0.5)
sns.despine()
plt.tight_layout()
plt.savefig(results_dir / 'real_vs_predicho(scatter).png', dpi=150)
logging.info(f"Gráfico de dispersión guardado en '{results_dir / 'real_vs_predicho(scatter).png'}'")
if SHOW_PLOTS:
    plt.show()

# Gráfico 2: Serie Temporal de Predicciones vs. Reales
logging.info("Generando gráfico de serie de tiempo: Real vs. Predicho.")
results_df = pd.DataFrame({'Timestamp': df.loc[y_test.index, 'Timestamp'], 'Real': y_test, 'Predicho': predicciones})
results_df.sort_values(by='Timestamp', inplace=True)

plt.figure(figsize=(15, 7))
sns.set_theme(style="whitegrid")
plt.plot(results_df['Timestamp'], results_df['Real'], label='Valor Real', color='#1f77b4', linewidth=2)
plt.plot(results_df['Timestamp'], results_df['Predicho'], label='Valor Predicho', color='#ff7f0e', linestyle='--', alpha=0.9)
plt.title('Comparación de Turbidez Real vs. Predicha en el Tiempo (XGBoost)', fontsize=18, fontweight='bold')
plt.xlabel('Fecha y Hora', fontsize=14)
plt.ylabel('Turbidez', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(results_dir / 'real_vs_predicho(time).png', dpi=150)
logging.info(f"Gráfico de serie de tiempo guardado en '{results_dir / 'real_vs_predicho(time).png'}'")
if SHOW_PLOTS:
    plt.show()

# Gráfico 3: Importancia de Características
logging.info("Generando gráfico de importancia de características.")
importancias = modelo_xgb.feature_importances_
columnas = X.columns
importancia_df = pd.DataFrame({'Característica': columnas, 'Importancia': importancias})
importancia_df = importancia_df.sort_values(by='Importancia', ascending=True)

plt.figure(figsize=(12, 8))
sns.set_theme(style="whitegrid")
sns.barplot(x='Importancia', y='Característica', data=importancia_df, hue='Característica', palette="viridis", legend=False)
plt.title('Importancia de cada Característica en el Modelo XGBoost', fontsize=18, fontweight='bold')
plt.xlabel('Importancia Relativa', fontsize=14)
plt.ylabel('Característica', fontsize=14)
for i, v in enumerate(importancia_df['Importancia']):
    plt.text(v + 0.001, i, f"{v:.3f}", color='black', va='center', fontsize=11)
plt.tight_layout()
sns.despine()
plt.savefig(results_dir / 'importancia_caracteristicas.png', dpi=150)
logging.info(f"Gráfico de importancia de características guardado en '{results_dir / 'importancia_caracteristicas.png'}'")
if SHOW_PLOTS:
    plt.show()

logging.info("===== FIN DEL PROCESO DE ENTRENAMIENTO Y EVALUACIÓN =====")