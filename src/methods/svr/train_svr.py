import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# --- 1. CONFIGURACIÓN PRINCIPAL ---
# Elección del kernel a usar: 'linear', 'rbf' o 'poly'
KERNEL_A_USAR = 'linear' 
USAR_BEST_PARAMS = False
SHOW_PLOTS = True
# ---------------------------------

def configurar_logging(log_path: Path):
    """Configura el sistema de logging para el script del modelo."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', filename=log_path, encoding='utf-8')

# --- 2. Definición de Rutas Centralizadas ---
model_path = Path('src') / 'methods' / 'svr'
if USAR_BEST_PARAMS:
    results_dir = model_path / f'best_results_{KERNEL_A_USAR}'
else:
    results_dir = model_path / f'results_{KERNEL_A_USAR}'
# crear directorio de resultados si no existe
results_dir.mkdir(parents=True, exist_ok=True)
log_name = f"train_svr_{KERNEL_A_USAR}.log" if not USAR_BEST_PARAMS else f"train_svr_{KERNEL_A_USAR}_best_params.log"
log_file = model_path / "logs" / log_name

configurar_logging(log_path=log_file)
logging.info(f"===== INICIO DEL ENTRENAMIENTO: SVR con kernel '{KERNEL_A_USAR}' =====")
print(f"===== Entrenando SVR con kernel: '{KERNEL_A_USAR}' =====")

# --- 3. Carga de Hiperparámetros ---
if USAR_BEST_PARAMS:
    best_params_path = results_dir / 'best_hyperparameters.json'
    try:
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
        logging.info(f"Hiperparámetros cargados: {best_params}")
        print(f"✅ Hiperparámetros cargados desde '{best_params_path}'")
    except FileNotFoundError:
        logging.error(f"Error: No se encontró el archivo de hiperparámetros en '{best_params_path}'")
        print(f"❌ Error: No se encontró el archivo de hiperparámetros. El script se detendrá.")
        exit()

# --- 4. Carga y Preparación de Datos ---
data_path = Path('data') / 'processed' / 'water_quality_processed.xlsx'
df = pd.read_excel(data_path, parse_dates=['Timestamp'])
y = df['Turbidity']
X = df.drop(columns=['Turbidity', 'Timestamp'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. Escalado de Datos ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 6. Creación y Entrenamiento del Modelo SVR ---
if USAR_BEST_PARAMS:
    modelo_svr = SVR(kernel=KERNEL_A_USAR, **best_params)
    print(f"✅ Modelo creado con kernel '{KERNEL_A_USAR}' y parámetros optimizados.")
else:
    if KERNEL_A_USAR == 'linear':
        modelo_svr = SVR(kernel='linear', C=1.0)
    elif KERNEL_A_USAR == 'rbf':
        modelo_svr = SVR(kernel='rbf', C=1.0, gamma='scale')
    elif KERNEL_A_USAR == 'poly':
        modelo_svr = SVR(kernel='poly', C=1.0, degree=3)
    print(f"✅ Modelo creado con kernel '{KERNEL_A_USAR}' y parámetros por defecto.")

modelo_svr.fit(X_train_scaled, y_train)
logging.info("Modelo entrenado con éxito.")
print("✅ Modelo entrenado con éxito.")

# --- 7. Predicción y Evaluación ---
predicciones = modelo_svr.predict(X_test_scaled)
mae = mean_absolute_error(y_test, predicciones)
rmse = np.sqrt(mean_squared_error(y_test, predicciones))
r2 = r2_score(y_test, predicciones)

print("\n--- Resultados de la Evaluación ---")
print(f"Error Absoluto Medio (MAE): {mae:.4f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.4f}")
print(f"Coeficiente de Determinación (R²): {r2:.4f}")

# --- 8. Exportación de Métricas ---
results_dir.mkdir(parents=True, exist_ok=True)
metrics = {'mae': mae, 'rmse': rmse, 'r2': r2}
with open(results_dir / 'model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"\n✅ Métricas exportadas a '{results_dir / 'model_metrics.json'}'")

# --- 9. Visualización de Resultados ---
logging.info("Generando visualizaciones de resultados.")

# Gráfico 1: Valor Real vs. Predicho (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid", palette="pastel")
sns.scatterplot(x=y_test, y=predicciones, alpha=0.7, s=10, edgecolor='k', color="#4F8DFD")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color="#FF6F61", linewidth=2, label="Predicción Perfecta")
plt.title(f'Valor Real vs. Predicho (SVR {KERNEL_A_USAR.capitalize()})', fontsize=18, fontweight='bold')
plt.xlabel('Valor Real de Turbidez', fontsize=14)
plt.ylabel('Valor Predicho de Turbidez', fontsize=14)
metrics_text = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}"
plt.legend(fontsize=12, loc='upper left')
plt.gca().text(0.05, 0.90, metrics_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(results_dir / 'real_vs_predicho(scatter).png', dpi=150)
if SHOW_PLOTS:
    plt.show()

# Gráfico 2: Serie Temporal de Predicciones vs. Reales
results_df = pd.DataFrame({'Timestamp': df.loc[y_test.index, 'Timestamp'], 'Real': y_test, 'Predicho': predicciones})
results_df.sort_values(by='Timestamp', inplace=True)

plt.figure(figsize=(15, 7))
sns.set_theme(style="whitegrid")
plt.plot(results_df['Timestamp'], results_df['Real'], label='Valor Real', color='#1f77b4', linewidth=2)
plt.plot(results_df['Timestamp'], results_df['Predicho'], label='Valor Predicho', color='#ff7f0e', linestyle='--', alpha=0.9)
plt.title(f'Comparación en el Tiempo (SVR {KERNEL_A_USAR.capitalize()})', fontsize=18, fontweight='bold')
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
if SHOW_PLOTS:
    plt.show()

# Gráfico 3: Importancia de Características (solo para kernel lineal)
if KERNEL_A_USAR == 'linear':
    importancias = np.abs(modelo_svr.coef_[0])
    columnas = X.columns
    importancia_df = pd.DataFrame({'Característica': columnas, 'Importancia': importancias})
    importancia_df = importancia_df.sort_values(by='Importancia', ascending=True)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importancia', y='Característica', data=importancia_df, hue='Característica', palette="plasma", legend=False)
    plt.title('Importancia de Características (SVR Lineal)', fontsize=18, fontweight='bold')
    plt.xlabel('Magnitud del Coeficiente', fontsize=14)
    plt.ylabel('Característica', fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / 'importancia_caracteristicas.png', dpi=150)
    if SHOW_PLOTS:
        plt.show()
else:
    logging.info(f"El gráfico de importancia de características no aplica para el kernel '{KERNEL_A_USAR}'.")
    print(f"\nℹ️ Nota: El gráfico de importancia de características no se genera para el kernel '{KERNEL_A_USAR}'.")

logging.info(f"===== FIN DEL ENTRENAMIENTO (kernel: {KERNEL_A_USAR}) =====")