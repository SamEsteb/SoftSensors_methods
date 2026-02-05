import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor
from pathlib import Path

# ==========================================
# 1. VARIABLES DE CONFIGURACIÓN Y RUTAS
# ==========================================
TIPO_DATASET = 1  # 1: Water Quality, 2: SRU2
ADD_FEATURES_TEMPORALES = True  # Agregar Features Temporales adicionales
ADD_FEATURES_LAG = False  # Agregar Features Lag adicionales
VIEW_GRAPH = True  # Visualizar gráfico de resultados
SAVE_GRAPH = True  # Guardar gráfico de resultados

# Se definen rutas
data_dir = Path("data")
if TIPO_DATASET == 1:
    nombre_dataset = "water_quality"
    TARGET_COLUMN = "Turbidity"
elif TIPO_DATASET == 2:
    nombre_dataset = "SRU2" 
    TARGET_COLUMN = "AI508"

DATASET = data_dir / f"{nombre_dataset}.csv"

# Se crea directorio para resultados si no existe
model_dir = Path("src") / "methods" / "mlp" 
results_dir = model_dir / f"results_{nombre_dataset}_sklearn"
results_dir.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. CARGA Y PREPROCESAMIENTO
# ==========================================

# Se obtienen columnas del csv para validación
df_temp = pd.read_csv(DATASET, nrows=0)
columnas = df_temp.columns.tolist()

# Se verifica que la TARGET_COLUMN esté en las columnas del dataset
if TARGET_COLUMN not in columnas:
    raise ValueError(f"La columna '{TARGET_COLUMN}' no se encuentra en el dataset.")
print(f"La columna '{TARGET_COLUMN}' está presente en el dataset.")

# Se cargan los datos asegurando el índice temporal
df = pd.read_csv(DATASET, parse_dates=["Timestamp"], index_col="Timestamp")

# Se asegura que los datos estén ordenados cronológicamente
is_equal = df.index.is_monotonic_increasing
print(f"¿Los datos están ordenados cronológicamente? {is_equal}")
if not is_equal:
    df = df.sort_index()
    print("Los datos han sido ordenados cronológicamente.")

# Se agregan Features temporales (Hora, Minuto y Día de la semana)
if ADD_FEATURES_TEMPORALES:
    print("Agregando Features temporales...")
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['minute'] = df.index.minute

# Se agregan Lag Features en función de la TARGET_COLUMN
if ADD_FEATURES_LAG:
    print("Agregando Features Lag...")
    df[f'{TARGET_COLUMN}_lag1'] = df[TARGET_COLUMN].shift(1)

# Se eliminan filas con NaNs generados por los shifts
print(f"Tamaño antes de eliminar NaNs: {len(df)}")
df = df.dropna()
print(f"Tamaño después de eliminar NaNs: {len(df)}")

# Se divide el dataset en entrenamiento y prueba (70%-30%)
print("Dividiendo el dataset en conjuntos de entrenamiento y prueba...")
train_size = int(len(df) * 0.7)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

# Se verifica la integridad de la división
if df_train.empty or df_test.empty:
    raise ValueError("Conjunto de entrenamiento o prueba vacío después de la división.")

if df_train.index.max() < df_test.index.min():
    print("La división entre entrenamiento y prueba se ha realizado correctamente.")
else:
    raise ValueError("Error en la división entre entrenamiento y prueba.")

print(f"Tamaño del conjunto de entrenamiento: {len(df_train)}")
print(f"Tamaño del conjunto de prueba: {len(df_test)}")

# ==========================================
# 3. PREPARACIÓN (SCALING)
# ==========================================

# Se identifican las columnas de features (X) y target (y)
feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
print(f"Features utilizadas ({len(feature_cols)}): {feature_cols}")

X_train_np = df_train[feature_cols].values.astype(np.float32)
y_train_np = df_train[TARGET_COLUMN].values.astype(np.float32)

X_test_np = df_test[feature_cols].values.astype(np.float32)
y_test_np = df_test[TARGET_COLUMN].values.astype(np.float32)

# Se inicializan y ajustan los escaladores (fit solo en train para evitar data leakage)
scalerx = StandardScaler()
scalery = StandardScaler()

X_train_scaled = scalerx.fit_transform(X_train_np)
y_train_scaled = scalery.fit_transform(y_train_np.reshape(-1, 1)).flatten()

X_test_scaled = scalerx.transform(X_test_np)
y_test_scaled = scalery.transform(y_test_np.reshape(-1, 1)).flatten()

# ==========================================
# 4. DEFINICIÓN Y ENTRENAMIENTO DEL MODELO MLP (SKLEARN)
# ==========================================

print("Iniciando entrenamiento con Scikit-learn MLPRegressor...")

# Configuración igual a train_mlp.py:
# input_dim -> 64 -> ReLU -> 64 -> ReLU -> 1
# Solver: Adam, Learning Rate: 0.001, Epochs (max_iter): 100, Batch Size: 64
model = MLPRegressor(
    hidden_layer_sizes=(64, 64),
    activation='relu',
    solver='adam',
    alpha=0.0001, # Default L2 penalty
    batch_size=64,
    learning_rate_init=0.001,
    max_iter=100,
    random_state=None, 
    verbose=True,
    early_stopping=False # Para replicar exactamente las 100 épocas sin parar antes
)

model.fit(X_train_scaled, y_train_scaled)

# ==========================================
# 5. EVALUACIÓN FINAL Y RESULTADOS
# ==========================================

# Predicciones
train_preds_scaled = model.predict(X_train_scaled)
test_preds_scaled = model.predict(X_test_scaled)

# Inversa del escalado
train_preds = scalery.inverse_transform(train_preds_scaled.reshape(-1, 1)).flatten()
test_preds = scalery.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()

# Métricas en Test
r2 = r2_score(y_test_np, test_preds)
mse = mean_squared_error(y_test_np, test_preds)
mae = mean_absolute_error(y_test_np, test_preds)
mape = mean_absolute_percentage_error(y_test_np, test_preds)

print("\n--- Test Metrics ---")
print(f"R² Score: {r2:.8f}")
print(f"MSE: {mse:.8f}")
print(f"MAE: {mae:.8f}")
print(f"MAPE: {mape:.8f}")

# ==========================================
# 6. GRÁFICOS Y GUARDADO
# ==========================================

if VIEW_GRAPH or SAVE_GRAPH:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Gráfico de Pérdida
    if hasattr(model, 'loss_curve_'):
        axes[0].plot(model.loss_curve_, label='Train Loss (Sklearn)')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Curva de Aprendizaje')
        axes[0].legend()
        axes[0].grid(True)
    else:
        axes[0].text(0.5, 0.5, 'Loss curve not available', ha='center')

    # Gráfico de Predicciones
    axes[1].plot(y_test_np, label='Real', linewidth=2)
    axes[1].plot(test_preds, label='Predicho', linewidth=2, linestyle='--')
    axes[1].set_xlabel('Muestras')
    axes[1].set_ylabel(TARGET_COLUMN)
    axes[1].set_title(f'Predicción vs Realidad (Sklearn)')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if SAVE_GRAPH:
        file_path = results_dir / f"training_results_MLP_sklearn.png"
        plt.savefig(file_path)
        print(f"Gráfico guardado en: {file_path}")
    else:
        plt.show()
