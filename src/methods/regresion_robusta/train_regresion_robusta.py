from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor, QuantileRegressor
from sklearn.preprocessing import PolynomialFeatures

# VARIABLES DE CONFIGURACIÓN
TIPO_DATASET = 2  # 1: Water Quality, 2: SRU2
ADD_FEATURES_TEMPORALES = True  # Agregar Features Temporales adicionales
ADD_FEATURES_LAG = False  # Agregar Features Lag adicionales
ADD_FEATURES_POLYNOMIAL = True  # Agregar Features Polinómicas adicionales
VIEW_GRAPH = False  # Visualizar gráfico de resultados (se cerrará automáticamente para seguir el loop)
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

# Directorio base de modelos
base_model_dir = Path("src") / "methods" / "regresion_robusta"

# Obtener columnas del csv
df = pd.read_csv(DATASET, nrows=0)
columnas = df.columns.tolist()
print(f"Columnas detectadas: {columnas}")

# Verificar target
if TARGET_COLUMN not in columnas:
    raise ValueError(f"La columna '{TARGET_COLUMN}' no se encuentra en el dataset.")

# Cargar y ordenar
df = pd.read_csv(DATASET, parse_dates=["Timestamp"], index_col="Timestamp")
if not df.index.is_monotonic_increasing:
    df = df.sort_index()
    print("Los datos han sido ordenados cronológicamente.")

# Feature Engineering
if ADD_FEATURES_TEMPORALES:
    df.loc[:, 'hour'] = df.index.hour
    df.loc[:, 'day_of_week'] = df.index.dayofweek
df.loc[:, 'minute'] = df.index.minute

if ADD_FEATURES_LAG:
    df.loc[:, f'{TARGET_COLUMN}_lag1'] = df[TARGET_COLUMN].shift(1)

# Target Futuro
if VENTANA_DE_PREDICCION > 0:
    print(f"Generando target para predecir {VENTANA_DE_PREDICCION} minutos a futuro...")
    n_steps_futuro = VENTANA_DE_PREDICCION
    FUTURE_TARGET_COLUMN = f'{TARGET_COLUMN}_future'
    df.loc[:, FUTURE_TARGET_COLUMN] = df[TARGET_COLUMN].shift(-n_steps_futuro)
    Y_COLUMN = FUTURE_TARGET_COLUMN    
else:
    print("No se usará ventana de predicción. Se predice el valor actual.")
    Y_COLUMN = TARGET_COLUMN 

# Limpieza
df = df.dropna()

# Split Train/Test
train_size = int(len(df) * 0.7)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]
print(f"Train size: {len(df_train)} | Test size: {len(df_test)}")

# Selección de Features
features = [col for col in df.columns if col not in [Y_COLUMN, TARGET_COLUMN, 'Timestamp']]
print(f"Features: {features}")

X_train = df_train[features]
y_train = df_train[Y_COLUMN]
X_test = df_test[features]
y_test = df_test[Y_COLUMN]

# Agregar características polinómicas
if ADD_FEATURES_POLYNOMIAL:
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
else:
    X_train_poly = X_train
    X_test_poly = X_test

# Escalado
print("Escalando datos...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

models = {
    "HuberRegressor": HuberRegressor(
        epsilon=1.35,     
        max_iter=2000,    
        alpha=0.0001,     
        tol=1e-06         
    ),

    "RANSAC": RANSACRegressor(
        random_state=None,
        min_samples=None, 
        residual_threshold=None, 
        max_trials=2000, 
        loss='absolute_error' 
    ),

    "TheilSen": TheilSenRegressor(
        random_state=None
    ),

    "QuantileReg_Median": QuantileRegressor(
        quantile=0.5, 
        solver='highs'    
    ) 
}

print("\n--- INICIANDO ENTRENAMIENTO INDIVIDUAL ---")

for model_name, model in models.items():
    print(f"\n Procesando modelo: {model_name} ...")
    
    # Crear carpeta específica para cada modelo
    folder_name = f"results_{nombre_dataset}_{model_name}"
    current_results_dir = base_model_dir / folder_name
    current_results_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Entrenar
        model.fit(X_train_scaled, y_train)
        
        # Predecir
        y_pred = model.predict(X_test_scaled)
        
        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"   > RMSE: {rmse:.4f} | R2: {r2:.4f} | MAE: {mae:.4f}")
        
        # Guardar Métricas en archivo de texto dentro de su carpeta
        metrics_file = current_results_dir / f"metrics_{model_name}_V({VENTANA_DE_PREDICCION})_P({ADD_FEATURES_POLYNOMIAL})_L({ADD_FEATURES_LAG})_T({ADD_FEATURES_TEMPORALES}).txt"
        with open(metrics_file, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Dataset: {nombre_dataset}\n")
            f.write(f"Ventana Prediccion: {VENTANA_DE_PREDICCION}\n")
            f.write("-" * 30 + "\n")
            f.write(f"RMSE: {rmse}\n")
            f.write(f"R2: {r2}\n")
            f.write(f"MAE: {mae}\n")
        
        # Generar y Guardar Gráfico
        if VIEW_GRAPH or SAVE_GRAPH:
            plt.figure(figsize=(14, 6))
            
            # Datos reales
            plt.plot(df_test.index, y_test, label='Valores Reales', color='blue', alpha=0.6)
            
            # Predicción del modelo actual
            plt.plot(df_test.index, y_pred, label=f'Predicción {model_name}', color='red', alpha=0.8, linewidth=1.5)
            
            plt.xlabel('Timestamp')
            plt.ylabel(TARGET_COLUMN)
            
            if VENTANA_DE_PREDICCION > 0:
                titulo = f'{model_name}: {TARGET_COLUMN} ({VENTANA_DE_PREDICCION} min. futuro)'
            else:
                titulo = f'{model_name}: {TARGET_COLUMN} (Actual)'
                
            plt.title(titulo)
            plt.legend(title=f'RMSE={rmse:.3f} | R2={r2:.3f} | MAE={mae:.3f}')
            plt.grid(True, alpha=0.3)
            
            if SAVE_GRAPH:
                graph_filename = f'plot_{model_name}_V({VENTANA_DE_PREDICCION})_P({ADD_FEATURES_POLYNOMIAL})_L({ADD_FEATURES_LAG})_T({ADD_FEATURES_TEMPORALES}).png'
                plt.savefig(current_results_dir / graph_filename)
                print(f"   > Gráfico guardado en: {current_results_dir / graph_filename}")
            
            if VIEW_GRAPH:
                plt.show(block=True) 
                plt.close()
                
    except Exception as e:
        print(f"   ! Error en {model_name}: {e}")

print("\nProceso finalizado. Revisa las subcarpetas creadas.")