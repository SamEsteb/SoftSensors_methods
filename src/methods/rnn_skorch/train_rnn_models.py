import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# --- IMPORTACIÓN DE SKORCH ---
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping

# ==========================================
# CONFIGURACIÓN GENERAL
# ==========================================
TIPO_DATASET = 1  # 1: Water Quality, 2: SRU2
ADD_FEATURES_TEMPORALES = True 
ADD_FEATURES_LAG = False 
VIEW_GRAPH = False  
SAVE_GRAPH = True 

# LISTA DE MODELOS A EJECUTAR
MODELOS_A_EVALUAR = ['LSTM', 'GRU', 'RNN'] 

# Configuración de Rutas
base_path = Path(__file__).parent.resolve()
data_dir = base_path.parents[2] / "data"

if TIPO_DATASET == 1:
    nombre_dataset = "water_quality"
    TARGET_COLUMN = "Turbidity"
elif TIPO_DATASET == 2:
    nombre_dataset = "SRU2" 
    TARGET_COLUMN = "AI508"

DATASET = data_dir / f"{nombre_dataset}.csv"

print(f"--- Iniciando proceso para Dataset: {nombre_dataset} ---")

# ==========================================
# CARGA Y PREPROCESAMIENTO 
# ==========================================

if not DATASET.exists():
    print(f"Error: No se encuentra el dataset en {DATASET}")
    sys.exit()

df = pd.read_csv(DATASET, parse_dates=["Timestamp"], index_col="Timestamp")

if not df.index.is_monotonic_increasing:
    df = df.sort_index()

if ADD_FEATURES_TEMPORALES:
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['minute'] = df.index.minute

if ADD_FEATURES_LAG:
    df[f'{TARGET_COLUMN}_lag1'] = df[TARGET_COLUMN].shift(1)

df = df.dropna()

# Split
train_size = int(len(df) * 0.7)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
print(f"Features: {len(feature_cols)} | Train Rows: {len(df_train)} | Test Rows: {len(df_test)}")

X_train_np = df_train[feature_cols].values.astype(np.float32)
y_train_np = df_train[TARGET_COLUMN].values.astype(np.float32)
X_test_np = df_test[feature_cols].values.astype(np.float32)
y_test_np = df_test[TARGET_COLUMN].values.astype(np.float32)

# Scaling
scalerx = StandardScaler()
scalery = StandardScaler()

X_train_scaled = scalerx.fit_transform(X_train_np)
y_train_scaled = scalery.fit_transform(y_train_np.reshape(-1, 1)).flatten()
X_test_scaled = scalerx.transform(X_test_np)
y_test_scaled = scalery.transform(y_test_np.reshape(-1, 1)).flatten()

# Ajuste de forma para Skorch
y_train_reshaped = y_train_scaled.reshape(-1, 1)
y_test_reshaped = y_test_scaled.reshape(-1, 1)

# ==========================================
# DEFINICIÓN DE LA CLASE DE RED NEURONAL (Universal)
# ==========================================

class UniversalRNN(nn.Module):
    """
    Clase flexible que implementa LSTM, GRU o RNN basándose en el parámetro 'kind'.
    """
    def __init__(self, n_features, hidden_dim=64, output_dim=1, kind='LSTM', num_layers=1):
        super(UniversalRNN, self).__init__()
        
        # Selección de arquitectura
        if kind == 'LSTM':
            self.rnn = nn.LSTM(n_features, hidden_dim, num_layers, batch_first=True)
        elif kind == 'GRU':
            self.rnn = nn.GRU(n_features, hidden_dim, num_layers, batch_first=True)
        else: # RNN
            self.rnn = nn.RNN(n_features, hidden_dim, num_layers, batch_first=True)
            
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (Batch, Features)
        # RNN requiere: (Batch, Seq_Len, Features)
        # se agrega una dimension de tiempo artificial
        x = x.unsqueeze(1) 
        
        # out shape: (Batch, Seq_Len, Hidden)
        out, _ = self.rnn(x)
        
        # Tomamos el último paso
        last_step = out[:, -1, :] 
        return self.fc(last_step)

# ==========================================
# BUCLE DE ENTRENAMIENTO
# ==========================================

for modelo_nombre in MODELOS_A_EVALUAR:
    print(f"\n" + "="*40)
    print(f"PROCESANDO MODELO: {modelo_nombre}")
    print("="*40)
    
    # Crear directorio específico para resultados (Ej: results_SRU2_LSTM)
    results_dir = base_path / f"results_{nombre_dataset}_{modelo_nombre}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar el Wrapper Skorch
    net = NeuralNetRegressor(
        module=UniversalRNN,
        module__n_features=X_train_scaled.shape[1],
        module__hidden_dim=64,
        module__kind=modelo_nombre,
        
        criterion=nn.MSELoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.001,
        max_epochs=100,
        batch_size=64,
        
        # Validacion interna del 20% para ver curvas de aprendizaje
        train_split=None if not VIEW_GRAPH else None, 
        
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=1, 
        
        callbacks=[
            # EarlyStopping(patience=15, monitor='valid_loss'), 
        ]
    )
    
    # Entrenar
    print(f"Entrenando {modelo_nombre}...")
    net.fit(X_train_scaled, y_train_reshaped)
    
    # Predecir
    train_preds_scaled = net.predict(X_train_scaled)
    test_preds_scaled = net.predict(X_test_scaled)
    
    # Métricas e Inversión de Escala
    test_preds = scalery.inverse_transform(test_preds_scaled).flatten()
    y_real = y_test_np 
    
    r2 = r2_score(y_real, test_preds)
    mse = mean_squared_error(y_real, test_preds)
    mae = mean_absolute_error(y_real, test_preds)
    mape = mean_absolute_percentage_error(y_real, test_preds)
    
    print(f"Resultados {modelo_nombre} -> MSE: {mse:.4f} | R2: {r2:.4f}")
    
    # Guardar Resultados en Texto
    with open(results_dir / "metrics.txt", "w") as f:
        f.write(f"Model: {modelo_nombre}\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"R2: {r2}\n")
        f.write(f"MAPE: {mape}\n")

    # Gráficos
    if SAVE_GRAPH:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        # Historial de Skorch (Loss)
        history = net.history
        train_loss = history[:, 'train_loss']
        val_loss = history[:, 'valid_loss'] if 'valid_loss' in history[0] else []
        
        axes[0].plot(train_loss, label='Train Loss')
        if len(val_loss) > 0:
            axes[0].plot(val_loss, label='Val Loss')
        axes[0].set_title(f'Curva de Aprendizaje ({modelo_nombre})')
        axes[0].legend()
        axes[0].grid(True)
        
        # Predicción vs Realidad
        axes[1].plot(y_real, label='Real', alpha=0.7)
        axes[1].plot(test_preds, label='Predicho', alpha=0.7, linestyle='--')
        axes[1].set_title(f'Predicción: {modelo_nombre}')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(results_dir / "plot_results.png")
        plt.close() 

print("\n--- Proceso completado ---")