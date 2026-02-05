import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pathlib import Path

# ==========================================
# 1. VARIABLES DE CONFIGURACIÓN Y RUTAS
# ==========================================
TIPO_DATASET = 1  # 1: Water Quality, 2: SRU2
ADD_FEATURES_TEMPORALES = True  # Agregar Features Temporales adicionales
ADD_FEATURES_LAG = False  # Agregar Features Lag adicionales
VIEW_GRAPH = True  # Visualizar gráfico de resultados
SAVE_GRAPH = True  # Guardar gráfico de resultados

# Se define el dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Dispositivo seleccionado: {device}')

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
results_dir = model_dir / f"results_{nombre_dataset}"
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
# 3. PREPARACIÓN PARA PYTORCH (SCALING & TENSORS)
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

# Se convierten a tensores
X_train_tensor = torch.tensor(X_train_scaled)
y_train_tensor = torch.tensor(y_train_scaled)
X_test_tensor = torch.tensor(X_test_scaled)
y_test_tensor = torch.tensor(y_test_scaled)

# Se crean los DataLoaders
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size, shuffle=False)

# ==========================================
# 4. DEFINICIÓN DEL MODELO MLP
# ==========================================

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Configuración de hiperparámetros
input_dim = X_train_tensor.shape[1] # Se detecta dinámicamente según las features agregadas
lr = 0.001
epochs = 100

model = MLPModel(input_dim=input_dim, hidden_dim=64, output_dim=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ==========================================
# 5. ENTRENAMIENTO
# ==========================================

train_losses = []
test_losses = [] 

print("Iniciando entrenamiento...")

for epoch in range(epochs):
    model.train()
    train_loss = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.view(-1, 1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Evaluación en Test por época (para monitoreo)
    model.eval()
    test_loss = 0
    with torch.no_grad():      
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.view(-1, 1))
            test_loss += loss.item()
            
    test_loss /= len(test_loader)
    test_losses.append(test_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs} || Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')

# ==========================================
# 6. EVALUACIÓN FINAL Y RESULTADOS
# ==========================================

# Generación de predicciones finales
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch)
        all_preds.append(preds.cpu())
        all_targets.append(y_batch)

# Inversa del escalado
all_preds = scalery.inverse_transform(torch.cat(all_preds).numpy()).flatten()
all_targets = scalery.inverse_transform(torch.cat(all_targets).unsqueeze(1).numpy()).flatten()

# Cálculo de métricas
r2 = r2_score(all_targets, all_preds)
mse = mean_squared_error(all_targets, all_preds)
mae = mean_absolute_error(all_targets, all_preds)
mape = mean_absolute_percentage_error(all_targets, all_preds)

print("\n--- Test Metrics ---")
print(f"R² Score: {r2:.8f}")
print(f"MSE: {mse:.8f}")
print(f"MAE: {mae:.8f}")
print(f"MAPE: {mape:.8f}")

# ==========================================
# 7. GRÁFICOS Y GUARDADO
# ==========================================

if VIEW_GRAPH or SAVE_GRAPH:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Gráfico de Pérdida
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(test_losses, label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Curvas de Aprendizaje')
    axes[0].legend()
    axes[0].grid(True)

    # Gráfico de Predicciones
    # Se grafican todos los puntos
    axes[1].plot(all_targets, label='Real', linewidth=2)
    axes[1].plot(all_preds, label='Predicho', linewidth=2, linestyle='--')
    axes[1].set_xlabel('Muestras')
    axes[1].set_ylabel(TARGET_COLUMN)
    axes[1].set_title(f'Predicción vs Realidad')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if SAVE_GRAPH:
        file_path = results_dir / f"training_results_MLP.png"
        plt.savefig(file_path)
        print(f"Gráfico guardado en: {file_path}")
    else:
        plt.show()