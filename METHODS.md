# Métodos de Regresión Implementados

Este documento describe los métodos de regresión implementados en este repositorio para predicción de sensores blandos (soft sensors).

## Tabla de Contenidos

1. [Métodos Lineales](#métodos-lineales)
   - [MLR - Regresión Lineal Múltiple](#mlr---regresión-lineal-múltiple)
   - [Ridge Regression](#ridge-regression)
   - [Lasso Regression](#lasso-regression)
   - [Elastic Net](#elastic-net)
   - [Regresión Robusta](#regresión-robusta)
2. [Métodos de Árboles](#métodos-de-árboles)
   - [Decision Tree](#decision-tree)
   - [Random Forest](#random-forest)
   - [XGBoost](#xgboost)
3. [Métodos Basados en Kernels](#métodos-basados-en-kernels)
   - [SVR - Support Vector Regression](#svr---support-vector-regression)
   - [GPR - Gaussian Process Regression](#gpr---gaussian-process-regression)
   - [SGPR - Sparse Gaussian Process Regression](#sgpr---sparse-gaussian-process-regression)
   - [Nystroem + SGD (Aproximación Escalable)](#nystroem--sgd-aproximación-escalable)
4. [Redes Neuronales](#redes-neuronales)
   - [MLP - Multi-Layer Perceptron (PyTorch)](#mlp---multi-layer-perceptron-pytorch)
   - [MLP - Multi-Layer Perceptron (Sklearn)](#mlp---multi-layer-perceptron-sklearn)
   - [RNN/LSTM/GRU (Skorch)](#rnnlstmgru-skorch)

---

## Métodos Lineales

### MLR - Regresión Lineal Múltiple

| Característica | Detalle |
|----------------|---------|
| **Archivo** | `src/methods/mlr/train_mlr.py` |
| **Biblioteca** | `sklearn.linear_model.LinearRegression` |
| **Tipo de modelo** | Regresión lineal simple |
| **Regularización** | Ninguna |
| **Input** | Features del dataset + features temporales (opcionales) |
| **Output** | Predicción continua del target |
| **Preprocesamiento** | StandardScaler (opcional, no afecta resultados) |
| **División datos** | 70% train, 30% test (secuencial) |

**Parámetros:**
- `n_jobs=-1`: Uso de todos los procesadores

**Métricas:**
- RMSE (Root Mean Squared Error)
- R² Score
- MAE (Mean Absolute Error)

**Visualizaciones:**
- Gráfico de predicción vs valores reales
- Gráfico de coeficientes del modelo

---

### Ridge Regression

| Característica | Detalle |
|----------------|---------|
| **Archivo** | `src/methods/regression_ridge/train_regression_ridge.py` |
| **Biblioteca** | `sklearn.linear_model.Ridge` |
| **Tipo de modelo** | Regresión lineal con regularización L2 |
| **Regularización** | L2 (Ridge) |
| **Input** | Features del dataset + features temporales (opcionales) |
| **Output** | Predicción continua del target |
| **Preprocesamiento** | StandardScaler |
| **División datos** | 70% train, 30% test (secuencial) |

**Parámetros:**
- `alpha=1.0`: Factor de regularización

**Métricas:**
- RMSE, R² Score, MAE

**Visualizaciones:**
- Gráfico de predicción vs valores reales
- Gráfico de coeficientes

---

### Lasso Regression

| Característica | Detalle |
|----------------|---------|
| **Archivo** | `src/methods/regression_lasso/train_regression_lasso.py` |
| **Biblioteca** | `sklearn.linear_model.Lasso` |
| **Tipo de modelo** | Regresión lineal con regularización L1 |
| **Regularización** | L1 (permite selección de features) |
| **Input** | Features del dataset + features temporales (opcionales) |
| **Output** | Predicción continua del target |
| **Preprocesamiento** | StandardScaler |
| **División datos** | 70% train, 30% test (secuencial) |

**Parámetros:**
- `alpha=0.1`: Factor de regularización
- `max_iter=2000`: Máximo de iteraciones

**Métricas:**
- RMSE, R² Score, MAE

**Visualizaciones:**
- Gráfico de predicción vs valores reales
- Gráfico de coeficientes (algunas features pueden tener coeficiente 0)

---

### Elastic Net

| Característica | Detalle |
|----------------|---------|
| **Archivo** | `src/methods/elastic_net/train_elastic_net.py` |
| **Biblioteca** | `sklearn.linear_model.ElasticNet` |
| **Tipo de modelo** | Regresión lineal con regularización L1 + L2 |
| **Regularización** | Combinación de L1 y L2 |
| **Input** | Features del dataset + features temporales (opcionales) |
| **Output** | Predicción continua del target |
| **Preprocesamiento** | StandardScaler |
| **División datos** | 70% train, 30% test (secuencial) |

**Parámetros:**
- `alpha=0.1`: Factor de regularización
- `l1_ratio=0.5`: Balance entre L1 y L2 (0 = Ridge, 1 = Lasso)
- `max_iter=2000`: Máximo de iteraciones

**Métricas:**
- RMSE, R² Score, MAE

---

### Regresión Robusta

| Característica | Detalle |
|----------------|---------|
| **Archivo** | `src/methods/regresion_robusta/train_regresion_robusta.py` |
| **Bibliotecas** | `sklearn.linear_model` (HuberRegressor, RANSACRegressor, TheilSenRegressor, QuantileRegressor) |
| **Tipo de modelo** | Regresión robusta (múltiples variantes) |
| **Regularización** | Varía según el modelo |
| **Input** | Features + features temporales + features polinómicas (degree=2) |
| **Output** | Predicción continua del target |
| **Preprocesamiento** | StandardScaler + PolynomialFeatures |
| **División datos** | 70% train, 30% test (secuencial) |

**Modelos incluidos:**
1. **HuberRegressor**: Regresión con pérdida Huber (menos sensible a outliers)
2. **RANSACRegressor**: RANSAC (Random Sample Consensus) - detecta inliers/outliers
3. **TheilSenRegressor**: Regresión Theil-Sen (mediana de pendientes)
4. **QuantileRegressor**: Regresión por cuantiles

**Parámetros por modelo:**
- **Huber**: `epsilon=1.35`, `max_iter=2000`, `alpha=0.0001`
- **RANSAC**: `max_trials=2000`, `loss='absolute_error'`
- **TheilSen**: `random_state=None`
- **Quantile**: `quantile=0.5` (mediana), `solver='highs'`

**Métricas:**
- RMSE, R² Score, MAE

---

## Métodos de Árboles

### Decision Tree

| Característica | Detalle |
|----------------|---------|
| **Archivo** | `src/methods/decision_tree/train_decision_tree.py` |
| **Biblioteca** | `sklearn.tree.DecisionTreeRegressor` |
| **Tipo de modelo** | Árbol de decisión para regresión |
| **Input** | Features del dataset + features temporales (opcionales) |
| **Output** | Predicción continua del target |
| **Preprocesamiento** | No requiere escalado |
| **División datos** | 70% train, 30% test (secuencial) |

**Parámetros:**
- `max_depth=20`: Profundidad máxima del árbol
- `min_samples_leaf=50`: Mínimo de muestras por hoja
- `min_samples_split=30`: Mínimo de muestras para split
- `max_features=None`: Usa todas las features

**Métricas:**
- RMSE, R² Score, MAE

**Visualizaciones:**
- Gráfico de predicción vs valores reales
- Gráfico de importancia de features

---

### Random Forest

| Característica | Detalle |
|----------------|---------|
| **Archivo** | `src/methods/random_forest/train_rf.py` |
| **Biblioteca** | `sklearn.ensemble.RandomForestRegressor` |
| **Tipo de modelo** | Ensemble de árboles de decisión |
| **Input** | Features + features temporales + features lag (opcionales) |
| **Output** | Predicción continua del target |
| **Preprocesamiento** | No requiere escalado |
| **División datos** | 70% train, 30% test (secuencial) |
| **Ventana de predicción** | Opcional (predicción a N minutos en el futuro) |

**Parámetros:**
- `n_estimators=200`: Número de árboles
- `random_state=None`: Semilla aleatoria
- `n_jobs=-1`: Uso de todos los procesadores

**Métricas:**
- RMSE, R² Score, MAE

**Visualizaciones:**
- Gráfico de predicción vs valores reales
- Gráfico de importancia de features

---

### XGBoost

| Característica | Detalle |
|----------------|---------|
| **Archivo** | `src/methods/xgboost/train_xgboost.py` |
| **Biblioteca** | `xgboost.XGBRegressor` |
| **Tipo de modelo** | Gradient Boosting optimizado |
| **Input** | Features + features temporales + features lag (opcionales) |
| **Output** | Predicción continua del target |
| **Preprocesamiento** | No requiere escalado |
| **División datos** | 70% train, 30% test (secuencial) |
| **Ventana de predicción** | Opcional (predicción a N minutos en el futuro) |

**Parámetros:**
- `n_estimators=200`: Número de boosting rounds
- `objective='reg:squarederror'`: Función de pérdida
- `random_state=None`: Semilla aleatoria
- `n_jobs=-1`: Uso de todos los procesadores

**Métricas:**
- RMSE, R² Score, MAE

**Visualizaciones:**
- Gráfico de predicción vs valores reales
- Gráfico de importancia de features

---

## Métodos Basados en Kernels

### SVR - Support Vector Regression

| Característica | Detalle |
|----------------|---------|
| **Archivo** | `src/methods/svr/train_svr.py` |
| **Biblioteca** | `sklearn.svm.SVR` |
| **Tipo de modelo** | Support Vector Regression |
| **Input** | Features + features temporales + features lag (opcionales) |
| **Output** | Predicción continua del target |
| **Preprocesamiento** | StandardScaler (requerido) |
| **División datos** | 70% train, 30% test (secuencial) |
| **Ventana de predicción** | Opcional |

**Parámetros:**
- `kernel`: 'linear', 'poly', 'rbf' (default), 'sigmoid'
- `C=1.0`: Parámetro de regularización
- `epsilon=0.1`: Tubo de tolerancia

**Métricas:**
- RMSE, R² Score, MAE

**Notas:**
- Con kernel 'linear' proporciona coeficientes de importancia
- Sensible a la escala de los datos

---

### GPR - Gaussian Process Regression

| Característica | Detalle |
|----------------|---------|
| **Archivo** | `src/methods/gpr/train_gpr.py` |
| **Biblioteca** | `sklearn.gaussian_process.GaussianProcessRegressor` |
| **Tipo de modelo** | Proceso Gaussiano para regresión |
| **Input** | Features + features temporales + features lag (opcionales) |
| **Output** | Predicción + intervalo de incertidumbre (95% CI) |
| **Preprocesamiento** | StandardScaler |
| **División datos** | 70% train, 30% test (secuencial) |
| **Limitación** | O(N³) - máximo 2000 muestras de entrenamiento |

**Parámetros:**
- `kernel`: RBF + WhiteKernel
- `n_restarts_optimizer=2`: Reinicios del optimizador
- `normalize_y=True`: Normalización de variable objetivo

**Kernel utilizado:**
```
C * RBF(length_scale=1.0) + WhiteKernel(noise_level=1)
```

**Métricas:**
- RMSE, R² Score, MAE

**Visualizaciones:**
- Predicción con intervalo de confianza del 95%

---

### SGPR - Sparse Gaussian Process Regression

| Característica | Detalle |
|----------------|---------|
| **Archivo** | `src/methods/gpr/train_sgpr.py` |
| **Biblioteca** | `GPy.models.SparseGPRegression` |
| **Tipo de modelo** | Proceso Gaussiano Escaso (Sparse GP) |
| **Input** | Features + features temporales + features lag (opcionales) |
| **Output** | Predicción + intervalo de incertidumbre (95% CI) |
| **Preprocesamiento** | StandardScaler |
| **División datos** | 70% train, 30% test (secuencial) |
| **Ventaja** | Escala mejor que GPR estándar |

**Parámetros:**
- `num_inducing=500`: Número de puntos inducidos
- `kernel`: RBF

**Instalación requerida:**
```bash
conda create --name GPy_Env python=3.9 -y
conda install -c conda-forge gpy pandas matplotlib scikit-learn openpyxl -y
```

**Métricas:**
- RMSE, R² Score, MAE

**Visualizaciones:**
- Predicción con intervalo de confianza del 95%

---

### Nystroem + SGD (Aproximación Escalable)

| Característica | Detalle |
|----------------|---------|
| **Archivo** | `src/methods/gpr/train_nystroem_sgd.py` |
| **Bibliotecas** | `sklearn.kernel_approximation.Nystroem` + `sklearn.linear_model.SGDRegressor` |
| **Tipo de modelo** | Aproximación de kernel escalable |
| **Input** | Features + features temporales + features lag (opcionales) |
| **Output** | Predicción continua del target |
| **Preprocesamiento** | StandardScaler |
| **División datos** | 70% train, 30% test (secuencial) |

**Arquitectura:**
1. **Nystroem**: Aproxima el feature map de un kernel RBF
2. **SGDRegressor**: Regresor lineal con gradiente descendente estocástico

**Parámetros:**
- `n_components=2000`: Número de componentes Nystroem
- `kernel='rbf'`: Kernel utilizado
- `gamma=0.2`: Parámetro del kernel RBF
- `max_iter=5000`: Iteraciones máximas SGD
- `early_stopping=True`: Parada temprana

**Métricas:**
- RMSE, R² Score, MAE

---

## Redes Neuronales

### MLP - Multi-Layer Perceptron (PyTorch)

| Característica | Detalle |
|----------------|---------|
| **Archivo** | `src/methods/mlp/train_mlp.py` |
| **Biblioteca** | `torch` (PyTorch) |
| **Tipo de modelo** | Red neuronal feedforward |
| **Input** | Features + features temporales + features lag (opcionales) |
| **Output** | Predicción continua del target |
| **Preprocesamiento** | StandardScaler (X e y) |
| **División datos** | 70% train, 30% test (secuencial) |

**Arquitectura:**
```
Input -> Linear(64) -> ReLU -> Linear(64) -> ReLU -> Linear(1)
```

**Parámetros:**
- `hidden_dim=64`: Dimensión de capas ocultas
- `lr=0.001`: Learning rate
- `epochs=100`: Número de épocas
- `batch_size=64`: Tamaño de batch
- `optimizer`: Adam
- `criterion`: MSELoss

**Métricas:**
- R² Score
- MSE
- MAE
- MAPE (Mean Absolute Percentage Error)

**Visualizaciones:**
- Curvas de aprendizaje (train/test loss)
- Predicción vs valores reales

---

### MLP - Multi-Layer Perceptron (Sklearn)

| Característica | Detalle |
|----------------|---------|
| **Archivo** | `src/methods/mlp/train_mlp_sklearn.py` |
| **Biblioteca** | `sklearn.neural_network.MLPRegressor` |
| **Tipo de modelo** | Red neuronal feedforward (sklearn) |
| **Input** | Features + features temporales + features lag (opcionales) |
| **Output** | Predicción continua del target |
| **Preprocesamiento** | StandardScaler (X e y) |
| **División datos** | 70% train, 30% test (secuencial) |

**Arquitectura:**
```
Input -> Dense(64) -> ReLU -> Dense(64) -> ReLU -> Dense(1)
```

**Parámetros:**
- `hidden_layer_sizes=(64, 64)`: Capas ocultas
- `activation='relu'`: Función de activación
- `solver='adam'`: Optimizador
- `alpha=0.0001`: Regularización L2
- `batch_size=64`: Tamaño de batch
- `learning_rate_init=0.001`: Learning rate
- `max_iter=100`: Máximo de iteraciones

**Métricas:**
- R² Score
- MSE
- MAE
- MAPE

**Visualizaciones:**
- Curva de pérdida
- Predicción vs valores reales

---

### RNN/LSTM/GRU (Skorch)

| Característica | Detalle |
|----------------|---------|
| **Archivo** | `src/methods/rnn_skorch/train_rnn_models.py` |
| **Bibliotecas** | `torch` + `skorch` (wrapper) |
| **Tipo de modelo** | Redes neuronales recurrentes |
| **Modelos disponibles** | LSTM, GRU, RNN |
| **Input** | Features + features temporales + features lag (opcionales) |
| **Output** | Predicción continua del target |
| **Preprocesamiento** | StandardScaler (X e y) |
| **División datos** | 70% train, 30% test (secuencial) |

**Arquitectura:**
```
Input -> RNN/LSTM/GRU(hidden_dim=64) -> Linear(64, 1)
```

**Clase implementada:** `UniversalRNN`
- Soporta LSTM, GRU y RNN
- Agrega dimensión de secuencia artificial (seq_len=1)

**Parámetros:**
- `hidden_dim=64`: Dimensión oculta
- `num_layers=1`: Número de capas
- `optimizer`: Adam
- `lr=0.001`: Learning rate
- `max_epochs=100`:Épocas
- `batch_size=64`: Tamaño de batch

**Métricas:**
- MSE
- MAE
- R² Score
- MAPE

**Visualizaciones:**
- Curvas de aprendizaje (train loss)
- Predicción vs valores reales

---

## Configuración General (Common)

Todos los métodos comparten ciertas configuraciones comunes:

### Features Temporales (Opcional)
Si `ADD_FEATURES_TEMPORALES = True`:
- `hour`: Hora del día (0-23)
- `day_of_week`: Día de la semana (0-6)
- `minute`: Minuto de la hora (0-59)

### Features Lag (Opcional)
Si `ADD_FEATURES_LAG = True`:
- `{TARGET_COLUMN}_lag1`: Valor anterior del target (shift de 1)

### Ventana de Predicción (Algunos modelos)
Si `VENTANA_DE_PREDICCION > 0`:
- Crea un target shifted para predecir N minutos en el futuro
- `{TARGET_COLUMN}_future`: Target desplazado N posiciones hacia atrás

### Datasets Soportados
- **Water Quality**: `TARGET_COLUMN = "Turbidity"`
- **SRU2**: `TARGET_COLUMN = "AI508"`

### División de Datos
- **Entrenamiento**: 70% (primeros datos)
- **Prueba**: 30% (últimos datos)
- **Orden**: Secuencial (respetando orden temporal, sin shuffle)

---

## Métricas Comunes

| Métrica | Descripción |
|---------|-------------|
| **MSE** | Mean Squared Error - Error cuadrático medio |
| **RMSE** | Root Mean Squared Error - Raíz del error cuadrático medio |
| **MAE** | Mean Absolute Error - Error absoluto medio |
| **R² Score** | Coeficiente de determinación (0-1, mayor es mejor) |
| **MAPE** | Mean Absolute Percentage Error - Error porcentual absoluto medio |

---

## Requisitos

```
pandas
numpy
matplotlib
scikit-learn
torch
xgboost
gpytorch      # Para algunos modelos de GPR
skorch        # Para modelos RNN
```

Para Sparse GP:
```
conda install -c conda-forge gpy
```
