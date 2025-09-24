import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# ================================
# 1. Cargar datos 
# ================================
all_sheets = pd.read_excel("water_quality.xlsx", sheet_name=None)
df = pd.concat(all_sheets.values(), ignore_index=True)

df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True)
df = df.sort_values("Timestamp")
df = df.drop(columns=["Cl", "NO3"], errors="ignore")

# ================================
# 2. Crear variables predictoras
# ================================
df["day"] = df["Timestamp"].dt.day
df["hour"] = df["Timestamp"].dt.hour
df["weekday"] = df["Timestamp"].dt.weekday

X = df[["ph", "Temp", "Cond", "Turbidity", "day", "hour", "weekday"]]
y = df["DO"]

X = X.fillna(X.mean())
y = y.fillna(y.mean())

# ================================
# 3. Entrenar modelo con TODOS los datos
# ================================
model = Ridge(alpha=1.0)
model.fit(X, y)

df["y_pred"] = model.predict(X)

# ================================
# 4. Visualización resultados por día
# ================================
for dia, grupo in df.groupby(df["Timestamp"].dt.date):
    mse = mean_squared_error(grupo["DO"], grupo["y_pred"])
    plt.figure(figsize=(10,5))
    plt.plot(grupo["Timestamp"], grupo["DO"], label="Real", marker="o")
    plt.plot(grupo["Timestamp"], grupo["y_pred"], label="Predicho", marker="x")
    plt.xlabel("Hora")
    plt.ylabel("DO (Oxígeno disuelto)")
    plt.title(f"Regresión Ridge - {dia} (MSE={mse:.3f})")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
