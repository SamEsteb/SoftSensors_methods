import pandas as pd

# Ruta del archivo original
input_path = "data/raw/SRU2.csv"
output_path = "data/SRU2.csv"

# Leer el CSV original
df = pd.read_csv(input_path)

# Generar la columna de timestamps
start_time = "2008-11-04 00:00:00"
n_rows = len(df)  # cantidad de filas
timestamps = pd.date_range(start=start_time, periods=n_rows, freq="T")  # frecuencia minutal

# Insertar la columna al inicio del DataFrame
df.insert(0, "Timestamp", timestamps)

# Guardar el nuevo CSV
df.to_csv(output_path, index=False)

print(f"Archivo guardado con Timestamp en: {output_path}")
