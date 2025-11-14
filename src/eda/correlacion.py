# Celda 1: Correlación de Pearson

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuración ---
# Se define la ruta del archivo.
# Se utiliza r'...' (raw string) para interpretar correctamente las barras invertidas '\' en Windows.
ruta_archivo = r'data\water_quality.csv'

# --- Carga y Preparación de Datos ---
try:
    # Se leen los datos del archivo CSV
    df = pd.read_csv(ruta_archivo)
    
    # Se convierte la columna 'Timestamp' a datetime (opcional, pero buena práctica)
    # df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Se seleccionan solo las columnas numéricas para la correlación.
    # Se excluye 'Timestamp' ya que no es una variable numérica para este análisis.
    df_numeric = df.drop(columns=['Timestamp'])

    # --- Cálculo de Correlación ---
    # Se calcula la matriz de correlación de Pearson
    corr_matrix = df_numeric.corr(method='pearson')

    # --- Visualización ---
    # Se configura el tamaño de la figura
    plt.figure(figsize=(10, 8))
    
    # Se genera el heatmap usando seaborn
    sns.heatmap(
        corr_matrix, 
        annot=True,     # Se muestran los valores numéricos en cada celda
        cmap='coolwarm',# Se define el mapa de color (rojo=positivo, azul=negativo)
        fmt='.2f',      # Se formatean los números a 2 decimales
        linewidths=.5   # Se añade una leve separación entre celdas
    )
    
    # Se añade un título al gráfico
    plt.title('Heatmap de Correlación de Pearson')
    
    # Se muestra el gráfico
    plt.show()

except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta especificada: {ruta_archivo}")
except Exception as e:
    print(f"Ocurrió un error al procesar la Celda 1: {e}")