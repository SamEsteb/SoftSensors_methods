"""
EDA (Exploratory Data Analysis) para archivos Excel (Código antiguo basado en el EDA de Matlab)
Como solo lee una única hoja, se asume que el archivo Excel tiene una sola hoja con datos tabulares.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew

def EDA(file_path: Path, output_dir: Path, output_filename: str = "water_quality_stats.xlsx"):
    """
    Realiza un análisis estadístico exploratorio sobre un archivo Excel
    y guarda los resultados en un archivo Excel en la carpeta especificada.

    Parámetros:
        file_path (Path): Ruta al archivo de entrada (ej: Path("data/water_quality.xlsx"))
        output_dir (Path): Carpeta donde se guardará el archivo de salida
        output_filename (str): Nombre del archivo Excel de salida
    """

    # Cargar dataset
    df = pd.read_excel(file_path)

    # Seleccionar solo columnas numéricas
    df_num = df.select_dtypes(include=[np.number])

    # Eliminar columnas que son todo NaN
    df_num = df_num.dropna(axis=1, how="all")

    # Si no queda nada
    if df_num.empty:
        raise ValueError("No hay variables numéricas válidas para analizar.")

    # Diccionario para resultados
    stats = {}

    for col in df_num.columns:
        serie = df_num[col].dropna()
        if serie.empty:
            continue

        stats[col] = {
            "Mean": serie.mean(),
            "Variance": serie.var(),
            "Kurtosis": kurtosis(serie, fisher=False, bias=False),
            "Skewness": skew(serie, bias=False),
            "Mode": serie.mode().iloc[0] if not serie.mode().empty else np.nan,
            "Median": serie.median(),
            "Range": serie.max() - serie.min(),
            "Min": serie.min(),
            "Max": serie.max(),
        }

    # Convertir a DataFrame
    stats_df = pd.DataFrame(stats).T

    # Crear carpeta de salida si no existe
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    stats_df.to_excel(output_path, index=True)

    print(f"✅ Análisis exportado exitosamente a: {output_path}")

if __name__ == "__main__":
    # Ruta al archivo de entrada
    input_file = Path("data/water_quality.xlsx")
    output_dir = Path("src/eda/")
    EDA(input_file, output_dir)