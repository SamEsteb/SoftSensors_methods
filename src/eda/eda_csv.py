"""
Nuevo EDA (Exploratory Data Analysis) para archivos CSV
El archivo CSV de salida tendrá el mismo nombre que el archivo de entrada, pero con el sufijo '_stats.csx'
y se guardará en la carpeta 'src/eda'.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew

def analizar_csv_eda(file_path: Path, output_dir: Path):
    """
    Realiza un análisis estadístico exploratorio sobre un archivo CSV
    y guarda los resultados en un nuevo archivo en src/eda.

    El archivo de salida tendrá el nombre original + '_stats.csx'.

    Parámetros:
        file_path (Path): Ruta al archivo CSV de entrada (ej: Path("data/water_quality.csv"))
    """

    # --- 1. Definir rutas de salida ---
    output_dir.mkdir(parents=True, exist_ok=True)

    base_filename = file_path.stem
    output_filename = f"{base_filename}_stats.csv"
    output_path = output_dir / output_filename

    # --- 2. Cargar dataset ---
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"❌ ERROR: Archivo no encontrado en: {file_path}")
        return
    except Exception as e:
        print(f"❌ ERROR: No se pudo leer el archivo {file_path}. Detalle: {e}")
        return

    # --- 3. Lógica de análisis ---
    
    df_num = df.select_dtypes(include=[np.number])
    df_num = df_num.dropna(axis=1, how="all")

    if df_num.empty:
        print(f"⚠️ ADVERTENCIA: No hay variables numéricas válidas para analizar en {file_path}.")
        return

    stats = {}

    for col in df_num.columns:
        serie = df_num[col].dropna()
        if serie.empty:
            continue
        
        # Se calculan las estadísticas
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

    # --- 4. Guardar resultados ---
    try:
        stats_df.to_csv(output_path, index=True)
        print(f"✅ Análisis exportado exitosamente a: {output_path}")
    except Exception as e:
        print(f"❌ ERROR: No se pudo guardar el archivo en {output_path}. Detalle: {e}")


if __name__ == "__main__":
    # Se define la lista de archivos a procesar
    archivos_a_procesar = [
        Path("data/water_quality.csv"),
        Path("data/SRU2.csv")
    ]
    output_dir = Path("src/eda")

    print("--- Iniciando análisis EDA ---")
    
    for input_file in archivos_a_procesar:
        print(f"\nProcesando: {input_file}...")
        analizar_csv_eda(input_file, output_dir)

    print("\n--- Análisis completado ---")