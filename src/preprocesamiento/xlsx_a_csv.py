import pandas as pd
from pathlib import Path

def convertir_excel_a_csv(excel_file_path: Path):
    """
    Convierte un archivo Excel (.xlsx) a un archivo CSV.

    Parámetros:
        excel_file_path (Path): Ruta al archivo Excel de entrada.
    """
    
    # Se valida que el archivo de entrada exista
    if not excel_file_path.exists():
        print(f"Error: El archivo no se encontró en {excel_file_path}")
        return
    
    # Se cambia .xlsx a .csv 
    csv_file_path = excel_file_path.with_suffix('.csv')

    try:
        # --- Paso 1: Leer el archivo Excel ---
        print(f"Cargando archivo: {excel_file_path.name}...")
        df = pd.read_excel(excel_file_path)

        # --- Paso 2: Guardar en archivo CSV ---
        df.to_csv(csv_file_path, index=False)
        
        print(f"Archivo convertido y guardado en:\n   {csv_file_path}")

    except Exception as e:
        print(f"Ocurrió un error durante la conversión: {e}")

# --- BLOQUE DE EJECUCIÓN ---
if __name__ == "__main__":

    archivo_a_convertir = Path("data/raw/processed/water_quality_processed.xlsx") 
    convertir_excel_a_csv(archivo_a_convertir)