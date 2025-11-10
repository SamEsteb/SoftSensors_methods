import pandas as pd
from pathlib import Path
import logging

def configurar_logging(log_path: Path):
    """Configura el sistema de logging para el script."""
    # Se asegura de que el directorio de logs exista.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_path,
        encoding='utf-8'
    )

def procesar_excel(input_path: Path, output_path: Path):
    """
    Lee, limpia y guarda datos de calidad del agua, registrando cada paso en un log.
    """
    logging.info("===== INICIO DEL PROCESO DE PREPROCESAMIENTO =====")
    logging.info(f"Archivo de entrada: '{input_path}'")
    logging.info(f"Archivo de salida: '{output_path}'")
    
    try:
        # --- 1. Carga de datos desde múltiples hojas ---
        logging.info("Paso 1: Cargando datos desde el archivo Excel...")
        xls = pd.ExcelFile(input_path)
        sheet_names = xls.sheet_names
        logging.info(f"Hojas encontradas: {sheet_names}")
        
        df_list = []
        for sheet in sheet_names:
            logging.info(f"-> Leyendo hoja: '{sheet}'")
            df_sheet = pd.read_excel(xls, sheet_name=sheet, decimal=',')
            df_list.append(df_sheet)
            
        df = pd.concat(df_list, ignore_index=True)
        filas_iniciales = len(df)
        
        logging.info(f"Se cargaron y combinaron {len(df_list)} hojas. Total de filas iniciales: {filas_iniciales}")
        print(f"✅ Se cargaron y combinaron {len(df_list)} hojas con un total de {filas_iniciales} filas.")

    except FileNotFoundError:
        error_msg = f"Error Crítico: El archivo en la ruta '{input_path}' no fue encontrado."
        logging.error(error_msg)
        print(f"❌ {error_msg}")
        return
    except Exception as e:
        error_msg = f"Ocurrió un error inesperado al leer el archivo: {e}"
        logging.error(error_msg, exc_info=True) # exc_info=True registra el traceback completo
        print(f"❌ {error_msg}")
        return

    # --- 2. Corrección y Limpieza de Datos ---
    logging.info("Paso 2: Realizando limpieza y transformaciones...")

    # Se eliminan columnas con nan no necesarias.
    cols_a_eliminar = ['Cl', 'NO3']
    cols_encontradas = [col for col in cols_a_eliminar if col in df.columns]
    if cols_encontradas:
        df.drop(columns=cols_encontradas, inplace=True)
        logging.info(f"Columnas eliminadas: {cols_encontradas}")
    else:
        logging.info("No se encontraron las columnas 'Cl' o 'NO3' para eliminar.")

    # Se manejan los formatos mixtos en la columna Timestamp.
    logging.info("Procesando columna 'Timestamp' con formatos mixtos...")
    original_timestamp = df['Timestamp'].copy()
    excel_serial_dates = pd.to_numeric(original_timestamp, errors='coerce')
    
    # Se cuentan los diferentes tipos de formato de fecha encontrados
    num_serial = excel_serial_dates.notna().sum()
    num_texto = excel_serial_dates.isna().sum()
    logging.info(f"Se encontraron {num_serial} timestamps en formato numérico (serial de Excel).")
    logging.info(f"Se encontraron {num_texto} timestamps en formato de texto ('dd-mm-yyyy hh:mm').")

    # Se muestran ejemplos de la conversión de fechas numéricas
    if num_serial > 0:
        logging.info("Ejemplos de conversión de timestamp numérico a fecha:")
        ejemplos = df[excel_serial_dates.notna()].head(3)
        for index, row in ejemplos.iterrows():
            serial = row['Timestamp']
            convertido = pd.to_datetime(serial, unit='D', origin='1899-12-30')
            logging.info(f"  - Valor original: {serial} -> Convertido: {convertido.strftime('%Y-%m-%d %H:%M:%S')}")

    # Se realiza la conversión
    df['Timestamp'] = pd.to_datetime(excel_serial_dates, unit='D', origin='1899-12-30', errors='coerce')
    fill_values = pd.to_datetime(original_timestamp, format='%d-%m-%Y %H:%M', errors='coerce')
    df['Timestamp'] = df['Timestamp'].fillna(fill_values)

    # Se eliminan filas con valores nulos
    filas_antes_dropna = len(df)
    logging.info(f"Buscando valores nulos... Filas antes de la eliminación: {filas_antes_dropna}")
    df.dropna(inplace=True)
    filas_despues_dropna = len(df)
    filas_eliminadas = filas_antes_dropna - filas_despues_dropna
    
    if filas_eliminadas > 0:
        logging.warning(f"Se eliminaron {filas_eliminadas} filas que contenían al menos un valor nulo.")
    else:
        logging.info("No se eliminaron filas por valores nulos.")

    # Se ordenan los datos
    logging.info("Ordenando el DataFrame por 'Timestamp' de forma ascendente.")
    df.sort_values(by='Timestamp', inplace=True)

    # --- 3. Guardado del archivo procesado ---
    logging.info("Paso 3: Guardando los datos procesados...")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(output_path, index=False, engine='xlsxwriter')
        
        logging.info(f"Se guardaron {len(df)} filas y {len(df.columns)} columnas en '{output_path}'.")
        print(f"✅ Datos procesados y guardados exitosamente en: '{output_path}'")
    except Exception as e:
        error_msg = f"Ocurrió un error al guardar el archivo de salida: {e}"
        logging.error(error_msg, exc_info=True)
        print(f"❌ {error_msg}")

    logging.info("===== FIN DEL PROCESO DE PREPROCESAMIENTO =====")


if __name__ == '__main__':
    # --- Definición de Rutas con Pathlib ---
    base_dir = Path('.') # Directorio actual
    log_dir = base_dir / 'logs'
    log_file = log_dir / 'preprocesamiento_arreglar_excel.log'
    
    # Se configura el logging antes de ejecutar cualquier otra cosa.
    configurar_logging(log_path=log_file)

    data_dir = base_dir / 'data/raw'
    input_file = data_dir / 'water_quality.xlsx'
    processed_dir = data_dir / 'processed'
    output_file = processed_dir / 'water_quality_processed.xlsx'

    # Se llama a la función de procesamiento.
    procesar_excel(input_path=input_file, output_path=output_file)