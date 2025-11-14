"""
Crea gráficos de Time Series, Histogramas y Boxplots Semanales para cada variable de los datasets
de water_quality y SRU2.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generar_graficos_eda(filepath, dataset_name):
    """
    Carga un dataset, itera sobre sus columnas numéricas y genera tres gráficos
    para cada una: serie de tiempo, histograma y boxplot semanal.
    Los gráficos se guardan en una estructura de directorios específica.
    """
    
    print(f"--- Procesando dataset: {dataset_name} ---")
    
    # --- 1. Carga y Preparación de Datos ---
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta: {filepath}")
        return
    except Exception as e:
        print(f"Error al leer el archivo {filepath}: {e}")
        return

    # Se verifica si la columna 'Timestamp' existe
    if 'Timestamp' not in df.columns:
        print(f"Error: El dataset {dataset_name} no contiene una columna 'Timestamp'.")
        return
        
    try:
        # Se convierte la columna 'Timestamp' a formato datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    except Exception as e:
        print(f"Error al convertir 'Timestamp' a datetime en {dataset_name}: {e}")
        return

    # Se establece 'Timestamp' como el índice del DataFrame
    df.set_index('Timestamp', inplace=True)
    
    # Se obtienen los nombres de las columnas (variables) a graficar
    variables = df.columns
    
    base_output_dir = os.path.join('src', 'eda', f'graficos_{dataset_name}')
    
    # --- 2. Generación de Gráficos por Variable ---
    
    # Se itera sobre cada variable para generar los gráficos
    for variable in variables:
        if not pd.api.types.is_numeric_dtype(df[variable]):
            print(f"Omitiendo columna no numérica: {variable}")
            continue
            
        print(f"Generando gráficos para: {variable}...")
        
        # Se crea el directorio específico para la variable
        var_dir = os.path.join(base_output_dir, variable)
        os.makedirs(var_dir, exist_ok=True)
        
        # Se eliminan valores NaN para esta variable específica antes de graficar
        data_to_plot = df[[variable]].dropna()

        if data_to_plot.empty:
            print(f"No hay datos válidos para graficar en la variable: {variable}")
            continue

        # --- Gráfico 1: Series de Tiempo (Timestamp vs. Value) ---
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(data_to_plot.index, data_to_plot[variable])
            plt.title(f'Serie de Tiempo: {variable}', fontsize=16)
            plt.xlabel('Timestamp', fontsize=12)
            plt.ylabel(variable, fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            save_path = os.path.join(var_dir, f'timeseries_{variable}.png')
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"Error al generar gráfico de series de tiempo para {variable}: {e}")
            plt.close()

        # --- Gráfico 2: Histograma de Frecuencias ---
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(data_to_plot[variable], kde=True, bins=30)
            plt.title(f'Histograma de Frecuencias: {variable}', fontsize=16)
            plt.xlabel(variable, fontsize=12)
            plt.ylabel('Frecuencia', fontsize=12)
            plt.tight_layout()
            save_path = os.path.join(var_dir, f'histogram_{variable}.png')
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"Error al generar histograma para {variable}: {e}")
            plt.close()

        # --- Gráfico 3: Boxplot Semanal ---
        try:
            plt.figure(figsize=(15, 7))
            
            df_temp = data_to_plot.copy()
            df_temp['Week'] = df_temp.index.to_period('W').astype(str)
            unique_weeks = sorted(df_temp['Week'].unique())
            
            ax = sns.boxplot(x='Week', y=variable, data=df_temp, order=unique_weeks)
            plt.title(f'Boxplot Semanal: {variable}', fontsize=16)
            plt.xlabel('Semana', fontsize=12)
            plt.ylabel(variable, fontsize=12)
            
            # Se maneja el etiquetado del eje X si hay muchas semanas
            if len(unique_weeks) > 10:
                tick_spacing = max(1, len(unique_weeks) // 10)
                ticks_to_show = ax.get_xticks()[::tick_spacing]
                labels_to_show = [label.get_text() for i, label in enumerate(ax.get_xticklabels()) if i % tick_spacing == 0]
                
                ax.set_xticks(ticks_to_show)
                ax.set_xticklabels(labels_to_show, rotation=45, ha='right')
            else:
                plt.xticks(rotation=45, ha='right')

            plt.tight_layout()
            save_path = os.path.join(var_dir, f'boxplot_weekly_{variable}.png')
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"Error al generar boxplot semanal para {variable}: {e}")
            plt.close()

    print(f"--- Procesamiento de {dataset_name} finalizado. ---\n")

# --- 3. Ejecución del Proceso ---

# Se define la lista de datasets a procesar
datasets_a_procesar = [
    {'filepath': r'data\water_quality.csv', 'name': 'water_quality'},
    {'filepath': r'data\SRU2.csv', 'name': 'sru2'}
]

# Se itera sobre la lista y se llama a la función principal
for dataset in datasets_a_procesar:
    generar_graficos_eda(dataset['filepath'], dataset['name'])

print("Proceso de generación de gráficos completado.")