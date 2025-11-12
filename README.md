# SoftSensors_methods
Recopilación de métodos convencionales de Soft Sensors para estimar la variable "turbidity" (water quality) y "AI508" (SRU)

## Descripción
- Proyecto para agrupar implementaciones y análisis de métodos clásicos de Soft Sensors orientados a la estimación de turbidez.
- Enfocado en reproducibilidad: datos ya preprocesados en /data y scripts organizados en /src.

## Objetivos
- Centralizar datasets y código para experimentación.
- Documentar análisis exploratorio (EDA), preprocesamiento y cada método por separado.
- Facilitar la incorporación de nuevos métodos (cada uno en su propia carpeta bajo src/methods).

## Estructura del repositorio
```
/ (raíz)
├── .gitignore
├── README.md
📁 data/
│   📁 raw/
│   │   └── water_quality.xlsx
│   📁 processed/
│   │   └── water_quality_processed.xlsx
│   ├── SRU2.csv
│   └── water_quality.csv
📁 src/
    📁 eda/
    │   ├── eda_csv.py
    │   ├── eda_xlsx.py
    │   ├── SRU2_stats.csv
    │   └── water_quality_stats.csv
    📁 extra/
    │   └── directorio.py
    📁 methods/
    │   └── <cada método tiene su propia carpeta>
    📁 preprocesamiento/
        ├── corregir_xlsx.py
        └── xlsx_a_csv.py
```

Notas clave
- Datasets principales para entrenar y evaluar: SRU2.csv y water_quality.csv.
- En data/raw está el archivo original (water_quality.xlsx). Los archivos en data/ son los ya preparados para uso.
- En src/eda se guardan los scripts y resultados del análisis exploratorio (estadísticas resumidas).
- Cada método debe tener una carpeta propia en src/methods que incluya: código de entrenamiento/inferencia, README con descripción del método y requisitos.
- Los scripts en src/preprocesamiento sirven para convertir/corregir el Excel; se asume que los CSV ya están listos para usar.

## Notas a Futuro

En EDA estaría joya agregar un análisis de correlación y causalidad, para agregar en el Paper.