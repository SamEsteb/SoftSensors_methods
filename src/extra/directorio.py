"""
Módulo para generar un archivo con la estructura jerárquica de un directorio para el README.
"""

import os

def generar_arbol_directorio(
    ruta_base: str,
    archivo_salida: str = "directorio.txt",
    ignorar_carpetas: list = None,
    ignorar_extensiones: list = None,
    profundidad_maxima: int = None
):
    """
    Genera un archivo con la estructura del directorio en formato jerárquico.

    Parámetros:
        ruta_base (str): Ruta del directorio raíz.
        archivo_salida (str): Nombre del archivo de salida.
        ignorar_carpetas (list): Carpetas a ignorar (por nombre exacto).
        ignorar_extensiones (list): Extensiones de archivo a ignorar (sin punto, ej: 'log', 'tmp').
        profundidad_maxima (int): Número máximo de niveles de subcarpetas a explorar.
    """

    if ignorar_carpetas is None:
        ignorar_carpetas = []
    if ignorar_extensiones is None:
        ignorar_extensiones = []

    def recorrer_directorio(ruta, prefijo="", nivel=0):
        if profundidad_maxima is not None and nivel > profundidad_maxima:
            return []

        elementos = []
        try:
            for entrada in sorted(os.listdir(ruta)):
                ruta_completa = os.path.join(ruta, entrada)
                if os.path.isdir(ruta_completa):
                    if entrada in ignorar_carpetas:
                        continue
                    elementos.append(f"{prefijo}📁 {entrada}/")
                    elementos.extend(
                        recorrer_directorio(ruta_completa, prefijo + "│   ", nivel + 1)
                    )
                else:
                    extension = os.path.splitext(entrada)[1][1:].lower()
                    if extension in ignorar_extensiones:
                        continue
                    elementos.append(f"{prefijo}├── {entrada}")
        except PermissionError:
            elementos.append(f"{prefijo}⚠️ [Sin permisos para acceder]")
        return elementos

    estructura = recorrer_directorio(ruta_base)
    with open(archivo_salida, "w", encoding="utf-8") as f:
        f.write(f"Estructura del directorio: {ruta_base}\n\n")
        f.write("\n".join(estructura))

    print(f"✅ Archivo '{archivo_salida}' generado correctamente.")


if __name__ == "__main__":
    generar_arbol_directorio(
        ruta_base=".",  # Directorio actual
        archivo_salida="directorio.txt",
        ignorar_carpetas=[".git", "__pycache__"],
        ignorar_extensiones=["log", "tmp"],
        profundidad_maxima=3  # Número de niveles de subcarpetas
    )
