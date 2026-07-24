import os
import pandas as pd

# 1. Obtenemos la ruta absoluta de la raíz del proyecto (PIM5_V1)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Como tu archivo 'Base_de_datos.xlsx' está en la raíz de PIM5_V1, lo apuntamos ahí directamente
path_data = os.path.join(BASE_DIR, "Base_de_datos.xlsx")

print(f"Buscando el archivo en: {path_data}")

# 3. Validamos si el archivo realmente existe en esa ruta antes de leerlo
if not os.path.exists(path_data):
    print(f"Error crítico: El archivo no existe en la ruta {path_data}")
else:
    # 4. Cargamos el archivo Excel
    df = pd.read_excel(path_data)
    print(f"¡Dataset cargado con éxito!")
    print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")