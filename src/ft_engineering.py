# src/ft_engineering.py()

# Librerías
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer

from cargar_datos import cargarDatos

# 1. Cargamos los datos
df = cargarDatos()

# 2. INGENIERÍA DE CARACTERÍSTICAS (Nuevas variables financieras)
# A. Ratio de Endeudamiento (Cuota vs Salario)
df['ratio_endeudamiento'] = df['cuota_pactada'] / (df['salario_cliente'] + 1)

# B. Capacidad de Pago Disponible
df['capacidad_pago_disponible'] = df['salario_cliente'] - df['cuota_pactada']

# 3. Hagamos el split de Features/Target
# Eliminamos el target y variables que causan Data Leakage o son fechas/IDs
columnas_a_descartar = [
    'Pago_atiempo', 
    'fecha_prestamo', 
    'saldo_mora', 
    'saldo_total', 
    'saldo_principal', 
    'saldo_mora_codeudor',
    'puntaje'  # Mantenemos puntaje_datacredito que es el score previo externo
]

X = df.drop(columns=columnas_a_descartar, errors='ignore') # Features
y = df['Pago_atiempo'] # Target

# 4. Definimos los tipos de variables
num_features = X.select_dtypes('number').columns
cat_features = X.select_dtypes(include=['object', 'string', 'category']).columns

print(f'Features numéricas ({len(num_features)}): {list(num_features)}')
print(f'Features categóricas ({len(cat_features)}): {list(cat_features)}')

# 1. Creamos una función normal fuera del pipeline (en lugar de la lambda)
def convertir_a_string(df_cat):
    return df_cat.astype(str)

# 5. Creamos pipelines para cada ruta
## Ruta 1: Numéricas (Imputación + Escalado)
num_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

# 2. En la Ruta 2 de Categóricas usación la función con nombre:
cat_transformer = Pipeline(
    steps=[
        ('to_str', FunctionTransformer(convertir_a_string)), # <--- Usamos la función normal
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]
)
# 6. Combinamos en ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

# 7. Dividimos los datos en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Aplicamos el preprocesamiento
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

print(f'X_train_preprocessed shape: {X_train_preprocessed.shape}')
print(f'X_test_preprocessed shape: {X_test_preprocessed.shape}')