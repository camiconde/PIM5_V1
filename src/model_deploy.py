#src/model_deploy.py

import os
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Agregar 'src' al path de Python para cargar funciones de ft_engineering
sys.path.append(os.path.dirname(__file__))
from ft_engineering import convertir_a_string

# 1. Cargar el modelo guardado
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl')

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Modelo 'model.pkl' cargado exitosamente en FastAPI.")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    model = None

# 2. Inicializar la aplicación FastAPI
app = FastAPI(
    title="API de Evaluación de Riesgo Crediticio",
    description="Endpoint para realizar predicciones de riesgo crediticio (soporta lotes/batch).",
    version="1.0.0"
)

# 3. Definir estructura de entrada para cada cliente
class CreditRequest(BaseModel):
    tipo_credito: int = 4
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    salario_cliente: float
    total_otros_prestamos: float = 0
    cuota_pactada: float
    puntaje_datacredito: float
    cant_creditosvigentes: int = 1
    huella_consulta: int = 1
    creditos_sectorFinanciero: int = 1
    creditos_sectorCooperativo: int = 0
    creditos_sectorReal: int = 0
    promedio_ingresos_datacredito: float
    tipo_laboral: str = "Empleado"
    tendencia_ingresos: str = "Estable"

# Endpoint raíz
@app.get("/")
def home():
    return {"status": "ok", "message": "API de Evaluación de Riesgo Crediticio activa"}

# Endpoint de predicción por lotes
@app.post("/predict")
def predict(requests: List[CreditRequest]):
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo no está disponible.")

    try:
        # Convertir lista de pydantic models a DataFrame
        data_dicts = [req.dict() for req in requests]
        df_input = pd.DataFrame(data_dicts)

        # Aplicar las transformaciones de ingeniería de características
        df_input['ratio_endeudamiento'] = df_input['cuota_pactada'] / (df_input['salario_cliente'] + 1)
        df_input['capacidad_pago_disponible'] = df_input['salario_cliente'] - df_input['cuota_pactada']

        # Predecir
        predictions = model.predict(df_input)
        probabilities = model.predict_proba(df_input)[:, 1]

        # Formatear respuesta
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            results.append({
                "cliente_index": i,
                "prediccion": int(pred),
                "resultado": "Aprobado / Pago a tiempo" if pred == 1 else "Rechazado / Riesgo de mora",
                "probabilidad_pago": round(float(proba), 4)
            })

        return {"total_registros": len(results), "resultados": results}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error durante la predicción: {str(e)}")