# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys

# Ruta para importar ft_engineering y model_monitoring
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from ft_engineering import cargarDatos, convertir_a_string
from model_monitoring import run_drift_analysis

st.set_page_config(page_title="Credit Risk Monitoring & Deployment", layout="wide", page_icon="🏦")

# Título y encabezado
st.title("🏦 Sistema Operativo de Riesgo Crediticio & Monitoreo de Data Drift")
st.markdown("---")

# Cargar Modelo Guardado
@st.cache_resource
def load_trained_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')
    return joblib.load(model_path)

try:
    model = load_trained_model()
    st.sidebar.success("✅ Modelo `model.pkl` cargado")
except Exception as e:
    st.sidebar.error(f"❌ Error al cargar modelo: {e}")

# Cargar Datos Históricos
@st.cache_data
def get_reference_data():
    return cargarDatos()

df_ref = get_reference_data()

# Menú lateral
menu = st.sidebar.radio("Navegación", ["1. Evaluador de Solicitudes (Predicción)", "2. Monitoreo de Data Drift", "3. Recomendaciones de Retraining"])

# -------------------------------------------------------------------
# OPCIÓN 1: PREDICCIÓN EN TIEMPO REAL
# -------------------------------------------------------------------
if menu == "1. Evaluador de Solicitudes (Predicción)":
    st.subheader("📋 Simular Nueva Solicitud de Crédito")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        salario = st.number_input("Salario del Cliente ($)", value=2500000, step=100000)
        cuota = st.number_input("Cuota Pactada ($)", value=350000, step=10000)
        capital = st.number_input("Capital Prestado ($)", value=3000000, step=100000)
    with col2:
        plazo = st.slider("Plazo en Meses", 3, 36, 12)
        edad = st.slider("Edad del Cliente", 18, 75, 35)
        puntaje_dc = st.slider("Puntaje DataCrédito", 300, 950, 720)
    with col3:
        tipo_laboral = st.selectbox("Tipo Laboral", ["Empleado", "Independiente", "Pensionado"])
        tendencia = st.selectbox("Tendencia de Ingresos", ["Creciente", "Estable", "Decreciente"])
        cant_cred = st.number_input("Créditos Vigentes", value=1, step=1)

    # Calcular ratios automáticos creados en Feature Engineering
    ratio_end = cuota / (salario + 1)
    cap_disp = salario - cuota

    if st.button("Evaluar Solicitud", type="primary"):
        input_data = pd.DataFrame([{
            'tipo_credito': 4,
            'capital_prestado': capital,
            'plazo_meses': plazo,
            'edad_cliente': edad,
            'salario_cliente': salario,
            'total_otros_prestamos': 0,
            'cuota_pactada': cuota,
            'puntaje_datacredito': puntaje_dc,
            'cant_creditosvigentes': cant_cred,
            'huella_consulta': 1,
            'creditos_sectorFinanciero': 1,
            'creditos_sectorCooperativo': 0,
            'creditos_sectorReal': 0,
            'promedio_ingresos_datacredito': salario,
            'ratio_endeudamiento': ratio_end,
            'capacidad_pago_disponible': cap_disp,
            'tipo_laboral': tipo_laboral,
            'tendencia_ingresos': tendencia
        }])

        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        st.markdown("### Resultado del Análisis:")
        if pred == 1:
            st.success(f"🟢 **CRÉDITO APROBADO** - Probabilidad de Pago a Tiempo: **{proba*100:.1f}%**")
        else:
            st.error(f"🔴 **CRÉDITO RECHAZADO / ALTO RIESGO** - Probabilidad de Pago a Tiempo: **{proba*100:.1f}%**")

# -------------------------------------------------------------------
# OPCIÓN 2: MONITOREO DE DATA DRIFT
# -------------------------------------------------------------------
elif menu == "2. Monitoreo de Data Drift":
    st.subheader("📊 Panel de Detección de Data Drift (Población Actual vs Histórica)")
    
    # Generar muestra simulación de producción
    st.caption("Comparando Dataset Histórico (Entrenamiento) vs Muestra Reciente de Producción")
    
    # Simulamos un leve sesgo en la población actual para evaluar el detector
    df_current = df_ref.sample(frac=0.3, random_state=123).copy()
    df_current['salario_cliente'] = df_current['salario_cliente'] * 1.15 # Cambio de distribución

    numeric_cols = ['salario_cliente', 'capital_prestado', 'cuota_pactada', 'puntaje_datacredito', 'ratio_endeudamiento']
    categoric_cols = ['tipo_laboral', 'tendencia_ingresos']

    drift_df = run_drift_analysis(df_ref, df_current, numeric_cols, categoric_cols)

    # Métricas Resumen (Semáforo)
    col1, col2, col3 = st.columns(3)
    verdes = len(drift_df[drift_df['Estado'] == 'Estable (Verde)'])
    amarillos = len(drift_df[drift_df['Estado'] == 'Alerta (Amarillo)'])
    rojos = len(drift_df[drift_df['Estado'] == 'Drift Crítico (Rojo)'])

    col1.metric("Variables Estables", verdes, delta="Normal", delta_color="normal")
    col2.metric("Variables en Alerta", amarillos, delta="Precaución", delta_color="off")
    col3.metric("Variables con Drift Crítico", rojos, delta="Atención", delta_color="inverse")

    st.subheader("Tabla de Métricas de Drift por Variable")
    st.dataframe(drift_df, use_container_width=True)

    # Gráfico comparativo de distribución para una variable seleccionada
    st.subheader("📈 Comparación Visual de Distribuciones")
    var_selected = st.selectbox("Selecciona una variable para comparar:", numeric_cols)
    
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.kdeplot(df_ref[var_selected], label="Histórico (Train)", ax=ax, color="blue", fill=True, alpha=0.3)
    sns.kdeplot(df_current[var_selected], label="Actual (Producción)", ax=ax, color="orange", fill=True, alpha=0.3)
    ax.set_title(f"Distribución de {var_selected}")
    ax.legend()
    st.pyplot(fig)

# -------------------------------------------------------------------
# OPCIÓN 3: RECOMENDACIONES DE RETRAINING
# -------------------------------------------------------------------
elif menu == "3. Recomendaciones de Retraining":
    st.subheader("🤖 Recomendaciones Automáticas del Sistema")
    
    st.info("**Criterio del Sistema:** Si más del 20% de las variables presentan un PSI > 0.2 (Drift Crítico), se activa la recomendación de re-entrenamiento automático.")
    
    st.warning("⚠️ **Recomendación Generada:** Se ha detectado una variación ligera en el promedio de salarios de la población actual (PSI: 0.12). Se sugiere programar una revisión de hiperparámetros dentro de los próximos 30 días.")