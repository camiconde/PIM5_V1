# PIM5_V1
# 🏦 Modelo de Evaluación de Riesgo Crediticio & Monitoreo de Data Drift

## 📌 Descripción del Proyecto
Este proyecto implementa un flujo completo de Ciencia de Datos para predecir el comportamiento de pago de clientes crediticios, incorporando buenas prácticas de producción:
- **Ingeniería de Características:** Creación de Ratios Financieros (*Ratio de Endeudamiento* y *Capacidad de Pago Disponible*).
- **Prevención de Data Leakage:** Exclusión de saldos de mora concurrentes.
- **Modelamiento:** Evaluación comparativa entre Regresión Logística, Random Forest y Gradient Boosting, resultando seleccionada la **Regresión Logística** (F1-Score: 0.9757).
- **Monitoreo & Observabilidad:** Cálculo periódico de métricas de Data Drift (KS-Test, PSI, Jensen-Shannon, Chi-Cuadrado).
- **Interfaz Interactiva:** Despliegue con Streamlit para inferencias en tiempo real y panel de monitoreo.

---
## 📌 Resumen de Avances del Proyecto

### 🔹 Avance 1: Análisis Exploratorio de Datos (EDA) y Comprensión del Negocio
- **Exploración de la Cartera:** Se analizaron 10,763 registros históricos de créditos. Se identificó un desbalance de clases típico en riesgo financiero, con un **95.2%** de pagos a tiempo.
- **Relaciones Financieras:** Se identificó que la capacidad de pago y el puntaje en burós crediticios (`puntaje_datacredito`) son los mejores predictores de cumplimiento.
- **Detección de Data Leakage:** Se detectaron variables concurrentes al crédito (`saldo_mora`, `saldo_total`, `saldo_principal`) que reflejan el estado posterior del préstamo y debieron aislarse del modelo predictivo.

---

### 🔹 Avance 2: Ingeniería de Características, Modelado y Evaluación
- **Ingeniería de Características:** Se crearon ratios financieros clave como el **Ratio de Endeudamiento** (`cuota_pactada / salario_cliente`) y la **Capacidad de Pago Disponible**.
- **Pipelines y Preprocesamiento:** Se construyeron `Pipelines` con `ColumnTransformer` para automatizar el escalado de variables numéricas (`StandardScaler`) y la codificación de categóricas (`OneHotEncoder`).
- **Entrenamiento y Selección:** Se evaluaron tres algoritmos (*Logistic Regression*, *Random Forest* y *Gradient Boosting*). La **Regresión Logística** fue seleccionada como el mejor modelo por su alto rendimiento (**F1-Score: 0.9757**) e interpretabilidad financiera.
- **Serialización:** El pipeline y modelo final se exportaron en `models/model.pkl`.

---

### 🔹 Avance 3: Monitoreo de Data Drift y Aplicación Interactiva
- **Monitoreo de Población (Data Drift):** Se implementó el módulo `src/model_monitoring.py` para calcular métricas estadísticas de variación entre la población histórica y la actual (**KS-Test**, **PSI**, **Jensen-Shannon** y **Chi-Cuadrado**).
- **Dashboard en Streamlit:** Se desarrolló la aplicación interactiva (`app.py`) con tres secciones:
  1. *Evaluador de solicitudes en tiempo real.*
  2. *Panel visual de monitoreo de drift con semáforos e indicadores.*
  3. *Recomendaciones automáticas de re-entrenamiento (retraining).*
  
---

### 🔹 Avance 4: Despliegue con API (FastAPI) y Contenedorización (Docker)
- **Despliegue de API Rest:** Se implementó `src/model_deploy.py` utilizando **FastAPI** y **Uvicorn**, exponiendo el endpoint `/predict` optimizado para recibir lotes de datos (*batch processing*) en formato JSON.
- **Empaquetado en Docker:** Se creó el `Dockerfile` y `.dockerignore` para empaquetar el entorno, código y dependencias en un contenedor liviano y reproducible.


## 🛠️ Estructura del Repositorio
PIM5_V1/
│── Base_de_datos.xlsx          # Dataset original
│── app.py                      # Aplicación interactiva Streamlit
│── requirements.txt            # Dependencias del proyecto
│── models/
│   └── model.pkl               # Pipeline y Modelo seleccionado guardado
└── src/
│── cargar_datos.py         # Carga y limpieza inicial
│── ft_engineering.py       # Pipeline de preprocesamiento y transformaciones
│── model_training_evaluation.py # Entrenamiento y tabla comparativa
└── model_monitoring.py     # Lógica estadística para Data Drift

---

## 🚀 Cómo ejecutar la Aplicación
1. Activar el entorno virtual:
   .\venv\Scripts\activate

2. Instalar dependencias:
    pip install -r requirements.txt

3. Ejecutar la aplicación Streamlit:
    streamlit run app.py
