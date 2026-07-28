# 💳 Sistema de Evaluación e Inferencia de Riesgo Crediticio (PIM5)

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=camiconde_PIM5_V1&metric=alert_status)](https://sonarcloud.io/summary/overall?id=camiconde_PIM5_V1)
[![Quality Gate Security](https://sonarcloud.io/api/project_badges/measure?project=camiconde_PIM5_V1&metric=security_rating)](https://sonarcloud.io/summary/overall?id=camiconde_PIM5_V1)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Container](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/)

---

## 📌 Contexto de Negocio
Este proyecto implementa una solución integral de **Machine Learning para la Evaluación del Riesgo Crediticio** en una entidad financiera. El objetivo principal es predecir la probabilidad de cumplimiento de pago (`Pago_atiempo`) en solicitudes de crédito para optimizar las decisiones de aprobación, reduciendo la tasa de morosidad y maximizando la colocación segura de capital.

---

## 📊 Principales Hallazgos del EDA (Avance 1)
* **Desbalance de Clases:** Se identificó que aproximadamente el **95.2%** de los registros corresponden a pagos a tiempo, mientras que un **4.8%** corresponde a incumplimientos. Se priorizó el uso de métricas como **ROC-AUC**, **F1-Score** y **Precision-Recall** sobre el *Accuracy*.
* **Capacidad de Pago:** Existe una clara correlación entre una cuota pactada superior al **35%-40% del salario mensual** del cliente y el incremento en el riesgo de mora.
* **Historial Crediticio:** El puntaje de buró (`puntaje_datacredito`) presenta una mediana significativamente mayor (>700 puntos) en el grupo de cumplimiento respecto al de mora (<550 puntos).
* **Prevención de Data Leakage:** Se identificaron y excluyeron del set de entrenamiento variables posteriores a la originación del crédito (`saldo_mora`, `saldo_total`, días de atraso acumulados) para asegurar un modelo predictivo realista en producción.

---

## 📁 Arquitectura del Proyecto
La estructura del proyecto cumple estrictamente con los estándares MLOps e integración continua requeridos:

```text
PIM5_V1/
├── .github/
│   └── workflows/
│       └── sonarcloud.yml       # Integración CI con SonarCloud
├── models/
│   └── model.pkl                # Modelo entrenado empacado
├── src/
│   ├── cargar_datos.py          # Script de carga dinámica de datos
│   ├── comprension_eda.ipynb    # Notebook de exploración de datos (EDA)
│   ├── ft_engineering.py        # Pipelines de transformación y preprocess
│   ├── model_training_evaluation.py # Entrenamiento y evaluación del modelo
│   ├── model_deploy.py          # API REST desarrollada con FastAPI
│   └── model_monitoring.py      # Monitoreo de Data Drift
├── .dockerignore                # Exclusiones para la imagen Docker
├── .gitignore                   # Exclusiones para el repositorio de Git
├── app.py                       # Aplicación interactiva en Streamlit
├── Base_de_datos.xlsx           # Dataset histórico de créditos
├── Dockerfile                   # Configuración del contenedor Docker
├── README.md                    # Documentación principal del proyecto
├── requirements.txt             # Dependencias del entorno
└── sonar-project.properties     # Configuración para análisis en SonarCloud

🚀 Guía de Ejecución Local
1. Requisitos Previos
Python 3.11+ instalado.

Git y Docker Desktop (opcional para contenedorización).

2. Configuración del Entorno
Bash
# Clonar el repositorio
git clone [https://github.com/camiconde/PIM5_V1.git](https://github.com/camiconde/PIM5_V1.git)
cd PIM5_V1

# Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate  # En Linux/Mac
# venv\Scripts\activate   # En Windows Power Shell

# Instalar dependencias
pip install -r requirements.txt
3. Ejecución de la API REST (FastAPI)
Bash
uvicorn src.model_deploy:app --host 0.0.0.0 --port 8000 --reload
Documentación Interactiva (Swagger UI): http://localhost:8000/docs

Endpoint de Predicción en Lote: POST /predict

4. Ejecución del Dashboard (Streamlit)
Bash
streamlit run app.py
Accede a la interfaz web interactiva desde tu navegador en http://localhost:8501 para simular evaluaciones crediticias y visualizar el monitoreo del modelo.

🐳 Ejecución con Docker
Construir la imagen:
Bash
docker build -t riesgo-crediticio-api .
Ejecutar el contenedor:
Bash
docker run -d -p 8000:8000 --name api_riesgo riesgo-crediticio-api

🔍 Integración de MLOps y Calidad de Código (SonarCloud)
El repositorio incluye integración continua (CI) mediante GitHub Actions y SonarCloud para auditar:

Calidad y Mantenibilidad del Código: Evaluación constante de buenas prácticas en Python.

Seguridad: Inspección de vulnerabilidades y buenas prácticas en contenedores Docker (ej. ejecución como usuario sin privilegios appuser).

Duplicaciones y Estilo: Garantía de código limpio y modular.