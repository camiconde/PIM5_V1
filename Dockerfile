# 1. Imagen base oficial de Python (liviana)
FROM python:3.11-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar el archivo de requerimientos e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar todo el código de tu proyecto al contenedor
COPY . .

# 5. Exponer los puertos (8000 para la API de FastAPI, 8501 para Streamlit)
EXPOSE 8000 8501

# 6. Comando por defecto para iniciar la API REST con Uvicorn
CMD ["uvicorn", "src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]