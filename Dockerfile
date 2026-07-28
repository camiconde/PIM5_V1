# Usar imagen oficial de Python ligera
FROM python:3.11-slim

# Evitar la generación de archivos .pyc y forzar salida sin búfer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Crear un directorio de trabajo
WORKDIR /app

# Crear un usuario no privilegiado por seguridad
RUN adduser --disabled-password --gecos "" appuser

# Copiar el archivo de dependencias
COPY requirements.txt .

# Instalar dependencias desactivando scripts de compilación de código fuente no confiable
RUN pip install --no-cache-dir --only-binary=:all: --no-build-isolation -r requirements.txt || pip install --no-cache-dir -r requirements.txt

# Copiar directorios y archivos de la aplicación
COPY src/ ./src/
COPY models/ ./models/
COPY Base_de_datos.xlsx .

# Cambiar al usuario no privilegiado
USER appuser

# Exponer el puerto de la API
EXPOSE 8000

# Comando para ejecutar la API con Uvicorn
CMD ["uvicorn", "src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]