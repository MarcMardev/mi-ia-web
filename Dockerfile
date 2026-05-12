# 1. Usar una imagen ligera de Python
FROM python:3.11-slim
# 2. Instalar dependencias del sistema actualizadas para OpenCV
# Reemplazamos libgl1-mesa-glx y agregamos dependencias
RUN apt-get update &amp;&amp; apt-get install -y \
libgl1 \
libglib2.0-0 \
libsm6 \
libxext6 \
libxrender1 \
&amp;&amp; rm -rf /var/lib/apt/lists/*
# 3. Crear carpeta de trabajo
WORKDIR /app
# 4. Copiar el archivo de requerimientos e instalar librerías
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# 5. Copiar el resto del código y el modelo entrenado
COPY . .
# 6. Exponer el puerto de Streamlit
EXPOSE 8501
# 7. Comando para arrancar la app
CMD [&quot;streamlit&quot;, &quot;run&quot;, &quot;app.py&quot;, &quot;--server.port=8501&quot;, \
&quot;--server.address=0.0.0.0&quot;]
