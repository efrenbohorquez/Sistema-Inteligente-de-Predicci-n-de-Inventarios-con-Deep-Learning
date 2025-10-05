# 🚀 Guía de Despliegue - SmartForecast

Esta guía te ayudará a desplegar SmartForecast en diferentes entornos.

## 📋 Requisitos Previos

### Sistema Local
- Python 3.8+
- Git
- 4GB RAM mínimo (8GB recomendado)
- 2GB espacio libre en disco

### Servicios Cloud (Opcional)
- Cuenta en GitHub (para código)
- Hugging Face Spaces (para demo online)
- Google Colab (para entrenamiento)
- Docker (para containerización)

## 🏠 Despliegue Local

### 1. Instalación Rápida
```bash
# Clonar repositorio
git clone https://github.com/efrenbohorquez/Sistema-Inteligente-de-Predicci-n-de-Inventarios-con-Deep-Learning.git
cd Sistema-Inteligente-de-Predicci-n-de-Inventarios-con-Deep-Learning

# Crear entorno virtual
python -m venv smartforecast_env
source smartforecast_env/bin/activate  # Linux/Mac
# smartforecast_env\Scripts\activate    # Windows

# Instalar dependencias
pip install -r requirements.txt

# Verificar instalación
python -c "import tensorflow, gradio; print('✅ Instalación exitosa')"
```

### 2. Ejecución Paso a Paso
```bash
# 1. Entrenar modelo general
python modelo_general.py

# 2. Entrenar modelo específico
python modelo_especifico.py

# 3. Lanzar dashboard
python app_gradio.py
```

### 3. Acceder a la Aplicación
- **URL Local**: http://127.0.0.1:7861
- **Dashboard**: Interfaz web completa
- **API**: Endpoints disponibles para integración

## ☁️ Despliegue en la Nube

### Hugging Face Spaces

1. **Preparar archivos**:
```bash
# Crear app.py para Hugging Face
cp app_gradio.py app.py

# Editar configuración de puerto
# En app.py, cambiar:
# interface.launch(server_port=7861, share=False)
# Por:
# interface.launch()
```

2. **Subir a Hugging Face**:
```bash
git remote add hf https://huggingface.co/spaces/tu-usuario/smartforecast
git push hf main
```

### Google Colab

1. **Notebook de despliegue**:
```python
# Instalar dependencias
!pip install tensorflow gradio plotly scikit-learn

# Clonar repositorio
!git clone https://github.com/efrenbohorquez/Sistema-Inteligente-de-Predicci-n-de-Inventarios-con-Deep-Learning.git
%cd Sistema-Inteligente-de-Predicci-n-de-Inventarios-con-Deep-Learning

# Ejecutar modelos
!python modelo_general.py
!python modelo_especifico.py

# Lanzar app con túnel público
!python app_gradio.py --share
```

### Docker

1. **Dockerfile**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7861

CMD ["python", "app_gradio.py"]
```

2. **Docker Compose**:
```yaml
version: '3.8'
services:
  smartforecast:
    build: .
    ports:
      - "7861:7861"
    volumes:
      - ./data:/app/data
    environment:
      - PYTHONPATH=/app
```

3. **Comandos Docker**:
```bash
# Construir imagen
docker build -t smartforecast .

# Ejecutar contenedor
docker run -p 7861:7861 smartforecast

# Usar Docker Compose
docker-compose up -d
```

## 🌐 Despliegue en Producción

### AWS EC2

1. **Configurar instancia**:
```bash
# Instalar dependencias del sistema
sudo apt update
sudo apt install python3-pip git nginx -y

# Clonar y configurar aplicación
git clone https://github.com/efrenbohorquez/Sistema-Inteligente-de-Predicci-n-de-Inventarios-con-Deep-Learning.git
cd Sistema-Inteligente-de-Predicci-n-de-Inventarios-con-Deep-Learning
pip3 install -r requirements.txt
```

2. **Configurar Nginx**:
```nginx
server {
    listen 80;
    server_name tu-dominio.com;

    location / {
        proxy_pass http://127.0.0.1:7861;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

3. **Servicio systemd**:
```ini
[Unit]
Description=SmartForecast App
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Sistema-Inteligente-de-Predicci-n-de-Inventarios-con-Deep-Learning
ExecStart=/usr/bin/python3 app_gradio.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### Heroku

1. **Procfile**:
```
web: python app_gradio.py --server-port=$PORT --server-name=0.0.0.0
```

2. **runtime.txt**:
```
python-3.9.18
```

3. **Despliegue**:
```bash
heroku create tu-app-smartforecast
git push heroku main
```

## 🔧 Configuración de Producción

### Variables de Entorno
```bash
# .env file
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=tu-secret-key-aqui
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
```

### Optimizaciones
```python
# En app_gradio.py para producción
interface.launch(
    server_name="0.0.0.0",
    server_port=7861,
    share=False,
    show_error=False,
    enable_queue=True,
    max_threads=10
)
```

## 📊 Monitoreo y Logs

### Configurar Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smartforecast.log'),
        logging.StreamHandler()
    ]
)
```

### Métricas de Rendimiento
```bash
# Instalar herramientas de monitoreo
pip install psutil prometheus_client

# Monitorear recursos
htop
nvidia-smi  # Para GPU
```

## 🚨 Solución de Problemas

### Errores Comunes

1. **Error de memoria**:
```bash
# Reducir batch size en modelos
BATCH_SIZE = 64  # En lugar de 256
```

2. **Puerto ocupado**:
```bash
# Cambiar puerto en app_gradio.py
server_port=7862  # En lugar de 7861
```

3. **Dependencias faltantes**:
```bash
# Reinstalar dependencias
pip install --force-reinstall -r requirements.txt
```

### Logs de Debug
```bash
# Ejecutar con logs verbose
python -u app_gradio.py --verbose

# Verificar logs del sistema
tail -f /var/log/syslog
```

## 📈 Escalabilidad

### Load Balancer
```nginx
upstream smartforecast_app {
    server 127.0.0.1:7861;
    server 127.0.0.1:7862;
    server 127.0.0.1:7863;
}

server {
    location / {
        proxy_pass http://smartforecast_app;
    }
}
```

### Base de Datos
```python
# Configurar PostgreSQL para datos de producción
DATABASE_URL = "postgresql://user:password@host:port/database"
```

## 🔒 Seguridad

### HTTPS
```bash
# Certificado SSL con Let's Encrypt
sudo certbot --nginx -d tu-dominio.com
```

### Autenticación
```python
# Agregar autenticación a Gradio
interface.launch(
    auth=("admin", "password123"),
    auth_message="Acceso restringido a SmartForecast"
)
```

## 📞 Soporte

- 📧 **Email**: contacto@efrenbohorquez.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/efrenbohorquez/Sistema-Inteligente-de-Predicci-n-de-Inventarios-con-Deep-Learning/issues)
- 📚 **Docs**: [Documentación completa](README.md)

---

🚀 **¡Tu SmartForecast está listo para producción!**