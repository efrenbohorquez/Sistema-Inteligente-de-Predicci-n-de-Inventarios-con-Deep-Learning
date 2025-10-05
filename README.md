# 🚀 SmartForecast: Sistema Inteligente de Predicción de Inventarios con Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0+-orange.svg)](https://tensorflow.org/)
[![Gradio](https://img.shields.io/badge/Gradio-5.49.0+-brightgreen.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Descripción

SmartForecast es un sistema avanzado de predicción de inventarios que utiliza redes neuronales LSTM (Long Short-Term Memory) para generar predicciones precisas de ventas y demanda. El proyecto implementa dos enfoques complementarios:

- **Modelo General**: Entrenado con datos de múltiples productos para predicciones globales
- **Modelo Específico**: Entrenado exclusivamente con datos de un producto individual para máxima precisión

## 🎯 Características Principales

### 🔧 Funcionalidades Core
- **Predicción LSTM Avanzada**: Utiliza redes neuronales recurrentes para capturar patrones temporales complejos
- **Análisis Comparativo**: Evaluación automática entre modelos general y específico
- **Interfaz Web Interactiva**: Dashboard profesional con Gradio para visualización de resultados
- **Métricas Completas**: MAE, MSE, RMSE y visualizaciones detalladas

### 📊 Capacidades Analíticas
- Procesamiento de series temporales multivariadas
- Normalización automática de datos con MinMaxScaler
- Generación de secuencias temporales optimizadas
- Evaluación cruzada de modelos con métricas estándar

## 🏗️ Arquitectura del Sistema

```
SmartForecast/
├── 📊 Procesamiento de Datos
│   ├── series_temporales.csv      # Datos originales de ventas
│   ├── datos_procesados.csv       # Datos preparados para ML
│   └── resumen_preprocesamiento.json
│
├── 🧠 Modelos de Deep Learning
│   ├── modelo_general.py          # LSTM para todos los productos
│   ├── modelo_especifico.py       # LSTM para producto individual
│   └── modelo_general_output/     # Resultados del modelo general
│       ├── modelo_general.h5
│       ├── evaluation_results.json
│       ├── predictions_vs_actuals.png
│       └── training_loss.png
│
├── 📈 Análisis Específico
│   └── modelo_especifico_output/  # Comparación de modelos
│       ├── comparison_results.json
│       └── model_comparison.png
│
├── 🌐 Interfaz Web
│   └── app_gradio.py             # Dashboard interactivo
│
└── 📚 Documentación
    ├── README.md
    ├── Entrega_2_Informe.md
    ├── Entrega_3_Informe.md
    └── entrega_4_informe_ejecutivo.md
```

## 🚀 Instalación Rápida

### Prerrequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git (opcional, para clonar el repositorio)

### Clonar el Repositorio
```bash
git clone https://github.com/efrenbohorquez/Sistema-Inteligente-de-Predicci-n-de-Inventarios-con-Deep-Learning.git
cd Sistema-Inteligente-de-Predicci-n-de-Inventarios-con-Deep-Learning
```

### Instalar Dependencias
```bash
pip install -r requirements.txt
```

### Verificar Instalación
```bash
python -c "import tensorflow, gradio, pandas, numpy; print('✅ Todas las dependencias instaladas correctamente')"
```

## 🎮 Guía de Uso

### 1. Entrenar Modelo General
```bash
python modelo_general.py
```
**Salida esperada:**
- Modelo entrenado guardado en `modelo_general_output/modelo_general.h5`
- Métricas de evaluación en formato JSON
- Gráficos de entrenamiento y predicciones

### 2. Entrenar Modelo Específico y Comparar
```bash
python modelo_especifico.py
```
**Salida esperada:**
- Comparación automática con el modelo general
- Análisis de mejora en precisión
- Visualizaciones comparativas

### 3. Lanzar Dashboard Interactivo
```bash
python app_gradio.py
```
**Acceder a:** http://127.0.0.1:7861

## 📊 Resultados Demostrados

### Comparación de Rendimiento
| Modelo | RMSE | Mejora |
|--------|------|--------|
| **Modelo General** | 497.37 | Baseline |
| **Modelo Específico** | 60.92 | **8x mejor** |

### Métricas del Modelo General
- **Parámetros**: 31,901
- **MAE**: 0.4316
- **RMSE**: 26.86
- **Tiempo de entrenamiento**: ~5 épocas

## 🔧 Configuración Avanzada

### Personalizar Hiperparámetros
```python
# En modelo_general.py
LOOK_BACK = 6        # Ventana temporal
EPOCHS = 50          # Épocas de entrenamiento
BATCH_SIZE = 256     # Tamaño del lote
```

### Cambiar Producto para Análisis Específico
```python
# En modelo_especifico.py
PRODUCT_ID = "tu_codigo_producto"  # Código del producto a analizar
```

## 🌐 Dashboard Interactivo

El dashboard de Gradio incluye:

### 📊 Pestaña: Resumen General
- Estadísticas del dataset
- Información de preprocesamiento
- Métricas clave de rendimiento

### 🔍 Pestaña: Exploración de Datos
- Visualización de series temporales
- Selector interactivo de productos
- Análisis de tendencias

### 📈 Pestaña: Resultados del Modelo General
- Gráficos de pérdida de entrenamiento
- Predicciones vs valores reales
- Métricas detalladas (MAE, MSE, RMSE)

### 🎯 Pestaña: Comparación de Modelos
- Análisis lado a lado
- Gráficos comparativos de rendimiento
- Recomendaciones automáticas

## 📁 Estructura de Datos

### Formato de Entrada
```csv
bodega,codigo_bodega,producto,codigo_producto,fecha,ventas,año,mes,trimestre
BDG-U4TNO,BOD-3B47,P14417,14709969,2024-09-01,125.5,2024,9,3
```

### Variables Principales
- **fecha**: Timestamp de la venta
- **ventas**: Cantidad vendida (variable objetivo)
- **codigo_producto**: Identificador único del producto
- **bodega**: Código del almacén

## 🔬 Metodología Técnica

### Preprocesamiento
1. **Limpieza de datos**: Eliminación de valores nulos y atípicos
2. **Normalización**: Escalado MinMax [0,1] para optimizar LSTM
3. **Secuenciación**: Creación de ventanas temporales (look_back=6)
4. **División**: 80% entrenamiento, 20% validación

### Arquitectura LSTM
```python
modelo = Sequential([
    LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
```

### Optimización
- **Optimizer**: Adam
- **Loss**: Mean Squared Error
- **Regularización**: Dropout (0.2)
- **Early Stopping**: Monitoreo de pérdida de validación

## 📈 Casos de Uso

### 1. Retail y Comercio Electrónico
- Predicción de demanda estacional
- Optimización de stock de seguridad
- Planificación de compras

### 2. Manufactura
- Gestión de materias primas
- Planificación de producción
- Control de inventario WIP

### 3. Distribución y Logística
- Optimización de almacenes
- Planificación de rutas
- Gestión de inventario multicanal

## 🛠️ Personalización

### Agregar Nuevos Productos
1. Actualizar `datos_procesados.csv` con nuevos datos
2. Modificar `PRODUCT_ID` en `modelo_especifico.py`
3. Re-entrenar los modelos

### Extender Funcionalidades
- Implementar nuevas arquitecturas (GRU, Transformer)
- Agregar variables exógenas (precio, promociones)
- Integrar con APIs de inventario

## 📚 Documentación Adicional

- [📖 Informe Técnico Completo](Entrega_3_Informe.md)
- [📋 Informe Ejecutivo](entrega_4_informe_ejecutivo.md)
- [📊 Análisis de Rendimiento](Entrega_2_Informe.md)

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

### Áreas de Contribución
- 🧠 Nuevos modelos de ML/DL
- 📊 Visualizaciones mejoradas
- 🔧 Optimizaciones de rendimiento
- 📚 Documentación y ejemplos
- 🧪 Tests unitarios

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 👨‍💻 Autor

**Efrén Bohórquez**
- GitHub: [@efrenbohorquez](https://github.com/efrenbohorquez)
- LinkedIn: [Efrén Bohórquez](https://linkedin.com/in/efrenbohorquez)
- Email: contacto@efrenbohorquez.com

## 🙏 Agradecimientos

- TensorFlow team por el excelente framework de Deep Learning
- Gradio team por la increíble librería de interfaces web
- Comunidad de ciencia de datos por las mejores prácticas
- Colaboradores y beta testers del proyecto

## 📊 Estadísticas del Proyecto

- **Líneas de código**: ~1,200
- **Modelos implementados**: 2 (General + Específico)
- **Formatos soportados**: CSV, JSON
- **Métricas disponibles**: 6 (MAE, MSE, RMSE, etc.)
- **Visualizaciones**: 8+ gráficos interactivos

---

<div align="center">

**⭐ Si este proyecto te resultó útil, ¡no olvides darle una estrella! ⭐**

[🚀 Demo en Vivo](http://127.0.0.1:7861) | [📖 Documentación](docs/) | [🐛 Reportar Bug](https://github.com/efrenbohorquez/Sistema-Inteligente-de-Predicci-n-de-Inventarios-con-Deep-Learning/issues)

</div>