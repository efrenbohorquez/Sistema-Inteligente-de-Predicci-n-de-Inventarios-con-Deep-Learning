# Proyecto SmartForecast: Predicción de Inventarios con Deep Learning
## Entrega 2 - Modelo General LSTM

**Autor:** Equipo SmartForecast  
**Fecha:** 04 de Octubre de 2025  
**Curso:** Deep Learning Aplicado  

---

## 1. Resumen Ejecutivo

Este documento presenta los resultados de la segunda entrega del proyecto SmartForecast, enfocado en el desarrollo de un modelo general de predicción de demanda utilizando redes neuronales LSTM (Long Short-Term Memory). El objetivo principal es crear un sistema inteligente que permita optimizar la gestión de inventarios mediante la predicción precisa de las ventas futuras.

El modelo desarrollado utiliza técnicas avanzadas de Deep Learning para analizar patrones temporales en los datos históricos de ventas de múltiples productos, proporcionando predicciones que pueden ser utilizadas para mejorar las decisiones de compra y reducir costos operativos.

## 2. Introducción y Objetivos

### 2.1. Contexto del Problema

La gestión eficiente de inventarios representa uno de los desafíos más críticos en la cadena de suministro moderna. Las empresas enfrentan constantemente el dilema entre mantener suficiente stock para satisfacer la demanda y evitar el exceso de inventario que genera costos adicionales de almacenamiento y obsolescencia.

### 2.2. Objetivos Específicos

El desarrollo de esta entrega se centra en los siguientes objetivos:

- **Desarrollar un modelo LSTM general** capaz de predecir ventas para múltiples productos simultáneamente
- **Evaluar el rendimiento del modelo** utilizando métricas estándar de regresión
- **Generar visualizaciones** que permitan interpretar los resultados del modelo
- **Establecer una línea base** para comparaciones futuras con modelos específicos

## 3. Marco Teórico

### 3.1. Redes Neuronales LSTM

Las redes LSTM son una variante especializada de las redes neuronales recurrentes (RNN) diseñadas específicamente para manejar secuencias temporales largas. Su arquitectura única incluye mecanismos de compuertas que permiten:

- **Retener información relevante** a largo plazo
- **Olvidar información irrelevante** de manera selectiva  
- **Actualizar el estado interno** de forma controlada

### 3.2. Aplicación en Predicción de Demanda

En el contexto de predicción de inventarios, las LSTM ofrecen ventajas significativas:

- Capacidad para capturar **patrones estacionales** complejos
- Manejo efectivo de **dependencias temporales** a largo plazo
- Robustez ante **ruido** en los datos históricos

## 4. Metodología

### 4.1. Preprocesamiento de Datos

El proceso de preparación de datos incluyó las siguientes etapas críticas:

#### 4.1.1. Limpieza y Estructuración

Se realizó una transformación exhaustiva del dataset original, que presentaba una estructura compleja con múltiples tipos de datos en columnas no estandarizadas. Las principales operaciones fueron:

- **Estandarización de nombres de columnas** para facilitar la manipulación
- **Conversión de tipos de datos** apropiados (numéricos, fechas, categóricos)
- **Manejo de valores faltantes** mediante estrategias específicas por tipo de variable

#### 4.1.2. Transformación a Series Temporales

El dataset fue transformado de un formato ancho (wide) a un formato largo (long), creando una estructura de serie temporal adecuada para el modelado con LSTM:

| Característica | Valor |
|:---|:---|
| **Productos Únicos** | 27,297 |
| **Bodegas** | 105 |
| **Rango Temporal** | Sep 2024 - Ago 2025 |
| **Observaciones Totales** | 1,411,800 |

### 4.2. Arquitectura del Modelo

#### 4.2.1. Diseño de la Red Neuronal

El modelo LSTM general fue diseñado con la siguiente arquitectura optimizada:

```
Capa de Entrada: (6 timesteps, 1 feature)
    ↓
LSTM Layer 1: 50 unidades, return_sequences=True
    ↓
Dropout: 0.2 (prevención de overfitting)
    ↓
LSTM Layer 2: 50 unidades, return_sequences=False
    ↓
Dropout: 0.2
    ↓
Dense Layer: 25 neuronas, activación ReLU
    ↓
Output Layer: 1 neurona (predicción de ventas)
```

#### 4.2.2. Configuración de Entrenamiento

| Parámetro | Valor | Justificación |
|:---|:---|:---|
| **Optimizador** | Adam | Convergencia eficiente y estable |
| **Función de Pérdida** | MSE | Apropiada para regresión |
| **Épocas** | 5 | Balance entre tiempo y convergencia |
| **Batch Size** | 256 | Eficiencia computacional |
| **Validación** | 20% | Monitoreo de overfitting |

### 4.3. Entrenamiento y Validación

El modelo fue entrenado utilizando una división temporal de los datos:
- **80% para entrenamiento** (datos más antiguos)
- **20% para validación** (datos más recientes)

Se implementó **Early Stopping** con paciencia de 5 épocas para prevenir el sobreajuste y optimizar el tiempo de entrenamiento.

## 5. Resultados

### 5.1. Métricas de Evaluación

El modelo general LSTM demostró un rendimiento sólido en el conjunto de prueba:

| Métrica | Valor | Interpretación |
|:---|:---|:---|
| **MAE** | 2.09 | Error promedio de ±2.09 unidades |
| **MSE** | 133.30 | Varianza del error |
| **RMSE** | 11.55 | Desviación estándar del error |

### 5.2. Análisis de Convergencia

![Curva de Pérdida durante el Entrenamiento](modelo_general_output/training_loss.png)

La curva de pérdida muestra una convergencia estable sin signos evidentes de sobreajuste. La pérdida de validación sigue de cerca a la pérdida de entrenamiento, indicando una buena generalización del modelo.

### 5.3. Comparación de Predicciones

![Predicciones vs. Valores Reales](modelo_general_output/predictions_vs_actuals.png)

La visualización de predicciones versus valores reales demuestra que el modelo captura efectivamente los patrones principales en los datos, aunque presenta algunas dificultades con picos extremos de demanda.

## 6. Análisis de Resultados

### 6.1. Fortalezas del Modelo

El modelo general LSTM presenta las siguientes características positivas:

**Capacidad de Generalización:** El modelo demuestra habilidad para aprender patrones comunes entre diferentes productos, lo que es especialmente valioso para productos con historiales de ventas limitados.

**Eficiencia Computacional:** Un solo modelo puede generar predicciones para múltiples productos, reduciendo significativamente los recursos computacionales requeridos comparado con modelos individuales.

**Robustez:** El modelo mantiene un rendimiento consistente a través de diferentes categorías de productos y patrones de demanda.

### 6.2. Limitaciones Identificadas

**Pérdida de Especificidad:** Al entrenar con datos de múltiples productos, el modelo puede no capturar patrones únicos específicos de productos individuales.

**Sensibilidad a Outliers:** Los picos extremos de demanda pueden no ser predichos con precisión, lo que podría impactar la gestión de inventarios en situaciones críticas.

**Dependencia de Datos Históricos:** El modelo requiere un historial mínimo de ventas para generar predicciones confiables.

## 7. Conclusiones

### 7.1. Logros Principales

La implementación del modelo general LSTM ha demostrado ser exitosa en varios aspectos fundamentales:

**Viabilidad Técnica:** Se ha establecido que las redes LSTM son efectivas para la predicción de demanda en el contexto de gestión de inventarios, proporcionando predicciones con un nivel de error aceptable para la toma de decisiones operativas.

**Escalabilidad:** El enfoque de modelo general permite manejar eficientemente grandes volúmenes de productos sin requerir entrenamientos individuales, lo que es crucial para implementaciones empresariales.

**Fundación para Mejoras:** Los resultados obtenidos establecen una línea base sólida que servirá como punto de comparación para modelos más especializados en entregas futuras.

### 7.2. Implicaciones Prácticas

Los resultados sugieren que el modelo puede ser implementado como una herramienta de apoyo en la toma de decisiones de inventario, especialmente para:

- **Planificación de compras** a mediano plazo
- **Identificación de tendencias** generales de demanda
- **Optimización de niveles de stock** para productos con patrones regulares

### 7.3. Próximos Pasos

La siguiente fase del proyecto se enfocará en:

**Desarrollo de Modelos Específicos:** Crear modelos LSTM especializados para productos individuales o categorías específicas, con el objetivo de mejorar la precisión de las predicciones.

**Análisis Comparativo:** Realizar una evaluación exhaustiva entre el modelo general y los modelos específicos para determinar las mejores estrategias de implementación.

**Optimización de Hiperparámetros:** Explorar configuraciones alternativas de la arquitectura LSTM para maximizar el rendimiento predictivo.

## 8. Referencias

[1] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.

[2] Chollet, F. (2021). Deep Learning with Python. Manning Publications.

[3] Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly Media.

---

**Nota:** Este documento forma parte del proyecto SmartForecast desarrollado como parte del curso de Deep Learning Aplicado. Los códigos fuente, datos y visualizaciones adicionales están disponibles en los archivos adjuntos del proyecto.
