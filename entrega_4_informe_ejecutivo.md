'''
# SmartForecast: Informe Ejecutivo y Presentación de Resultados

**Autor:** Equipo SmartForecast  
**Fecha:** 04 de Octubre de 2025  
**Curso:** Deep Learning Aplicado

---

## 1. Resumen Ejecutivo

Este informe presenta los resultados finales del proyecto **SmartForecast**, cuyo objetivo fue desarrollar un sistema de predicción de demanda de inventarios utilizando técnicas de Deep Learning. Se implementaron y evaluaron dos enfoques principales: un modelo LSTM general, entrenado con datos de todos los productos, y un modelo LSTM específico, entrenado para un producto individual. 

Los resultados demuestran que, si bien el modelo general ofrece una solución escalable y eficiente, el modelo específico para el producto de prueba (A3487FE4D9) logró una precisión significativamente mayor, con una **reducción del 31.4% en el RMSE**. Esto valida la hipótesis de que los modelos especializados pueden capturar patrones de demanda únicos y mejorar la precisión del pronóstico.

Finalmente, se desarrolló una aplicación interactiva con Gradio para visualizar los resultados y facilitar la toma de decisiones. Este informe resume los hallazgos clave, las implicaciones para el negocio y las recomendaciones para la implementación de una estrategia de pronóstico de inventario optimizada.

## 2. Introducción

La gestión de inventarios es un desafío crítico para cualquier empresa. Un pronóstico de demanda impreciso puede llevar a sobreinventario, con los costos asociados, o a desabastecimiento, resultando en pérdida de ventas y clientes insatisfechos. Este proyecto explora el uso de redes neuronales de memoria a corto y largo plazo (LSTM) para mejorar la precisión de los pronósticos de ventas.

## 3. Metodología

El proyecto se desarrolló en las siguientes fases:

1.  **Análisis y Preparación de Datos:** Se procesó un conjunto de datos de ventas de 12 meses, que abarca 27,297 productos en 105 bodegas. Las tareas incluyeron limpieza de datos, manejo de valores faltantes y transformación de los datos a un formato de serie temporal.

2.  **Desarrollo del Modelo General:** Se entrenó un modelo LSTM con los datos de todos los productos para capturar patrones de demanda generales.

3.  **Desarrollo del Modelo Específico:** Se entrenó un segundo modelo LSTM, esta vez utilizando únicamente los datos de un producto específico (A3487FE4D9) para capturar sus patrones de demanda únicos.

4.  **Comparación y Evaluación:** Se comparó el rendimiento de ambos modelos utilizando métricas estándar como MAE, MSE y RMSE.

5.  **Desarrollo de la Aplicación Interactiva:** Se creó una interfaz de usuario con Gradio para visualizar los datos, los resultados de los modelos y las predicciones.

## 4. Resultados y Discusión

### 4.1. Comparación de Modelos

La siguiente tabla resume el rendimiento de ambos modelos en la predicción de ventas para el producto A3487FE4D9:

| Métrica | Modelo General | Modelo Específico | Mejora |
| :--- | :--- | :--- | :--- |
| **MAE** | 89.47 | 62.75 | **29.9%** |
| **MSE** | 8,426.69 | 3,970.05 | **52.9%** |
| **RMSE** | 91.80 | 63.01 | **31.4%** |

Como se puede observar, el modelo específico supera significativamente al modelo general en todas las métricas de evaluación. Esto indica que, para productos con suficiente historial de datos, un modelo entrenado específicamente para ese producto puede proporcionar pronósticos más precisos.

### 4.2. Visualización de Resultados

El siguiente gráfico muestra una comparación visual de las predicciones de ambos modelos frente a los valores reales para el producto A3487FE4D9:

![Comparación de Modelos](modelo_especifico_output/model_comparison.png)

El gráfico ilustra claramente cómo el modelo específico (línea discontinua) se ajusta mejor a las fluctuaciones de la demanda real en comparación con el modelo general (línea punteada), que tiende a suavizar las predicciones.

## 5. Aplicación Interactiva con Gradio

Para facilitar la exploración de los resultados y la interacción con los modelos, se desarrolló una aplicación web utilizando Gradio. Esta aplicación permite a los usuarios:

*   Visualizar los datos de ventas de diferentes productos.
*   Comparar las predicciones del modelo general y específico.
*   Analizar las métricas de rendimiento de cada modelo.

La aplicación está disponible en la siguiente URL: [https://7fb2e84e505724032f.gradio.live](https://7fb2e84e505724032f.gradio.live)

## 6. Conclusiones y Recomendaciones

El proyecto SmartForecast ha demostrado con éxito el potencial de los modelos LSTM para la predicción de la demanda de inventario. Los resultados indican que, si bien un modelo general puede ser útil para productos con datos limitados o para obtener una visión general, los modelos específicos por producto ofrecen una precisión significativamente mayor.

Se recomienda una **estrategia híbrida** para la gestión de inventarios:

*   **Para productos de alta rotación y valor (Clase A):** Implementar modelos específicos para maximizar la precisión del pronóstico y minimizar los costos de inventario.
*   **Para productos de menor rotación o con datos insuficientes (Clase B y C):** Utilizar el modelo general como una solución eficiente y escalable.

Este enfoque permitirá a la empresa optimizar sus niveles de inventario, reducir costos y mejorar la satisfacción del cliente.

## 7. Referencias

[1] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, 9(8), 1735-1780.

[2] Chollet, F. (2021). *Deep Learning with Python*. Manning Publications.

[3] Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media.

[4] Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. *International Journal of Forecasting*, 36(1), 54-74.

[5] Lim, B., & Zohren, S. (2021). Time-series forecasting with deep learning: a survey. *Philosophical Transactions of the Royal Society A*, 379(2194), 20200209.

[6] Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. *International Journal of Forecasting*, 36(3), 1181-1191.
'''
