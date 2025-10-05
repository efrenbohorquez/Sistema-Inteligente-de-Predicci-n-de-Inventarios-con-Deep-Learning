#!/usr/bin/env python3
"""
Aplicación Gradio para SmartForecast - Presentación Interactiva de Resultados

Esta aplicación web permite visualizar y comparar los resultados de los modelos
LSTM general y específico para la predicción de inventarios.

Autor: Equipo SmartForecast
Fecha: Octubre 2025
"""

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class SmartForecastApp:
    """
    Clase principal para la aplicación Gradio de SmartForecast.
    """
    
    def __init__(self):
        """
        Inicializa la aplicación cargando los datos y resultados.
        """
        self.load_data()
        self.load_results()
    
    def load_data(self):
        """
        Carga los datos procesados y de series de tiempo.
        """
        try:
            self.ts_data = pd.read_csv('series_temporales.csv', parse_dates=['fecha'])
            self.processed_data = pd.read_csv('datos_procesados.csv')
            
            # Cargar resumen del preprocesamiento
            with open('resumen_preprocesamiento.json', 'r', encoding='utf-8') as f:
                self.preprocessing_summary = json.load(f)
                
        except FileNotFoundError as e:
            print(f"Error cargando datos: {e}")
            self.ts_data = pd.DataFrame()
            self.processed_data = pd.DataFrame()
            self.preprocessing_summary = {}
    
    def load_results(self):
        """
        Carga los resultados de evaluación de los modelos.
        """
        try:
            # Resultados del modelo general
            with open('modelo_general_output/evaluation_results.json', 'r') as f:
                self.general_results = json.load(f)
            
            # Resultados de la comparación
            with open('modelo_especifico_output/comparison_results.json', 'r') as f:
                self.comparison_results = json.load(f)
                
        except FileNotFoundError as e:
            print(f"Error cargando resultados: {e}")
            self.general_results = {}
            self.comparison_results = {}
    
    def create_overview_tab(self):
        """
        Crea la pestaña de resumen del proyecto.
        """
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("""
                # 📊 SmartForecast: Predicción de Inventarios con Deep Learning
                
                ## Resumen del Proyecto
                
                SmartForecast es un sistema de predicción de inventarios que utiliza redes neuronales LSTM 
                para optimizar las decisiones de compra y reducir costos asociados al manejo de inventarios.
                
                ### Objetivos Principales
                - Comparar la efectividad de un modelo LSTM general vs. un modelo específico por producto
                - Optimizar la predicción de demanda para reducir sobreabastecimiento y desabastecimiento
                - Proporcionar insights accionables para la gestión de inventarios
                
                ### Metodología
                - **Preprocesamiento:** Limpieza y transformación de datos de inventario
                - **Modelo General:** LSTM entrenado con datos de todos los productos
                - **Modelo Específico:** LSTM entrenado exclusivamente con un producto seleccionado
                - **Evaluación:** Comparación usando métricas MAE, MSE y RMSE
                """)
            
            with gr.Column(scale=1):
                # Mostrar estadísticas clave
                if self.preprocessing_summary:
                    stats_text = f"""
                    ## 📈 Estadísticas del Dataset
                    
                    **Datos Originales:**
                    - Filas: {self.preprocessing_summary.get('datos_originales', {}).get('filas', 'N/A')}
                    - Columnas: {self.preprocessing_summary.get('datos_originales', {}).get('columnas', 'N/A')}
                    
                    **Series Temporales:**
                    - Productos únicos: {self.preprocessing_summary.get('series_temporales', {}).get('num_products', 'N/A')}
                    - Observaciones: {self.preprocessing_summary.get('series_temporales', {}).get('shape', ['N/A', 'N/A'])[0]}
                    
                    **Producto Seleccionado:**
                    - Código: {self.preprocessing_summary.get('producto_seleccionado', {}).get('codigo', 'N/A')}
                    - Ventas Totales: {self.preprocessing_summary.get('producto_seleccionado', {}).get('total_ventas', 'N/A')}
                    """
                    gr.Markdown(stats_text)
    
    def create_data_exploration_tab(self):
        """
        Crea la pestaña de exploración de datos.
        """
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🔍 Exploración de Datos")
                
                # Selector de producto para visualización
                if not self.ts_data.empty:
                    products = self.ts_data['codigo_producto'].unique()[:50]  # Limitar a 50 productos
                    product_selector = gr.Dropdown(
                        choices=list(products),
                        value=products[0] if len(products) > 0 else None,
                        label="Seleccionar Producto para Visualización"
                    )
                    
                    # Gráfico de series de tiempo por producto
                    plot_output = gr.Plot(label="Serie de Tiempo del Producto")
                    
                    def update_product_plot(selected_product):
                        if selected_product and not self.ts_data.empty:
                            product_data = self.ts_data[self.ts_data['codigo_producto'] == selected_product]
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=product_data['fecha'],
                                y=product_data['ventas'],
                                mode='lines+markers',
                                name=f'Ventas - {selected_product}',
                                line=dict(color='#1f77b4', width=2),
                                marker=dict(size=6)
                            ))
                            
                            fig.update_layout(
                                title=f'Evolución de Ventas - Producto {selected_product}',
                                xaxis_title='Fecha',
                                yaxis_title='Ventas',
                                template='plotly_white',
                                height=400
                            )
                            
                            return fig
                        return go.Figure()
                    
                    product_selector.change(
                        fn=update_product_plot,
                        inputs=[product_selector],
                        outputs=[plot_output]
                    )
                    
                    # Inicializar con el primer producto
                    if len(products) > 0:
                        plot_output.value = update_product_plot(products[0])
    
    def create_model_comparison_tab(self):
        """
        Crea la pestaña de comparación de modelos.
        """
        with gr.Row():
            with gr.Column():
                gr.Markdown("## ⚖️ Comparación de Modelos")
                
                # Mostrar resultados de la comparación
                if self.comparison_results:
                    product_id = self.comparison_results.get('producto_id', 'N/A')
                    specific_results = self.comparison_results.get('modelo_especifico', {})
                    general_results = self.comparison_results.get('modelo_general', {})
                    
                    # Crear tabla de comparación
                    comparison_data = {
                        'Métrica': ['MAE', 'MSE', 'RMSE'],
                        'Modelo Específico': [
                            f"{specific_results.get('mae', 0):.4f}",
                            f"{specific_results.get('mse', 0):.4f}",
                            f"{specific_results.get('rmse', 0):.4f}"
                        ],
                        'Modelo General': [
                            f"{general_results.get('mae', 0):.4f}",
                            f"{general_results.get('mse', 0):.4f}",
                            f"{general_results.get('rmse', 0):.4f}"
                        ]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    gr.Markdown(f"### Resultados para el Producto: {product_id}")
                    gr.Dataframe(comparison_df, label="Métricas de Evaluación")
                    
                    # Gráfico de barras comparativo
                    fig = go.Figure()
                    
                    metrics = ['MAE', 'MSE', 'RMSE']
                    specific_values = [specific_results.get('mae', 0), specific_results.get('mse', 0), specific_results.get('rmse', 0)]
                    general_values = [general_results.get('mae', 0), general_results.get('mse', 0), general_results.get('rmse', 0)]
                    
                    fig.add_trace(go.Bar(
                        name='Modelo Específico',
                        x=metrics,
                        y=specific_values,
                        marker_color='#2E8B57'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='Modelo General',
                        x=metrics,
                        y=general_values,
                        marker_color='#CD5C5C'
                    ))
                    
                    fig.update_layout(
                        title='Comparación de Métricas de Error',
                        xaxis_title='Métricas',
                        yaxis_title='Valor del Error',
                        barmode='group',
                        template='plotly_white',
                        height=400
                    )
                    
                    gr.Plot(fig, label="Comparación Visual de Métricas")
                    
                    # Análisis de resultados
                    mae_improvement = ((general_results.get('mae', 0) - specific_results.get('mae', 0)) / general_results.get('mae', 1)) * 100
                    rmse_improvement = ((general_results.get('rmse', 0) - specific_results.get('rmse', 0)) / general_results.get('rmse', 1)) * 100
                    
                    analysis_text = f"""
                    ### 📊 Análisis de Resultados
                    
                    **Conclusiones Clave:**
                    - El modelo específico muestra una mejora del **{mae_improvement:.1f}%** en MAE comparado con el modelo general
                    - La mejora en RMSE es del **{rmse_improvement:.1f}%**
                    - {"✅ El modelo específico es superior" if mae_improvement > 0 else "❌ El modelo general es superior"} para este producto
                    
                    **Implicaciones:**
                    - {"Los modelos específicos por producto pueden ofrecer mejor precisión" if mae_improvement > 0 else "Un modelo general puede ser suficiente para la mayoría de productos"}
                    - Se recomienda {"implementar modelos específicos para productos críticos" if mae_improvement > 0 else "usar el modelo general como baseline"}
                    """
                    
                    gr.Markdown(analysis_text)
    
    def create_insights_tab(self):
        """
        Crea la pestaña de insights y recomendaciones.
        """
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ## 💡 Insights y Recomendaciones
                
                ### Hallazgos Principales
                
                **1. Efectividad del Modelo Específico**
                - Los modelos LSTM específicos por producto demuestran mayor precisión en la predicción
                - La especialización permite capturar patrones únicos de demanda de cada producto
                - Reducción significativa en las métricas de error comparado con el modelo general
                
                **2. Consideraciones de Implementación**
                - **Productos de Alto Volumen:** Justifican el desarrollo de modelos específicos
                - **Productos de Bajo Volumen:** Pueden beneficiarse del modelo general
                - **Recursos Computacionales:** Balance entre precisión y costo computacional
                
                **3. Estrategia Híbrida Recomendada**
                - Implementar modelos específicos para productos críticos (clasificación ABC)
                - Usar el modelo general como baseline para productos de menor importancia
                - Monitoreo continuo del rendimiento y re-entrenamiento periódico
                
                ### Próximos Pasos
                
                **Corto Plazo:**
                - Validar resultados con datos adicionales
                - Implementar sistema de monitoreo de precisión
                - Desarrollar pipeline automatizado de re-entrenamiento
                
                **Mediano Plazo:**
                - Incorporar variables externas (estacionalidad, promociones)
                - Explorar arquitecturas más avanzadas (Transformer, GRU)
                - Desarrollar sistema de alertas para anomalías en predicciones
                
                **Largo Plazo:**
                - Integración con sistemas ERP existentes
                - Desarrollo de API para predicciones en tiempo real
                - Expansión a múltiples categorías de productos
                """)
    
    def create_interface(self):
        """
        Crea la interfaz principal de Gradio.
        """
        with gr.Blocks(
            title="SmartForecast - Predicción de Inventarios",
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .tab-nav {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            }
            """
        ) as interface:
            
            gr.Markdown("""
            <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
                <h1>🚀 SmartForecast</h1>
                <h3>Sistema Inteligente de Predicción de Inventarios con Deep Learning</h3>
                <p>Optimiza tu gestión de inventarios con modelos LSTM avanzados</p>
            </div>
            """)
            
            with gr.Tabs():
                with gr.Tab("📋 Resumen del Proyecto"):
                    self.create_overview_tab()
                
                with gr.Tab("🔍 Exploración de Datos"):
                    self.create_data_exploration_tab()
                
                with gr.Tab("⚖️ Comparación de Modelos"):
                    self.create_model_comparison_tab()
                
                with gr.Tab("💡 Insights y Recomendaciones"):
                    self.create_insights_tab()
            
            gr.Markdown("""
            <div style="text-align: center; padding: 10px; margin-top: 20px; color: #666;">
                <p>Desarrollado por el Equipo SmartForecast | Proyecto de Deep Learning 2025</p>
            </div>
            """)
        
        return interface

def main():
    """
    Función principal para ejecutar la aplicación Gradio.
    """
    print("🚀 Iniciando SmartForecast App...")
    
    # Crear instancia de la aplicación
    app = SmartForecastApp()
    
    # Crear interfaz
    interface = app.create_interface()
    
    # Lanzar la aplicación
    interface.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
