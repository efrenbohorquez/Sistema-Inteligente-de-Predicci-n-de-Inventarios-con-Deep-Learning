#!/usr/bin/env python3
"""
SmartForecast - Modelo General LSTM para Predicción de Inventarios

Este módulo implementa un modelo LSTM (Long Short-Term Memory) general para la predicción
de ventas de inventarios. El modelo es entrenado con datos de múltiples productos para
capturar patrones generales de demanda y ventas.

Características principales:
- Arquitectura LSTM con capas de Dropout para regularización
- Normalización automática de datos con MinMaxScaler
- Generación de secuencias temporales optimizadas
- Métricas de evaluación completas (MAE, MSE, RMSE)
- Visualizaciones automáticas de resultados

Autor: Efrén Bohórquez
Repositorio: https://github.com/efrenbohorquez/Sistema-Inteligente-de-Predicci-n-de-Inventarios-con-Deep-Learning
Fecha: Octubre 2025
Versión: 1.0.0
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
import json
from typing import Tuple, Dict, Any

class GeneralLSTMModel:
    """
    Clase para el modelo LSTM general de predicción de ventas.
    """
    
    def __init__(self, data_path: str, output_dir: str = 'modelo_general_output'):
        """
        Inicializa el modelo.
        
        Args:
            data_path (str): Ruta al archivo CSV con los datos de series de tiempo.
            output_dir (str): Directorio para guardar los resultados.
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
        self.evaluation_results = {}
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carga los datos y los prepara para el modelo LSTM.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tupla con datos de entrenamiento (X, y).
        """
        print("Cargando y preparando datos...")
        # Leer CSV
        self.data = pd.read_csv(self.data_path)
        # Limpiar nombres de columnas de posibles espacios o caracteres extraños
        self.data.columns = self.data.columns.str.strip()
        print(f"Columnas encontradas: {list(self.data.columns)}")
        # Convertir la columna fecha al tipo datetime
        self.data['fecha'] = pd.to_datetime(self.data['fecha'])
        # Reducir dataset al 10% para entrenamiento más rápido
        self.data = self.data.sample(frac=0.1, random_state=42).reset_index(drop=True)
        
        # Usar solo la columna de ventas para el modelo univariado
        sales_data = self.data[['ventas']].values.astype('float32')
        
        # Normalizar los datos
        scaled_sales = self.scaler.fit_transform(sales_data)
        
        # Crear secuencias de datos
        X, y = self._create_sequences(scaled_sales, look_back=6)
        
        print(f"Datos preparados: X shape={X.shape}, y shape={y.shape}")
        return X, y

    def _create_sequences(self, dataset: np.ndarray, look_back: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea secuencias de entrada (X) y salida (y) para el modelo LSTM.
        """
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    def build_model(self, look_back: int = 6):
        """
        Construye la arquitectura del modelo LSTM.
        """
        print("Construyendo el modelo LSTM...")
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        print("Modelo construido y compilado.")
        self.model.summary()

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 20, batch_size: int = 64):
        """
        Entrena el modelo LSTM.
        """
        print("Entrenando el modelo...")
        # Reshape de X para que sea [muestras, timesteps, caracteristicas]
        X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        self.history = self.model.fit(
            X_train_reshaped,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2, # Usar 20% de los datos para validación
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            ]
        )
        print("Entrenamiento completado.")

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Evalúa el modelo en el conjunto de prueba.
        """
        print("Evaluando el modelo...")
        X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # Realizar predicciones
        predictions = self.model.predict(X_test_reshaped)
        
        # Invertir la normalización para obtener valores reales
        y_test_inv = self.scaler.inverse_transform([y_test])
        predictions_inv = self.scaler.inverse_transform(predictions)
        
        # Calcular métricas
        mae = mean_absolute_error(y_test_inv[0], predictions_inv[:,0])
        mse = mean_squared_error(y_test_inv[0], predictions_inv[:,0])
        rmse = np.sqrt(mse)
        
        self.evaluation_results = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        }
        
        print(f"Resultados de la evaluación:")
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        
        # Guardar resultados
        with open(os.path.join(self.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
            
        return y_test_inv[0], predictions_inv[:,0]

    def plot_results(self, y_true, y_pred):
        """
        Genera y guarda gráficos de los resultados.
        """
        print("Generando gráficos de resultados...")
        
        # 1. Gráfico de pérdida de entrenamiento y validación
        plt.figure(figsize=(12, 6))
        plt.plot(self.history.history['loss'], label='Pérdida de Entrenamiento')
        plt.plot(self.history.history['val_loss'], label='Pérdida de Validación')
        plt.title('Pérdida del Modelo Durante el Entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Pérdida (MSE)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'training_loss.png'), dpi=300)
        plt.close()
        
        # 2. Gráfico de predicciones vs. valores reales (muestra)
        plt.figure(figsize=(14, 7))
        sample_size = 300
        plt.plot(y_true[:sample_size], label='Valores Reales', marker='.')
        plt.plot(y_pred[:sample_size], label='Predicciones', marker='.')
        plt.title(f'Comparación de Predicciones vs. Valores Reales (Muestra de {sample_size} puntos)')
        plt.xlabel('Índice de Tiempo')
        plt.ylabel('Ventas')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'predictions_vs_actuals.png'), dpi=300)
        plt.close()
        
        print(f"Gráficos guardados en {self.output_dir}")

    def save_model(self):
        """
        Guarda el modelo entrenado.
        """
        model_path = os.path.join(self.output_dir, 'modelo_general.h5')
        self.model.save(model_path)
        print(f"Modelo guardado en {model_path}")

def main():
    """
    Función principal para ejecutar el pipeline del modelo general.
    """
    print("=== INICIANDO PIPELINE DEL MODELO GENERAL LSTM ===")
    
    # Configuración
    DATA_FILE = 'series_temporales.csv'
    LOOK_BACK = 6
    EPOCHS = 5 # Reducido para ejecución más rápida en este entorno
    BATCH_SIZE = 256
    
    # Crear instancia del modelo
    lstm_model = GeneralLSTMModel(data_path=DATA_FILE)
    
    # Cargar y preparar datos
    X, y = lstm_model.load_and_prepare_data()
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    print(f"División de datos: Train={len(X_train)}, Test={len(X_test)}")
    
    # Construir el modelo
    lstm_model.build_model(look_back=LOOK_BACK)
    
    # Entrenar el modelo
    lstm_model.train_model(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
    
    # Evaluar el modelo
    y_true, y_pred = lstm_model.evaluate_model(X_test, y_test)
    
    # Generar gráficos
    lstm_model.plot_results(y_true, y_pred)
    
    # Guardar el modelo
    lstm_model.save_model()
    
    print("\n=== PIPELINE DEL MODELO GENERAL COMPLETADO EXITOSAMENTE ===")

if __name__ == "__main__":
    main()

