#!/usr/bin/env python3
"""
Script para el Modelo Específico y Comparación - SmartForecast

Este script entrena un modelo LSTM para un producto específico y lo compara
con el modelo general previamente entrenado.

Autor: Equipo SmartForecast
Fecha: Octubre 2025
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
import json
from typing import Tuple, Dict, Any

class SpecificLSTMModel:
    def __init__(self, data_path: str, product_id: str, general_model_path: str, output_dir: str = 'modelo_especifico_output'):
        self.data_path = data_path
        self.product_id = product_id
        self.general_model_path = general_model_path
        self.output_dir = output_dir
        self.data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
        self.evaluation_results = {}
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_and_prepare_data(self, look_back: int = 3):
        print(f"Cargando y preparando datos para el producto: {self.product_id}")
        full_data = pd.read_csv(self.data_path, parse_dates=['fecha'])
        self.data = full_data[full_data['codigo_producto'] == self.product_id].copy()
        
        sales_data = self.data[['ventas']].values.astype('float32')
        scaled_sales = self.scaler.fit_transform(sales_data)
        
        X, y = self._create_sequences(scaled_sales, look_back)
        return X, y

    def _create_sequences(self, dataset: np.ndarray, look_back: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    def build_model(self, look_back: int = 3):
        print("Construyendo el modelo LSTM específico...")
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.summary()

    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 1):
        print("Entrenando el modelo específico...")
        X_reshaped = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        self.history = self.model.fit(
            X_reshaped, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, 
            verbose=1, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)]
        )

    def evaluate_specific_model(self, X_test: np.ndarray, y_test: np.ndarray):
        print("Evaluando el modelo específico...")
        X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predictions = self.model.predict(X_test_reshaped)
        
        y_test_inv = self.scaler.inverse_transform([y_test])
        predictions_inv = self.scaler.inverse_transform(predictions)
        
        mae = mean_absolute_error(y_test_inv[0], predictions_inv[:,0])
        mse = mean_squared_error(y_test_inv[0], predictions_inv[:,0])
        rmse = np.sqrt(mse)
        
        return {'mae': mae, 'mse': mse, 'rmse': rmse}, y_test_inv[0], predictions_inv[:,0]

    def evaluate_general_model(self, X_test: np.ndarray, y_test: np.ndarray):
        print("Evaluando el modelo general en los datos del producto específico...")
        general_model = load_model(self.general_model_path)
        X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        full_sales_data = pd.read_csv(self.data_path)['ventas'].values.astype('float32').reshape(-1, 1)
        general_scaler = MinMaxScaler(feature_range=(0, 1))
        general_scaler.fit(full_sales_data)

        specific_sales_data = self.data[['ventas']].values.astype('float32')
        scaled_specific_data = general_scaler.transform(specific_sales_data)
        X_specific_general_scaled, y_specific_general_scaled = self._create_sequences(scaled_specific_data, look_back=6)
        
        _, X_test_general, _, y_test_general = train_test_split(X_specific_general_scaled, y_specific_general_scaled, test_size=0.2, random_state=42, shuffle=False)

        X_test_reshaped = np.reshape(X_test_general, (X_test_general.shape[0], X_test_general.shape[1], 1))
        predictions = general_model.predict(X_test_reshaped)
        
        y_test_inv = general_scaler.inverse_transform([y_test_general])
        predictions_inv = general_scaler.inverse_transform(predictions)
        
        mae = mean_absolute_error(y_test_inv[0], predictions_inv[:,0])
        mse = mean_squared_error(y_test_inv[0], predictions_inv[:,0])
        rmse = np.sqrt(mse)
        
        return {'mae': mae, 'mse': mse, 'rmse': rmse}, y_test_inv[0], predictions_inv[:,0]

    def plot_comparison(self, y_true, specific_preds, general_preds):
        plt.figure(figsize=(14, 7))
        plt.plot(y_true, label='Valores Reales', marker='o', linestyle='-')
        plt.plot(specific_preds, label='Predicciones Modelo Específico', marker='x', linestyle='--')
        plt.plot(general_preds, label='Predicciones Modelo General', marker='s', linestyle=':')
        plt.title(f'Comparación de Modelos para el Producto {self.product_id}')
        plt.xlabel('Índice de Tiempo')
        plt.ylabel('Ventas')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'), dpi=300)
        plt.close()

def main():
    print("=== INICIANDO PIPELINE DEL MODELO ESPECÍFICO Y COMPARACIÓN ===\n")
    
    DATA_FILE = 'series_temporales.csv'
    PRODUCT_ID = 'A3487FE4D9' # Producto seleccionado en el preprocesamiento
    GENERAL_MODEL_PATH = 'modelo_general_output/modelo_general.h5'
    LOOK_BACK = 3
    
    specific_model_pipeline = SpecificLSTMModel(DATA_FILE, PRODUCT_ID, GENERAL_MODEL_PATH)
    
    X, y = specific_model_pipeline.load_and_prepare_data(look_back=LOOK_BACK)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    specific_model_pipeline.build_model(look_back=LOOK_BACK)
    specific_model_pipeline.train_model(X_train, y_train)
    
    specific_eval, y_true_specific, specific_preds = specific_model_pipeline.evaluate_specific_model(X_test, y_test)
    general_eval, y_true_general, general_preds = specific_model_pipeline.evaluate_general_model(X_test, y_test)
    
    print("\n=== RESULTADOS DE LA COMPARACIÓN ===[0m")
    print(f"Producto: {PRODUCT_ID}")
    print("\nModelo Específico:")
    print(f"  MAE: {specific_eval['mae']:.4f}, MSE: {specific_eval['mse']:.4f}, RMSE: {specific_eval['rmse']:.4f}")
    print("\nModelo General:")
    print(f"  MAE: {general_eval['mae']:.4f}, MSE: {general_eval['mse']:.4f}, RMSE: {general_eval['rmse']:.4f}")
    
    # Guardar resultados de la comparación
    comparison_results = {
        'producto_id': PRODUCT_ID,
        'modelo_especifico': specific_eval,
        'modelo_general': general_eval
    }
    with open(os.path.join(specific_model_pipeline.output_dir, 'comparison_results.json'), 'w') as f:
        json.dump(comparison_results, f, indent=2)

    # Asegurarse de que los arrays de predicciones tengan la misma longitud para graficar
    min_len = min(len(y_true_specific), len(specific_preds), len(general_preds))
    specific_model_pipeline.plot_comparison(y_true_specific[:min_len], specific_preds[:min_len], general_preds[:min_len])

    print('\n=== PIPELINE DE COMPARACIÓN COMPLETADO EXITOSAMENTE ===')

if __name__ == "__main__":
    main()

