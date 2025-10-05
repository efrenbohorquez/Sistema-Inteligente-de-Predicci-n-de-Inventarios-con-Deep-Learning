# Contribuciones - SmartForecast

¡Gracias por tu interés en contribuir a SmartForecast! Este documento te guiará através del proceso de contribución.

## 🤝 Cómo Contribuir

### 1. Fork y Clone
```bash
# Fork el repositorio en GitHub, luego:
git clone https://github.com/tu-usuario/Sistema-Inteligente-de-Predicci-n-de-Inventarios-con-Deep-Learning.git
cd Sistema-Inteligente-de-Predicci-n-de-Inventarios-con-Deep-Learning
```

### 2. Configurar Entorno de Desarrollo
```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Dependencias de desarrollo
```

### 3. Crear Rama para tu Feature
```bash
git checkout -b feature/nombre-descriptivo
```

## 📋 Tipos de Contribuciones Bienvenidas

### 🐛 Reportar Bugs
- Usa el template de issue para bugs
- Incluye información de tu entorno (OS, Python version, etc.)
- Provee pasos para reproducir el error
- Adjunta logs o screenshots si es relevante

### 💡 Proponer Features
- Abre un issue describiendo la funcionalidad propuesta
- Explica el caso de uso y beneficios
- Discute la implementación antes de empezar a programar

### 🔧 Contribuciones de Código
- **Modelos ML/DL**: Nuevas arquitecturas (GRU, Transformer, etc.)
- **Visualizaciones**: Gráficos mejorados o nuevos dashboards
- **Optimizaciones**: Mejoras de rendimiento o eficiencia
- **Tests**: Casos de prueba unitarios e integración
- **Documentación**: Mejoras en docs, ejemplos, tutorials

### 📚 Contribuciones de Documentación
- Mejorar README.md
- Agregar ejemplos de uso
- Crear tutorials paso a paso
- Traducir documentación

## 🎯 Estándares de Código

### Python Style Guide
Seguimos [PEP 8](https://pep8.org/) con algunas adaptaciones:

```python
# ✅ Buen ejemplo
def predict_sales(data: pd.DataFrame, model: tf.keras.Model) -> np.ndarray:
    """
    Predice ventas usando el modelo entrenado.
    
    Args:
        data (pd.DataFrame): Datos de entrada preprocesados
        model (tf.keras.Model): Modelo LSTM entrenado
        
    Returns:
        np.ndarray: Predicciones de ventas
    """
    processed_data = preprocess_data(data)
    predictions = model.predict(processed_data)
    return predictions
```

### Documentación de Funciones
- Usar docstrings estilo Google
- Documentar todos los parámetros y valores de retorno
- Incluir ejemplos cuando sea relevante
- Agregar información sobre excepciones

### Commits
Usar [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Tipos de commit
feat: nueva funcionalidad
fix: corrección de bug
docs: cambios en documentación
style: formateo, sin cambios de código
refactor: refactoring de código
test: agregar o corregir tests
chore: cambios en build, dependencias, etc.

# Ejemplos
git commit -m "feat: add GRU model implementation"
git commit -m "fix: resolve data normalization issue"
git commit -m "docs: update API documentation"
```

## 🧪 Testing

### Ejecutar Tests
```bash
# Todos los tests
pytest

# Tests específicos
pytest tests/test_models.py

# Con cobertura
pytest --cov=src tests/
```

### Escribir Tests
```python
import pytest
import numpy as np
from src.models.lstm_general import GeneralLSTMModel

def test_model_initialization():
    """Test que el modelo se inicializa correctamente."""
    model = GeneralLSTMModel("test_data.csv")
    assert model.data_path == "test_data.csv"
    assert model.model is None

def test_data_preprocessing():
    """Test del preprocesamiento de datos."""
    # Implementar test...
    pass
```

## 📊 Benchmarking

Para contribuciones de modelos, incluir benchmarks:

```python
# Ejemplo de benchmark
def benchmark_model_performance():
    """Benchmark de rendimiento del modelo."""
    results = {
        "model_name": "LSTM_v2",
        "rmse": 45.23,
        "mae": 32.15,
        "training_time": "5.2 minutes",
        "inference_time": "0.05 seconds"
    }
    return results
```

## 🚀 Process de Pull Request

### 1. Pre-requisitos
- [ ] Fork del repositorio actualizado
- [ ] Rama feature creada desde main/master
- [ ] Código sigue los estándares establecidos
- [ ] Tests pasan localmente
- [ ] Documentación actualizada

### 2. Crear Pull Request
- Usar el template de PR
- Título descriptivo y conciso
- Descripción detallada de cambios
- Referenciar issues relacionados
- Incluir screenshots si hay cambios visuales

### 3. Review Process
1. **Automated Checks**: Tests, linting, coverage
2. **Code Review**: Revisión por maintainers
3. **Testing**: Verificación en diferentes entornos
4. **Merge**: Una vez aprobado, merge a main

## 📝 Templates

### Issue Template - Bug Report
```markdown
## 🐛 Bug Report

**Descripción del Bug**
Descripción clara y concisa del problema.

**Pasos para Reproducir**
1. Ir a '...'
2. Ejecutar '...'
3. Ver error

**Comportamiento Esperado**
Qué debería haber pasado.

**Screenshots**
Si aplica, agregar screenshots.

**Entorno**
- OS: [e.g. Windows 10]
- Python Version: [e.g. 3.9]
- TensorFlow Version: [e.g. 2.20.0]
```

### Pull Request Template
```markdown
## 📋 Pull Request

**Tipo de Cambio**
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

**Descripción**
Describe tus cambios en detalle.

**Testing**
- [ ] Tests existentes pasan
- [ ] Nuevos tests agregados
- [ ] Tests manuales realizados

**Checklist**
- [ ] Mi código sigue el style guide
- [ ] He revisado mi propio código
- [ ] He comentado código difícil de entender
- [ ] He actualizado la documentación
```

## 👥 Comunidad

### Canales de Comunicación
- **Issues**: Para bugs y feature requests
- **Discussions**: Para preguntas generales y ideas
- **Email**: contacto@efrenbohorquez.com para temas sensibles

### Código de Conducta
- Sé respetuoso y constructivo
- Acepta críticas constructivas
- Ayuda a otros miembros de la comunidad
- Mantén discusiones técnicas y relevantes

## 🏆 Reconocimiento

Los contribuidores serán reconocidos en:
- README.md (sección de contribuidores)
- CHANGELOG.md (para cada release)
- Menciones en redes sociales
- Invitaciones a eventos especiales del proyecto

## 📞 ¿Necesitas Ayuda?

- 📧 Email: contacto@efrenbohorquez.com
- 💬 GitHub Discussions
- 🐛 GitHub Issues (para problemas técnicos)

¡Gracias por contribuir a SmartForecast! 🚀