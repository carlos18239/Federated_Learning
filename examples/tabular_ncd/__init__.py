"""
Tabular NCD Federated Learning Module.

Módulo para clasificación binaria federada de mortalidad prematura
por enfermedades no transmisibles (NCD) usando datos tabulares.

Componentes:
    - data_preparation.py: Preprocesamiento de datos con preprocessor global
    - tnn_training.py: DataManager y rutinas de entrenamiento
    - cnn.py: Arquitecturas MLP para clasificación binaria
    - conversion.py: Conversión entre PyTorch y NumPy arrays
    - binary_classification.py: Cliente FL principal
    - judge.py: Early stopping y control de rondas
    - create_preprocessor.py: Crear preprocessor global
"""

__version__ = "1.0.0"
__all__ = [
    "DataManager",
    "MLP",
    "SimpleMLP",
    "Converter",
    "Judge",
    "training",
    "init_models",
]

from .tnn_training import DataManager
from .cnn import MLP, SimpleMLP
from .conversion import Converter
from .judge import Judge
