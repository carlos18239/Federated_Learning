"""
Neural Network Architecture for Tabular NCD Binary Classification.

Define el MLP (Multi-Layer Perceptron) para clasificación binaria
de mortalidad prematura por enfermedades no transmisibles (NCD).

Red completamente conectada (fully connected) apropiada para datos tabulares.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-Layer Perceptron para clasificación binaria de datos tabulares.
    
    Arquitectura:
        - Input: n features (determinado por el preprocessor)
        - Hidden Layer 1: 128 neurons + ReLU + Dropout(0.3)
        - Hidden Layer 2: 64 neurons + ReLU + Dropout(0.3)
        - Hidden Layer 3: 32 neurons + ReLU
        - Output: 1 neuron (logit para clasificación binaria)
    
    Nota: La salida es un logit (sin sigmoid), compatible con BCEWithLogitsLoss
    """
    
    def __init__(self, in_features: int):
        """
        Inicializa la red MLP.
        
        Args:
            in_features: Número de features de entrada (depende del preprocessor)
        """
        super(MLP, self).__init__()
        
        # Capas fully connected
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)  # Salida: 1 logit
        
        # Dropout para regularización (prevenir overfitting)
        self.dropout = nn.Dropout(0.3)
        
        # Guardar dimensión de entrada para referencia
        self.in_features = in_features
    
    def forward(self, x):
        """
        Forward pass del MLP.
        
        Args:
            x: Tensor de entrada con shape (batch_size, in_features)
        
        Returns:
            Tensor de salida con shape (batch_size, 1) conteniendo logits
        """
        # Layer 1: Linear + ReLU + Dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 2: Linear + ReLU + Dropout
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 3: Linear + ReLU (sin dropout antes de la capa final)
        x = self.fc3(x)
        x = F.relu(x)
        
        # Output layer: Linear (logit sin activación)
        x = self.fc4(x)
        
        return x
    
    def predict_proba(self, x):
        """
        Predice probabilidades (útil para evaluación).
        
        Args:
            x: Tensor de entrada
        
        Returns:
            Probabilidades entre 0 y 1
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs
    
    def predict(self, x, threshold: float = 0.5):
        """
        Predice clases binarias (0 o 1).
        
        Args:
            x: Tensor de entrada
            threshold: Umbral de clasificación (default: 0.5)
        
        Returns:
            Predicciones binarias (0 o 1)
        """
        probs = self.predict_proba(x)
        return (probs >= threshold).long()


class SimpleMLP(nn.Module):
    """
    Versión más simple del MLP (útil para pruebas rápidas o pocos datos).
    
    Arquitectura más ligera:
        - Input: n features
        - Hidden Layer 1: 64 neurons + ReLU
        - Hidden Layer 2: 32 neurons + ReLU
        - Output: 1 neuron (logit)
    """
    
    def __init__(self, in_features: int):
        super(SimpleMLP, self).__init__()
        
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.in_features = in_features
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    """Test de las arquitecturas de red"""
    
    print("=" * 60)
    print("Testing MLP Architecture")
    print("=" * 60)
    
    # Supongamos que tenemos 50 features después del preprocessing
    in_features = 50
    batch_size = 8
    
    # Crear modelo
    model = MLP(in_features=in_features)
    print(f"\n✓ Modelo MLP creado con {in_features} features de entrada")
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total de parámetros: {total_params:,}")
    print(f"  - Parámetros entrenables: {trainable_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(batch_size, in_features)
    output = model(dummy_input)
    
    print(f"\n✓ Forward pass exitoso")
    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Output range: [{output.min():.2f}, {output.max():.2f}]")
    
    # Test predict_proba
    probs = model.predict_proba(dummy_input)
    print(f"\n✓ Predict proba exitoso")
    print(f"  - Probabilities shape: {probs.shape}")
    print(f"  - Probabilities range: [{probs.min():.4f}, {probs.max():.4f}]")
    
    # Test predict
    predictions = model.predict(dummy_input)
    print(f"\n✓ Predict exitoso")
    print(f"  - Predictions: {predictions.squeeze()}")
    
    # Test SimpleMLP
    print("\n" + "=" * 60)
    print("Testing SimpleMLP Architecture")
    print("=" * 60)
    
    simple_model = SimpleMLP(in_features=in_features)
    simple_params = sum(p.numel() for p in simple_model.parameters())
    print(f"\n✓ Modelo SimpleMLP creado")
    print(f"  - Total de parámetros: {simple_params:,}")
    
    simple_output = simple_model(dummy_input)
    print(f"✓ Forward pass SimpleMLP exitoso")
    print(f"  - Output shape: {simple_output.shape}")
    
    print("\n" + "=" * 60)
    print("✓ Todas las pruebas pasaron correctamente")
    print("=" * 60)
