"""
Training Data Manager for Tabular NCD Classification.

Maneja la carga de datos tabulares procesados (train/val/test CSVs) 
y proporciona DataLoaders de PyTorch para el entrenamiento federado.

Patrón Singleton igual que en image_classification para mantener 
una única instancia del DataManager durante toda la ejecución.
"""
import os
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from typing import Tuple
from .data_preparation import preprocess_split_dfs


class TabularDataset(Dataset):
    """
    Dataset tabular compatible con PyTorch.
    Carga datos desde un DataFrame con features numéricas + columna 'target'.
    """
    def __init__(self, dataframe: pd.DataFrame, target_col: str = "target"):
        """
        Args:
            dataframe: DataFrame con features numéricas + columna target
            target_col: Nombre de la columna target (default: 'target')
        """
        self.X = dataframe.drop(columns=[target_col]).values.astype("float32")
        self.y = dataframe[target_col].values.astype("float32")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )


class DataManager:
    """
    Maneja datasets y DataLoaders para clasificación tabular binaria.
    
    Singleton Pattern - igual que en image_classification/ic_training.py
    Esto asegura que solo haya una instancia cargando los datos en memoria.
    
    Uso:
        # Primera inicialización con threshold
        dm = DataManager.dm(th=100)
        
        # Accesos posteriores sin threshold
        dm = DataManager.dm()
        
        # Usar DataLoaders
        for X_batch, y_batch in dm.trainloader:
            # entrenar modelo
    """
    _singleton_dm = None

    @classmethod
    def dm(cls, th: int = 0, agent_name: str = 'a1'):
        """
        Obtener la instancia singleton del DataManager.
        
        Args:
            th: Threshold de cutoff para training (solo en primera inicialización)
            agent_name: Nombre del agente (determina qué datos cargar)
        
        Returns:
            DataManager: Instancia única del DataManager
        """
        if not cls._singleton_dm and th > 0:
            cls._singleton_dm = cls(th, agent_name)
        return cls._singleton_dm

    def __init__(self, cutoff_th: int, agent_name: str = 'a1'):
        """
        Inicializa el DataManager cargando los CSVs procesados.
        
        Args:
            cutoff_th: Umbral de cutoff para limitar datos de entrenamiento
                      (útil para simular escenarios con pocos datos)
            agent_name: Nombre del agente (a1, a2, a3, a4)
        """
        # Ruta base: examples/tabular_ncd/data/processed/{agent_name}/
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, "data", "processed", agent_name)
        
        # Cargar CSVs procesados
        train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
        test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

        # Aplicar preprocesamiento adicional si es necesario
        # (en este caso ya están procesados, pero mantenemos la función por compatibilidad)
        train_p, val_p, test_p, feat_cols = preprocess_split_dfs(train_df, val_df, test_df)

        # Crear PyTorch Datasets
        trainset = TabularDataset(train_p, target_col="target")
        valset = TabularDataset(val_p, target_col="target")
        testset = TabularDataset(test_p, target_col="target")

        # Crear DataLoaders con batch_size=32 (ajustable según necesidad)
        self.trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
        self.valloader = DataLoader(valset, batch_size=32, shuffle=False)
        self.testloader = DataLoader(testset, batch_size=32, shuffle=False)

        # Guardar metadata útil
        self.cutoff_threshold = cutoff_th
        self.input_dim = len(feat_cols)  # Número de features después del preprocesamiento
        self.agent_name = agent_name
        
        print(f"[DataManager] Inicializado para agente: {agent_name}")
        print(f"[DataManager] {self.input_dim} features")
        print(f"  - Train samples: {len(trainset)}")
        print(f"  - Val samples: {len(valset)}")
        print(f"  - Test samples: {len(testset)}")

    def get_random_batch(self, is_train: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Obtener un batch aleatorio de datos (útil para debugging/demos).
        
        Args:
            is_train: True para batch de training, False para test
        
        Returns:
            tuple: (features, labels) como tensores de PyTorch
        """
        loader = self.trainloader if is_train else self.testloader
        features, labels = next(iter(loader))
        return features, labels


def execute_tabular_training(dm: DataManager, net, criterion, optimizer, epochs: int = 10):
    """
    Rutina de entrenamiento para clasificación tabular binaria.
    
    Similar a execute_ic_training en image_classification, pero adaptado
    para datos tabulares y clasificación binaria.
    
    Args:
        dm: DataManager con los datos
        net: Red neuronal (MLP)
        criterion: Función de pérdida (ej: BCEWithLogitsLoss)
        optimizer: Optimizador (ej: Adam)
        epochs: Número de épocas de entrenamiento
    
    Returns:
        net: Red entrenada
    """
    net.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        num_batches = 0
        
        for X_batch, y_batch in dm.trainloader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = net(X_batch)
            
            # Reshape para compatibilidad con BCEWithLogitsLoss
            loss = criterion(logits, y_batch.unsqueeze(1).float())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
        
        avg_loss = running_loss / num_batches if num_batches > 0 else 0
        print(f'  Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}')
    
    return net


if __name__ == "__main__":
    """Test del DataManager"""
    print("=" * 60)
    print("Testing DataManager para Tabular NCD")
    print("=" * 60)
    
    # Inicializar con threshold de 100
    dm = DataManager.dm(th=100)
    
    # Verificar que funciona el singleton
    dm2 = DataManager.dm()
    assert dm is dm2, "Singleton pattern no funciona correctamente"
    print("\n✓ Singleton pattern funcionando correctamente")
    
    # Probar obtener un batch
    print("\n" + "=" * 60)
    print("Obteniendo batch aleatorio de training:")
    X_batch, y_batch = dm.get_random_batch(is_train=True)
    print(f"  - Shape de features: {X_batch.shape}")
    print(f"  - Shape de labels: {y_batch.shape}")
    print(f"  - Rango de features: [{X_batch.min():.2f}, {X_batch.max():.2f}]")
    print(f"  - Distribución de labels: {torch.bincount(y_batch.long())}")
    print("\n✓ DataManager funcionando correctamente")
