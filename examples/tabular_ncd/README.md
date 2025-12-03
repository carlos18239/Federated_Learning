# Federated Learning - Tabular NCD Classification

Sistema de aprendizaje federado para clasificación binaria de mortalidad prematura por enfermedades no transmisibles (NCD) usando datos tabulares distribuidos.

## 📋 Descripción

Este módulo implementa un cliente de federated learning para clasificar mortalidad prematura (`is_premature_ncd`) usando datos de hospitales distribuidos en diferentes nodos (Raspberry Pi).

**Características:**
- ✅ Preprocesamiento con pipeline compartido (OneHotEncoder + StandardScaler)
- ✅ Arquitectura MLP optimizada para datos tabulares
- ✅ Early stopping con Judge (max_rounds + patience)
- ✅ Métricas por ronda (val_acc, test_acc)
- ✅ Plots de evolución (local vs global)
- ✅ Compatible con el sistema FL centralizado

## 📁 Estructura de Archivos

```
examples/tabular_ncd/
├── data/
│   ├── data1.csv                      # Datos crudos Raspberry Pi 1
│   ├── data2.csv                      # Datos crudos Raspberry Pi 2
│   ├── data3.csv                      # Datos crudos Raspberry Pi 3
│   ├── data4.csv                      # Datos crudos Raspberry Pi 4
│   ├── preprocessor_global.joblib     # Pipeline compartido (CREAR PRIMERO)
│   ├── processed/                     # Salida del preprocesamiento (automático)
│   │   ├── a1/
│   │   │   ├── train.csv, val.csv, test.csv
│   │   ├── a2/
│   │   │   ├── train.csv, val.csv, test.csv
│   │   ├── a3/
│   │   │   ├── train.csv, val.csv, test.csv
│   │   └── a4/
│   │       ├── train.csv, val.csv, test.csv
│   └── models/                        # Modelos y métricas por agente
│       ├── a1/
│       │   ├── *.npz, metrics_*.csv, results_*.png
│       ├── a2/
│       ├── a3/
│       └── a4/
├── binary_classification.py           # Cliente FL principal
├── data_preparation.py                # Preprocesamiento de datos
├── tnn_training.py                    # DataManager y entrenamiento
├── cnn.py                             # Arquitecturas MLP
├── conversion.py                      # Conversión PyTorch ↔ NumPy
├── judge.py                           # Early stopping
├── create_preprocessor.py             # Crear preprocessor global
└── README.md                          # Este archivo
```

## 🚀 Guía de Uso

### Paso 1: Crear el Preprocessor Global (UNA SOLA VEZ)

El preprocessor debe crearse **una sola vez** usando todos los datos disponibles:

```bash
# Asegúrate de tener data1.csv, data2.csv, data3.csv, data4.csv en examples/tabular_ncd/data/
python -m examples.tabular_ncd.create_preprocessor
```

Esto creará `examples/tabular_ncd/data/preprocessor_global.joblib` que será usado por todos los agentes.

### Paso 2: Configuración del Sistema FL

Edita `setups/config_agent.json` y `setups/config_aggregator.json` si es necesario:

```json
{
  "aggr_ip": "localhost",  // o IP del servidor agregador
  "reg_socket": "8765",
  "model_path": "./data/agents",
  "polling": 1
}
```

### Paso 3: Iniciar el Sistema FL

**Terminal 1 - Base de datos:**
```bash
python -m fl_main.pseudodb.pseudo_db
```

**Terminal 2 - Agregador:**
```bash
python -m fl_main.aggregator.server_th
```

**Terminales 3-6 - Agentes (modo simulación):**
```bash
# Agente 1 (usa data1.csv)
python -m examples.tabular_ncd.binary_classification 1 50001 a1

# Agente 2 (usa data2.csv)
python -m examples.tabular_ncd.binary_classification 1 50002 a2

# Agente 3 (usa data3.csv)
python -m examples.tabular_ncd.binary_classification 1 50003 a3

# Agente 4 (usa data4.csv)
python -m examples.tabular_ncd.binary_classification 1 50004 a4
```

**Argumentos:**
- `1`: Modo simulación activado
- `5000X`: Socket de intercambio único por agente
- `aX`: Nombre del agente (determina qué archivo data usar)

### Modo Producción (Raspberry Pi)

## 📊 Salidas Generadas

### Modelos Guardados (`.npz`)
```
data/models/
├── 20251202-143052_init.npz          # Modelo inicial
├── 20251202-143105_global_r1.npz     # Global modelo ronda 1
├── 20251202-143120_local_r1.npz      # Local modelo ronda 1
├── 20251202-143135_global_r2.npz     # Global modelo ronda 2
└── ...
```

### Métricas (CSV)
```
data/models/metrics_ncd_rpi1.csv
```

Columnas: `timestamp`, `round`, `kind` (local/global), `val_acc`, `test_acc`

### Gráficas
```
data/models/results_ncd_rpi1.png
```

Muestra 2 subplots:
- Izquierda: Val Accuracy por ronda (local vs global)
- Derecha: Test Accuracy por ronda (local vs global)

## 🔧 Configuración Avanzada

### Ajustar el Modelo

En `cnn.py`, puedes modificar la arquitectura del MLP:

```python
class MLP(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)  # ← Cambiar tamaño
        self.fc2 = nn.Linear(128, 64)           # ← de capas aquí
        # ...
```

### Ajustar Hiperparámetros

En `binary_classification.py`:

```python
# Early stopping
judge = Judge(max_rounds=50, patience=5, min_delta=1e-4)

# Entrenamiento
optimizer = optim.Adam(net.parameters(), lr=1e-3)  # ← Learning rate
execute_tabular_training(dm, net, criterion, optimizer, epochs=10)  # ← Épocas
```

### Balanceo de Clases

En `data_preparation.py`, cambiar:

```python
cfg = {
    "balance_strategy": "undersample_majority"  # o "none"
}
```

## 📈 Monitoreo del Entrenamiento

Durante la ejecución verás logs como:

```
[Round 1] Esperando modelo global...
✓ Modelo global recibido (Round 1)
Evaluando modelo global...
[Global Model] Val Acc: 67.34% | Test Acc: 65.12%
--- Iniciando entrenamiento local ---
  Epoch [1/10] - Loss: 0.6234
  Epoch [2/10] - Loss: 0.5891
  ...
Evaluando modelo local...
[Local Model]  Val Acc: 71.23% | Test Acc: 69.45%
✓ Modelo local enviado
```

## 🐛 Troubleshooting

### Error: "Preprocessor no encontrado"
```bash
# Asegúrate de que existe:
ls examples/tabular_ncd/data/preprocessor_global.joblib
```

### Error: "CSVs procesados no encontrados"
```bash
# Ejecuta primero el preprocesamiento:
python -m examples.tabular_ncd.data_preparation
```

### Error: "Connection lost to the agent"
- Verifica la IP del agregador en `setups/config_agent.json`
- Asegúrate de que el agregador está corriendo
- Verifica firewall/puertos abiertos

### Modelos no convergen
- Aumenta `epochs` en `execute_tabular_training()`
- Ajusta learning rate en el optimizer
- Verifica distribución de clases con `balance_strategy="undersample_majority"`

## 📚 Referencias

- Sistema FL base: `examples/image_classification/`
- Preprocesamiento: `data_preparation.py` (usa `preprocessor_global.joblib`)
- Arquitectura: Similar a `examples/heart_disease/` pero para datos NCD

## 🤝 Contribuciones

Este módulo sigue el patrón de diseño del sistema FL centralizado existente:
- **Singleton Pattern** para DataManager y Converter
- **Async/Await** para comunicación WebSocket
- **NumPy serialization** para transmisión de modelos
- **Judge Pattern** para early stopping

---

**Nota:** Este es un sistema de investigación educativa. Para uso en producción, considera agregar:
- Validación de datos más robusta
- Manejo de errores de red
- Checkpointing de modelos
- Logging estructurado
- Tests unitarios
