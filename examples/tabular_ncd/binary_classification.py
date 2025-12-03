"""
Federated Learning Client for Tabular NCD Binary Classification.

Cliente FL para clasificación binaria de mortalidad prematura por NCD.
Implementa el patrón completo de FL con:
 - Inicialización y participación en la plataforma FL
 - Loop de entrenamiento federado (recibir global model → entrenar → enviar)
 - Métricas por ronda (val_acc, test_acc) guardadas en CSV
 - Early stopping con Judge (max_rounds + patience)
 - Plots finales de accuracy (local vs global)

Uso por nodo (Raspberry Pi):
    1. Ejecutar cliente FL: python -m examples.tabular_ncd.binary_classification
    
Modo simulación (múltiples agentes en una máquina):
    python -m examples.tabular_ncd.binary_classification 1 50001 a1
    python -m examples.tabular_ncd.binary_classification 1 50002 a2
    python -m examples.tabular_ncd.binary_classification 1 50003 a3
    python -m examples.tabular_ncd.binary_classification 1 50004 a4
"""
import logging
import os
import sys
import time
import csv
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from .conversion import Converter
from .tnn_training import DataManager, execute_tabular_training
from .cnn import MLP
from fl_main.agent.client import Client
from .judge import Judge
from .data_preparation import get_default_config, run_preprocessing

# ---------------- Configuración ----------------
DATASET_TAG = os.environ.get("DATASET_TAG", "ncd_dataset1")
MODELS_DIR = None  # se inicializa con _models_dir()


def _get_agent_name_from_client(client) -> str:
    """Obtiene el nombre del agente del cliente FL"""
    return getattr(client, 'agent_name', 'a1')


def _map_agent_to_data_file(agent_name: str) -> str:
    """
    Mapea el nombre del agente al archivo de datos correspondiente.
    a1 → data1.csv, a2 → data2.csv, a3 → data3.csv, a4 → data4.csv
    """
    mapping = {
        'a1': 'data1.csv',
        'a2': 'data2.csv', 
        'a3': 'data3.csv',
        'a4': 'data4.csv',
        'default_agent': 'data1.csv'  # Fallback
    }
    return mapping.get(agent_name, 'data1.csv')


def _models_dir(agent_name: str = 'a1') -> str:
    """Directorio para guardar modelos y métricas por agente"""
    global MODELS_DIR
    if MODELS_DIR is None:
        base = os.path.dirname(os.path.abspath(__file__))
        d = os.path.join(base, "data", "models", agent_name)
        os.makedirs(d, exist_ok=True)
        MODELS_DIR = d
    return MODELS_DIR


def ensure_data_preprocessed(agent_name: str) -> None:
    """
    Verifica que existan los datos procesados para el agente.
    Si no existen, ejecuta el preprocesamiento automáticamente.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(base_dir, "data", "processed", agent_name)
    
    train_csv = os.path.join(processed_dir, "train.csv")
    
    if os.path.exists(train_csv):
        logging.info(f"✓ Datos procesados encontrados para {agent_name}")
        return
    
    logging.info(f"⚠ Datos procesados no encontrados para {agent_name}")
    logging.info(f"Ejecutando preprocesamiento automático...")
    
    # Obtener archivo de datos correspondiente
    data_file = _map_agent_to_data_file(agent_name)
    
    # Configurar preprocesamiento
    cfg = get_default_config(base_dir, data_file=data_file)
    cfg['output_dir'] = processed_dir  # Directorio específico por agente
    
    # Verificar que existan los archivos necesarios
    if not os.path.exists(cfg['raw_data_path']):
        raise FileNotFoundError(
            f"Archivo de datos no encontrado: {cfg['raw_data_path']}\n"
            f"Asegúrate de tener {data_file} en examples/tabular_ncd/data/"
        )
    
    if not os.path.exists(cfg['preprocessor_path']):
        raise FileNotFoundError(
            f"Preprocessor no encontrado: {cfg['preprocessor_path']}\n"
            f"Debe estar en: examples/tabular_ncd/data/preprocessor_global.joblib\n"
            f"Nota: Este archivo debe ser creado primero con los datos de todos los nodos."
        )
    
    # Ejecutar preprocesamiento
    meta = run_preprocessing(cfg)
    logging.info(f"✓ Preprocesamiento completado para {agent_name}")
    logging.info(f"  - Features: {meta['n_features_transformed']}")
    logging.info(f"  - Train samples: {meta['train_samples']}")


# ============ Funciones Helper ============

def save_models_npz(models: Dict[str, np.ndarray], tag: str, agent_name: str = 'a1') -> str:
    """
    Guarda modelos como archivo .npz con timestamp.
    
    Args:
        models: Diccionario de arrays NumPy (parámetros del modelo)
        tag: Etiqueta descriptiva (ej: 'init', 'local_r1', 'global_r2')
        agent_name: Nombre del agente (para organizar archivos)
    
    Returns:
        str: Path del archivo guardado
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(_models_dir(agent_name), f"{ts}_{tag}.npz")
    
    # Limpiar nombres de keys (reemplazar / por _)
    clean = {k.replace("/", "_"): v for k, v in models.items()}
    np.savez_compressed(path, **clean)
    
    logging.info(f"[checkpoint] Modelo guardado: {path}")
    return path


def _metrics_csv_path(dataset_tag: str = "ncd_dataset1", agent_name: str = 'a1') -> str:
    """Path del CSV de métricas"""
    return os.path.join(_models_dir(agent_name), f"metrics_{dataset_tag}.csv")


def log_metrics_csv(round_idx: int, kind: str, val_acc: float, test_acc: float,
                    dataset_tag: str = "ncd_dataset1", agent_name: str = 'a1') -> None:
    """
    Registra métricas en CSV para análisis posterior.
    
    Args:
        round_idx: Número de ronda
        kind: 'local' o 'global'
        val_acc: Accuracy en validation set
        test_acc: Accuracy en test set
        dataset_tag: Tag del dataset (para múltiples experimentos)
        agent_name: Nombre del agente
    """
    csv_path = _metrics_csv_path(dataset_tag, agent_name)
    header = ["timestamp", "round", "kind", "val_acc", "test_acc"]
    row = [time.strftime("%Y-%m-%d %H:%M:%S"), round_idx, kind, val_acc, test_acc]
    
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


# ------------- Plotting Final (UNA IMAGEN) -------------

def plot_and_save_single_image(dataset_tag: str = "ncd_dataset1", save_dir: Optional[str] = None, agent_name: str = 'a1') -> None:
    """
    Genera una imagen PNG con 2 subplots mostrando la evolución del accuracy:
      - Subplot izquierdo: Val Accuracy (local vs global)
      - Subplot derecho: Test Accuracy (local vs global)
    
    Args:
        dataset_tag: Tag del dataset
        save_dir: Directorio de salida (default: _models_dir())
        agent_name: Nombre del agente
    """
    csv_path = _metrics_csv_path(dataset_tag, agent_name)
    if not os.path.exists(csv_path):
        logging.warning(f"No existe {csv_path} — no hay métricas para graficar.")
        return

    dfm = pd.read_csv(csv_path, parse_dates=["timestamp"], keep_default_na=True)
    if dfm.empty:
        logging.warning(f"{csv_path} está vacío — nada que graficar.")
        return

    dfm["round"] = dfm["round"].astype(int)
    dfm = dfm.sort_values("round")

    # Pivot tables para separar local vs global
    pivot_val = dfm.pivot_table(index="round", columns="kind", values="val_acc")
    pivot_test = dfm.pivot_table(index="round", columns="kind", values="test_acc")

    if save_dir is None:
        save_dir = _models_dir(agent_name)
    os.makedirs(save_dir, exist_ok=True)

    # Crear figura con 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    # VAL subplot (izquierda)
    ax = axes[0]
    if "local" in pivot_val.columns:
        ax.plot(pivot_val.index, pivot_val["local"], label="Local", marker="o", linewidth=2)
    if "global" in pivot_val.columns:
        ax.plot(pivot_val.index, pivot_val["global"], label="Global", marker="s", linewidth=2)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Validation Accuracy", fontsize=12)
    ax.set_title("Val Accuracy por Ronda", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # TEST subplot (derecha)
    ax = axes[1]
    if "local" in pivot_test.columns:
        ax.plot(pivot_test.index, pivot_test["local"], label="Local", marker="o", linewidth=2)
    if "global" in pivot_test.columns:
        ax.plot(pivot_test.index, pivot_test["global"], label="Global", marker="s", linewidth=2)
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Test Accuracy por Ronda", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    
    # Guardar imagen
    out_path = os.path.join(save_dir, f"results_{dataset_tag}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logging.info(f"[plot] Imagen guardada: {out_path}")


# ============ Metadatos y Funciones de Entrenamiento ============

class TrainingMetaData:
    """Metadata del entrenamiento (usado para FedAvg ponderado)"""
    num_training_data = 200  # Número de muestras de entrenamiento (ajustar según tu dataset)


def init_models(agent_name: str = 'a1') -> Dict[str, np.ndarray]:
    """
    Inicializa los modelos (estructura sin entrenar).
    
    Args:
        agent_name: Nombre del agente (para cargar datos correctos)
    
    Returns:
        Dict[str, np.ndarray]: Diccionario con parámetros del modelo
    """
    dm = DataManager.dm(agent_name=agent_name)
    in_dim = dm.input_dim
    
    # Configurar converter con constructor del modelo
    conv = Converter.cvtr()
    conv.set_model_ctor(lambda: MLP(in_features=in_dim))
    
    # Crear modelo nuevo
    net = MLP(in_features=in_dim)
    
    return conv.convert_nn_to_dict_nparray(net)


def training(models: Dict[str, np.ndarray], init_flag: bool = False, agent_name: str = 'a1') -> Dict[str, np.ndarray]:
    """
    Función de entrenamiento local.
    
    Args:
        models: Diccionario con parámetros del modelo (NumPy arrays)
        init_flag: True si es inicialización, False si es entrenamiento real
        agent_name: Nombre del agente (para cargar datos correctos)
    
    Returns:
        Dict[str, np.ndarray]: Modelos entrenados (NumPy arrays)
    """
    conv = Converter.cvtr()
    
    if init_flag:
        # Inicialización: preparar DataManager y retornar modelos sin entrenar
        DataManager.dm(th=int(TrainingMetaData.num_training_data / 4), agent_name=agent_name)
        return init_models(agent_name=agent_name)

    logging.info("--- Entrenamiento local iniciando ---")
    
    # Convertir NumPy dict → PyTorch model
    net = conv.convert_dict_nparray_to_nn(models)
    
    # Definir loss y optimizer
    criterion = nn.BCEWithLogitsLoss()  # Para clasificación binaria con logits
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    
    # Entrenar
    dm = DataManager.dm(agent_name=agent_name)
    trained_net = execute_tabular_training(dm, net, criterion, optimizer, epochs=10)
    
    logging.info("--- Entrenamiento local completado ---")
    
    # Convertir PyTorch model → NumPy dict
    return conv.convert_nn_to_dict_nparray(trained_net)


def _eval_split(models: Dict[str, np.ndarray], split: str, agent_name: str = 'a1') -> float:
    """
    Evalúa el modelo en un split específico (val o test).
    
    Args:
        models: Diccionario con parámetros del modelo
        split: 'val' o 'test'
        agent_name: Nombre del agente (para cargar datos correctos)
    
    Returns:
        float: Accuracy en el split
    """
    conv = Converter.cvtr()
    net = conv.convert_dict_nparray_to_nn(models)
    net.eval()
    
    dm = DataManager.dm(agent_name=agent_name)
    loader = dm.valloader if split == "val" else dm.testloader
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = net(X_batch)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().squeeze()
            
            correct += (preds == y_batch.long()).sum().item()
            total += y_batch.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy


# -------------------------- Main Loop --------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("=" * 60)
    logging.info("Iniciando Cliente FL para Clasificación Tabular NCD")
    logging.info("=" * 60)

    # Crear cliente FL (lee argumentos de línea de comando automáticamente)
    fl_client = Client()
    agent_name = _get_agent_name_from_client(fl_client)
    
    logging.info(f"Nombre del agente: {agent_name}")
    logging.info(f"IP local del agente: {fl_client.agent_ip}")
    logging.info(f"Archivo de datos: {_map_agent_to_data_file(agent_name)}")
    
    # Verificar y preprocesar datos si es necesario
    ensure_data_preprocessed(agent_name)

    # Inicializar Judge para early stopping
    # Ajustar max_rounds, patience, min_delta según necesidades
    judge = Judge(max_rounds=50, patience=5, min_delta=1e-4)

    # Preparar modelo inicial
    logging.info("\n--- Inicializando modelos ---")
    initial_models = training(dict(), init_flag=True, agent_name=agent_name)
    save_models_npz(initial_models, "init", agent_name)
    
    # Enviar modelo inicial al agregador
    fl_client.send_initial_model(initial_models)

    # Iniciar threads del cliente FL
    fl_client.start_fl_client()

    # Contadores
    training_count = 0
    gm_arrival_count = 0
    dataset_tag = f"{DATASET_TAG}_{agent_name}"

    logging.info("\n" + "=" * 60)
    logging.info("Iniciando Loop Principal de Federated Learning")
    logging.info("=" * 60)

    while True:
        # ========== 1. Esperar modelo global ==========
        logging.info(f"\n[Round {gm_arrival_count + 1}] Esperando modelo global...")
        global_models = fl_client.wait_for_global_model()
        gm_arrival_count += 1
        
        logging.info(f"✓ Modelo global recibido (Round {gm_arrival_count})")
        save_models_npz(global_models, f"global_r{gm_arrival_count}", agent_name)

        # ========== 2. Evaluar modelo global ==========
        logging.info("Evaluando modelo global...")
        acc_val_g = _eval_split(global_models, "val", agent_name)
        acc_tst_g = _eval_split(global_models, "test", agent_name)
        
        print(f"[Global Model] Val Acc: {100*acc_val_g:.2f}% | Test Acc: {100*acc_tst_g:.2f}%")
        log_metrics_csv(gm_arrival_count, "global", acc_val_g, acc_tst_g, dataset_tag, agent_name)

        # ========== 3. Verificar criterio de parada ==========
        should_continue = judge.update_and_should_continue(gm_arrival_count, val_acc=acc_val_g)
        if not should_continue:
            logging.info("\n" + "=" * 60)
            logging.info("✓ Criterio de parada satisfecho (Judge)")
            logging.info("=" * 60)
            break

        # ========== 4. Entrenamiento local ==========
        logging.info("\n--- Iniciando entrenamiento local ---")
        models = training(global_models, agent_name=agent_name)
        training_count += 1
        save_models_npz(models, f"local_r{training_count}", agent_name)

        # ========== 5. Evaluar modelo local ==========
        logging.info("Evaluando modelo local...")
        acc_val_l = _eval_split(models, "val", agent_name)
        acc_tst_l = _eval_split(models, "test", agent_name)
        
        print(f"[Local Model]  Val Acc: {100*acc_val_l:.2f}% | Test Acc: {100*acc_tst_l:.2f}%")
        log_metrics_csv(training_count, "local", acc_val_l, acc_tst_l, dataset_tag, agent_name)

        # ========== 6. Enviar modelo entrenado ==========
        logging.info("Enviando modelo local al agregador...")
        fl_client.send_trained_model(
            models, 
            int(TrainingMetaData.num_training_data), 
            acc_val_l
        )
        logging.info("✓ Modelo local enviado")

    # ========== Finalización ==========
    logging.info("\n" + "=" * 60)
    logging.info("Generando gráficas finales...")
    logging.info("=" * 60)
    
    # Generar plot final con todas las métricas
    plot_and_save_single_image(dataset_tag=dataset_tag, agent_name=agent_name)
    
    logging.info("\n" + "=" * 60)
    logging.info("✓ ENTRENAMIENTO FEDERADO COMPLETADO")
    logging.info("=" * 60)
    logging.info(f"Total de rondas globales: {gm_arrival_count}")
    logging.info(f"Total de entrenamientos locales: {training_count}")
    logging.info(f"Métricas guardadas en: {_metrics_csv_path(dataset_tag, agent_name)}")
    logging.info(f"Modelos guardados en: {_models_dir(agent_name)}")
