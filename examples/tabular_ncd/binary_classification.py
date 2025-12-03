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


def log_metrics_csv(round_idx: int, kind: str, 
                    global_acc: float, local_acc: float,
                    global_recall: float, local_recall: float,
                    num_messages: int, bytes_global: int, bytes_local: int,
                    bytes_round_total: int, bytes_cumulative: int,
                    latency_wait_global: float, round_time: float,
                    dataset_tag: str = "ncd_dataset1", agent_name: str = 'a1') -> None:
    """
    Registra métricas completas en CSV para análisis posterior.
    
    Args:
        round_idx: Número de ronda
        kind: 'global' o 'local'
        global_acc: Accuracy global
        local_acc: Accuracy local
        global_recall: Recall global calculado
        local_recall: Recall local
        num_messages: Número de mensajes intercambiados
        bytes_global: Bytes del modelo global
        bytes_local: Bytes del modelo local
        bytes_round_total: Total de bytes en la ronda
        bytes_cumulative: Bytes acumulados
        latency_wait_global: Latencia esperando modelo global
        round_time: Tiempo total de la ronda
        dataset_tag: Tag del dataset
        agent_name: Nombre del agente
    """
    csv_path = _metrics_csv_path(dataset_tag, agent_name)
    header = ["timestamp", "round", "global_accuracy", "local_accuracy", 
              "global_recall", "local_recall",
              "num_messages", "bytes_global", "bytes_local", 
              "bytes_round_total", "bytes_cumulative",
              "latency_wait_global", "round_time"]
    
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]  # ISO format con milisegundos
    row = [timestamp, round_idx, global_acc, local_acc, 
           global_recall, local_recall,
           num_messages, bytes_global, bytes_local,
           bytes_round_total, bytes_cumulative,
           latency_wait_global, round_time]
    
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


def _eval_recall(models: Dict[str, np.ndarray], split: str, agent_name: str = 'a1') -> float:
    """
    Calcula el recall del modelo en un split específico.
    Recall = TP / (TP + FN) - útil para detectar casos positivos.
    
    Args:
        models: Diccionario con parámetros del modelo
        split: 'val' o 'test'
        agent_name: Nombre del agente (para cargar datos correctos)
    
    Returns:
        float: Recall en el split
    """
    conv = Converter.cvtr()
    net = conv.convert_dict_nparray_to_nn(models)
    net.eval()
    
    dm = DataManager.dm(agent_name=agent_name)
    loader = dm.valloader if split == "val" else dm.testloader
    
    true_positives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = net(X_batch)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().squeeze()
            y_true = y_batch.long()
            
            # Calcular TP y FN
            true_positives += ((preds == 1) & (y_true == 1)).sum().item()
            false_negatives += ((preds == 0) & (y_true == 1)).sum().item()
    
    # Recall = TP / (TP + FN)
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    return recall


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

    # Inicializar Judge para early stopping basado en recall
    # - max_rounds=50: Límite absoluto de 50 rondas
    # - patience=20: Para si no mejora recall en 20 rondas consecutivas
    judge = Judge(max_rounds=50, patience=20, min_delta=1e-4)

    # Preparar modelo inicial
    logging.info("\n--- Inicializando modelos ---")
    initial_models = training(dict(), init_flag=True, agent_name=agent_name)
    save_models_npz(initial_models, "init", agent_name)
    
    # Enviar modelo inicial al agregador
    fl_client.send_initial_model(initial_models)

    # Iniciar threads del cliente FL
    fl_client.start_fl_client()

    # Contadores y métricas de comunicación
    training_count = 0
    gm_arrival_count = 0
    dataset_tag = f"{DATASET_TAG}_{agent_name}"
    bytes_cumulative = 0  # Bytes acumulados a lo largo de todas las rondas
    
    # Métricas de recall para agregación
    local_recalls = []  # Lista de recalls locales de todos los agentes

    logging.info("\n" + "=" * 60)
    logging.info("Iniciando Loop Principal de Federated Learning")
    logging.info("=" * 60)

    while True:
        round_start_time = time.time()  # Inicio de la ronda
        
        # ========== 1. Esperar modelo global ==========
        logging.info(f"\n[Round {gm_arrival_count + 1}] Esperando modelo global...")
        wait_start = time.time()
        global_models = fl_client.wait_for_global_model()
        latency_wait_global = time.time() - wait_start
        
        gm_arrival_count += 1
        
        logging.info(f"✓ Modelo global recibido (Round {gm_arrival_count}) - Latencia: {latency_wait_global:.4f}s")
        save_models_npz(global_models, f"global_r{gm_arrival_count}", agent_name)
        
        # Calcular tamaño del modelo global en bytes
        import pickle
        bytes_global = len(pickle.dumps(global_models))

        # ========== 2. Evaluar modelo global ==========
        logging.info("Evaluando modelo global (accuracy + recall)...")
        acc_global = _eval_split(global_models, "test", agent_name)
        recall_global_local = _eval_recall(global_models, "test", agent_name)  # Recall calculado localmente
        
        # TODO: En un sistema real, aquí se recibirían los recalls de todos los agentes
        # y se calcularía el recall global promedio. Por ahora usamos el recall local.
        global_recall = recall_global_local  # Simplificación: recall global = recall local
        
        print(f"[Global Model] Accuracy: {100*acc_global:.2f}% | Recall: {100*global_recall:.2f}%")

        # ========== 3. Verificar criterio de parada (basado en recall global) ==========
        should_continue = judge.update_and_should_continue(gm_arrival_count, global_recall=global_recall)
        if not should_continue:
            logging.info("\n" + "=" * 60)
            logging.info("✓ Criterio de parada satisfecho (Judge)")
            logging.info("=" * 60)
            
            # Registrar última ronda antes de salir
            round_time = time.time() - round_start_time
            bytes_round_total = bytes_global
            bytes_cumulative += bytes_round_total
            
            log_metrics_csv(
                round_idx=gm_arrival_count,
                kind="final_global",
                global_acc=acc_global,
                local_acc=0.0,  # No hay modelo local en la última iteración
                global_recall=global_recall,
                local_recall=0.0,
                num_messages=1,  # Solo recepción del modelo global
                bytes_global=bytes_global,
                bytes_local=0,
                bytes_round_total=bytes_round_total,
                bytes_cumulative=bytes_cumulative,
                latency_wait_global=latency_wait_global,
                round_time=round_time,
                dataset_tag=dataset_tag,
                agent_name=agent_name
            )
            break

        # ========== 4. Entrenamiento local ==========
        logging.info("\n--- Iniciando entrenamiento local ---")
        train_start = time.time()
        models = training(global_models, agent_name=agent_name)
        training_count += 1
        train_time = time.time() - train_start
        
        save_models_npz(models, f"local_r{training_count}", agent_name)
        logging.info(f"✓ Entrenamiento completado en {train_time:.2f}s")

        # ========== 5. Evaluar modelo local ==========
        logging.info("Evaluando modelo local (accuracy + recall)...")
        acc_local = _eval_split(models, "test", agent_name)
        recall_local = _eval_recall(models, "test", agent_name)
        
        print(f"[Local Model]  Accuracy: {100*acc_local:.2f}% | Recall: {100*recall_local:.2f}%")

        # ========== 6. Enviar modelo entrenado ==========
        logging.info("Enviando modelo local al agregador...")
        
        # Calcular tamaño del modelo local
        bytes_local = len(pickle.dumps(models))
        
        fl_client.send_trained_model(
            models, 
            int(TrainingMetaData.num_training_data), 
            recall_local  # Enviar recall local en lugar de accuracy
        )
        logging.info("✓ Modelo local enviado")
        
        # ========== 7. Calcular métricas de la ronda ==========
        round_time = time.time() - round_start_time
        
        # Métricas de comunicación
        num_messages = 2  # 1 recepción (global) + 1 envío (local)
        bytes_round_total = bytes_global + bytes_local
        bytes_cumulative += bytes_round_total
        
        logging.info(f"📊 Ronda {gm_arrival_count} completada en {round_time:.2f}s")
        logging.info(f"   - Bytes global: {bytes_global:,} | Bytes local: {bytes_local:,} | Total: {bytes_round_total:,}")
        logging.info(f"   - Bytes acumulados: {bytes_cumulative:,}")
        
        # ========== 8. Registrar métricas en CSV ==========
        log_metrics_csv(
            round_idx=gm_arrival_count,
            kind="training",
            global_acc=acc_global,
            local_acc=acc_local,
            global_recall=global_recall,
            local_recall=recall_local,
            num_messages=num_messages,
            bytes_global=bytes_global,
            bytes_local=bytes_local,
            bytes_round_total=bytes_round_total,
            bytes_cumulative=bytes_cumulative,
            latency_wait_global=latency_wait_global,
            round_time=round_time,
            dataset_tag=dataset_tag,
            agent_name=agent_name
        )

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
    logging.info(f"Total de bytes transferidos: {bytes_cumulative:,} ({bytes_cumulative/(1024*1024):.2f} MB)")
    logging.info(f"Métricas guardadas en: {_metrics_csv_path(dataset_tag, agent_name)}")
    logging.info(f"Modelos guardados en: {_models_dir(agent_name)}")
