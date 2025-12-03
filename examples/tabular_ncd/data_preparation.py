"""
Data Preparation for Tabular NCD Federated Learning.

Este módulo prepara los datos tabulares de defunciones para entrenamiento federado.
Cada nodo (hospital/Raspberry Pi) tiene su propio CSV con datos locales.

Pasos:
 1. Cargar datos crudos del hospital (data1.csv, data2.csv, etc.)
 2. Cargar preprocessor global compartido (preprocessor_global.joblib)
 3. Transformar features a matriz numérica usando OneHotEncoder/StandardScaler
 4. Dividir en train/val/test con estratificación
 5. Guardar CSVs procesados

Estructura esperada:
  examples/tabular_ncd/
    data/
      data1.csv, data2.csv, data3.csv, data4.csv  <- datos crudos por nodo
      preprocessor_global.joblib                   <- preprocessor compartido
      processed/                                   <- salida de este script
        train.csv, val.csv, test.csv
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import logging

def get_default_config(base_dir: str, data_file: str = "data1.csv") -> dict:
    """
    Genera configuración por defecto para el preprocesamiento.
    
    Args:
        base_dir: Directorio base del módulo (ej: examples/tabular_ncd/)
        data_file: Nombre del archivo CSV a procesar (data1.csv, data2.csv, etc.)
    
    Returns:
        dict: Configuración con todas las rutas y parámetros
    """
    # Asegurar que base_dir sea absoluto
    if not os.path.isabs(base_dir):
        base_dir = os.path.abspath(base_dir)
    
    # Directorio de datos: examples/tabular_ncd/data/
    data_dir = os.path.join(base_dir, "data")
    
    return {
        "raw_data_path": os.path.join(data_dir, data_file),
        "preprocessor_path": os.path.join(data_dir, "preprocessor_global.joblib"),
        "target_col": "is_premature_ncd",
        "train_frac": 0.7,
        "val_frac": 0.15,
        "test_frac": 0.15,
        "random_state": 42,
        "output_dir": os.path.join(data_dir, "processed"),
        "rename_target_to": "target",
        "drop_cols": ["hospital_cliente"],  # columnas a eliminar antes de transformar
        "balance_strategy": "none"  # 'none' o 'undersample_majority'
    }


def simple_undersample(X: np.ndarray, y: np.ndarray, random_state: int):
    """
    Undersampling simple de la clase mayoritaria para balancear clases.
    
    Args:
        X: Features
        y: Labels binarios
        random_state: Semilla para reproducibilidad
    
    Returns:
        tuple: (X_balanced, y_balanced)
    """
    rng = np.random.default_rng(random_state)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y
    if len(pos_idx) == len(neg_idx):
        return X, y
    
    # Tomar muestras de la clase mayoritaria igual a la minoritaria
    if len(pos_idx) < len(neg_idx):
        keep_neg = rng.choice(neg_idx, size=len(pos_idx), replace=False)
        sel = np.concatenate([pos_idx, keep_neg])
    else:
        keep_pos = rng.choice(pos_idx, size=len(neg_idx), replace=False)
        sel = np.concatenate([keep_pos, neg_idx])
    
    sel = rng.permutation(sel)
    return X[sel], y[sel]


def run_preprocessing(cfg: dict) -> dict:
    """
    Ejecuta el preprocesamiento completo con la configuración dada.
    
    Args:
        cfg: Diccionario de configuración de get_default_config()
    
    Returns:
        dict: Metadata del proceso (número de features, samples, etc.)
    """
    raw_path = cfg['raw_data_path']
    preproc_path = cfg['preprocessor_path']
    target_col = cfg['target_col']
    out_dir = cfg.get('output_dir', './data/processed')
    os.makedirs(out_dir, exist_ok=True)
    
    rename_target = cfg.get('rename_target_to', 'target')
    drop_cols = cfg.get('drop_cols', [])
    balance_strategy = cfg.get('balance_strategy', 'none')

    train_frac = float(cfg['train_frac'])
    val_frac = float(cfg['val_frac'])
    test_frac = float(cfg['test_frac'])
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, 'Fractions must sum to 1.'
    rnd = int(cfg.get('random_state', 42))

    # 1) Cargar datos crudos del nodo
    logging.info(f"Cargando datos desde: {raw_path}")
    df = pd.read_csv(raw_path)
    
    if target_col not in df.columns:
        raise ValueError(f'Target column {target_col} not found. Columns: {list(df.columns)}')

    # Eliminar columnas no deseadas (ej: hospital_cliente)
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=c)
            logging.info(f"Columna eliminada: {c}")

    # Extraer target (is_premature_ncd)
    y = df[target_col].astype(float).to_numpy()
    X_raw = df.drop(columns=[target_col])

    # 2) Cargar preprocessor global (compartido entre todos los nodos)
    logging.info(f"Cargando preprocessor desde: {preproc_path}")
    if not os.path.exists(preproc_path):
        raise FileNotFoundError(f"Preprocessor no encontrado: {preproc_path}")
    
    preproc = joblib.load(preproc_path)

    # 3) Transformar features usando el preprocessor
    # OneHotEncoder para categóricas + StandardScaler para numéricas
    X_mat = preproc.transform(X_raw)
    n_features = X_mat.shape[1]
    feature_cols = [f'f{i}' for i in range(n_features)]
    logging.info(f"Features transformadas: {n_features} columnas")
    logging.info(f"Distribución de clases original: {np.bincount(y.astype(int))}")

    # 4) Balanceo opcional de clases
    if balance_strategy == 'undersample_majority':
        X_mat, y = simple_undersample(X_mat, y, rnd)
        logging.info(f"Undersampling aplicado. Nuevas muestras: {len(y)}")
        logging.info(f"Distribución después de balanceo: {np.bincount(y.astype(int))}")

    # 5) Splits estratificados (mantiene proporción de clases)
    temp_frac = val_frac + test_frac
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_mat, y, test_size=temp_frac, random_state=rnd,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    if temp_frac > 0:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(test_frac / temp_frac), random_state=rnd,
            stratify=y_temp if len(np.unique(y_temp)) > 1 else None
        )
    else:
        X_val, X_test = np.empty((0, n_features)), np.empty((0, n_features))
        y_val, y_test = np.empty((0,)), np.empty((0,))

    def build_df(Xp: np.ndarray, yp: np.ndarray) -> pd.DataFrame:
        """Construir DataFrame con features + target"""
        d = pd.DataFrame(Xp, columns=feature_cols)
        d[rename_target] = yp.astype(int)
        return d

    # 6) Guardar CSVs procesados
    train_df = build_df(X_train, y_train)
    val_df = build_df(X_val, y_val)
    test_df = build_df(X_test, y_test)

    train_df.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(out_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(out_dir, 'test.csv'), index=False)

    logging.info(f"CSVs guardados en: {out_dir}")
    logging.info(f"  - train.csv: {len(train_df)} samples")
    logging.info(f"  - val.csv: {len(val_df)} samples")
    logging.info(f"  - test.csv: {len(test_df)} samples")

    return {
        'n_features_transformed': n_features,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'output_dir': out_dir,
        'class_distribution': np.bincount(y_train.astype(int)).tolist()
    }


def preprocess_split_dfs(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Función auxiliar para compatibilidad con el código existente.
    Simplemente retorna los DataFrames ya procesados.
    
    Args:
        train_df, val_df, test_df: DataFrames ya procesados
    
    Returns:
        tuple: (train_df, val_df, test_df, feature_cols)
    """
    feature_cols = [c for c in train_df.columns if c != 'target']
    return train_df, val_df, test_df, feature_cols


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test con data1.csv (Raspberry Pi 1)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = get_default_config(base_dir, data_file="data1.csv")
    
    print("=" * 60)
    print("Configuración generada para Raspberry Pi 1:")
    print("=" * 60)
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    
    if os.path.exists(cfg['raw_data_path']) and os.path.exists(cfg['preprocessor_path']):
        print("\n" + "=" * 60)
        print("Ejecutando preprocesamiento...")
        print("=" * 60)
        meta = run_preprocessing(cfg)
        print("\nMetadata del resultado:")
        for k, v in meta.items():
            print(f"  {k}: {v}")
    else:
        print("\n⚠️ Archivos no encontrados. Verifica las rutas:")
        print(f"   - Raw data: {cfg['raw_data_path']}")
        print(f"   - Preprocessor: {cfg['preprocessor_path']}")
        print("\nPara crear el preprocessor, ejecuta:")
        print("   python -m examples.tabular_ncd.create_preprocessor")
