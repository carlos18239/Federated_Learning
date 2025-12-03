"""
Script para crear el preprocessor global compartido.

Este script crea el preprocessor_global.joblib a partir de los datos
en examples/tabular_ncd/data/ (data1.csv a data4.csv).

El preprocessor incluye:
  - OneHotEncoder para columnas categóricas
  - StandardScaler para columnas numéricas

Este archivo se usa para transformar los datos de cada agente de forma consistente.
"""
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import logging


def identify_column_types(df: pd.DataFrame, target_col: str, drop_cols: list):
    """
    Identifica columnas categóricas y numéricas automáticamente.
    
    Args:
        df: DataFrame con los datos
        target_col: Nombre de la columna target (excluir)
        drop_cols: Lista de columnas a excluir
    
    Returns:
        tuple: (categorical_cols, numerical_cols)
    """
    # Excluir target y columnas a dropear
    exclude_cols = [target_col] + drop_cols
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    categorical_cols = []
    numerical_cols = []
    
    for col in feature_cols:
        # Si es tipo object/string o tiene pocos valores únicos, es categórica
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_cols.append(col)
        elif df[col].nunique() <= 10:  # Heurística: ≤10 valores únicos = categórica
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
    
    return categorical_cols, numerical_cols


def create_preprocessor(categorical_cols: list, numerical_cols: list):
    """
    Crea el ColumnTransformer con OneHotEncoder y StandardScaler.
    
    Args:
        categorical_cols: Lista de columnas categóricas
        numerical_cols: Lista de columnas numéricas
    
    Returns:
        ColumnTransformer: Preprocessor listo para fit
    """
    transformers = []
    
    # OneHotEncoder para categóricas
    if categorical_cols:
        transformers.append((
            'cat',
            OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
            categorical_cols
        ))
    
    # StandardScaler para numéricas
    if numerical_cols:
        transformers.append((
            'num',
            StandardScaler(),
            numerical_cols
        ))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop columnas no especificadas
    )
    
    return preprocessor


def create_global_preprocessor(
    target_col: str = "is_premature_ncd",
    drop_cols: list = None
):
    """
    Crea el preprocessor global a partir de todos los datasets en data/.
    
    Args:
        target_col: Nombre de la columna target
        drop_cols: Columnas a eliminar antes del preprocessing
    
    Returns:
        ColumnTransformer: Preprocessor ajustado (fitted)
    """
    if drop_cols is None:
        drop_cols = ["hospital_cliente"]
    
    # Rutas
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    output_path = os.path.join(data_dir, "preprocessor_global.joblib")
    
    logging.info("=" * 60)
    logging.info("Creando Preprocessor Global")
    logging.info("=" * 60)
    
    # 1. Cargar todos los datasets
    logging.info(f"\n1. Cargando datos desde: {data_dir}")
    all_dfs = []
    
    for i in range(1, 5):  # data1.csv a data4.csv
        csv_path = os.path.join(data_dir, f"data{i}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            logging.info(f"   ✓ data{i}.csv cargado: {len(df)} filas")
            all_dfs.append(df)
        else:
            logging.warning(f"   ⚠ data{i}.csv no encontrado, saltando...")
    
    if not all_dfs:
        raise FileNotFoundError(f"No se encontraron archivos data*.csv en {data_dir}")
    
    # Concatenar todos los datos
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logging.info(f"\n   Total combinado: {len(combined_df)} filas")
    
    # 2. Identificar tipos de columnas
    logging.info("\n2. Identificando tipos de columnas...")
    
    # Eliminar columnas que no se usarán
    for col in drop_cols:
        if col in combined_df.columns:
            combined_df = combined_df.drop(columns=col)
            logging.info(f"   - Eliminada: {col}")
    
    categorical_cols, numerical_cols = identify_column_types(
        combined_df, 
        target_col, 
        drop_cols
    )
    
    logging.info(f"\n   Columnas categóricas ({len(categorical_cols)}):")
    for col in categorical_cols:
        n_unique = combined_df[col].nunique()
        logging.info(f"     - {col}: {n_unique} valores únicos")
    
    logging.info(f"\n   Columnas numéricas ({len(numerical_cols)}):")
    for col in numerical_cols:
        min_val = combined_df[col].min()
        max_val = combined_df[col].max()
        logging.info(f"     - {col}: rango [{min_val:.2f}, {max_val:.2f}]")
    
    # 3. Crear preprocessor
    logging.info("\n3. Creando ColumnTransformer...")
    preprocessor = create_preprocessor(categorical_cols, numerical_cols)
    
    # 4. Ajustar preprocessor con todos los datos
    logging.info("\n4. Ajustando (fit) preprocessor con datos combinados...")
    
    # Separar features
    X = combined_df.drop(columns=[target_col])
    
    # Fit
    preprocessor.fit(X)
    logging.info("   ✓ Preprocessor ajustado correctamente")
    
    # Verificar dimensiones de salida
    X_transformed = preprocessor.transform(X)
    logging.info(f"\n   Dimensiones: {X.shape[1]} features → {X_transformed.shape[1]} features")
    
    # 5. Guardar preprocessor
    logging.info(f"\n5. Guardando preprocessor en: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(preprocessor, output_path)
    logging.info("   ✓ Preprocessor guardado exitosamente")
    
    logging.info("\n" + "=" * 60)
    logging.info("✓ Preprocessor Global Creado")
    logging.info("=" * 60)
    logging.info(f"\nArchivo: {output_path}")
    logging.info(f"Ahora puedes ejecutar: python -m examples.tabular_ncd.binary_classification")
    
    return preprocessor


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    print("\n" + "=" * 60)
    print("Script: Crear Preprocessor Global para FL")
    print("=" * 60)
    print("\nEste script creará preprocessor_global.joblib")
    print("usando los datos en examples/tabular_ncd/data/")
    print("")
    
    try:
        preprocessor = create_global_preprocessor(
            target_col="is_premature_ncd",
            drop_cols=["hospital_cliente"]
        )
        
        print("\n✅ PROCESO COMPLETADO")
        print("\n📋 Próximos pasos:")
        print("1. Iniciar DB: python -m fl_main.pseudodb.pseudo_db")
        print("2. Iniciar agregador: python -m fl_main.aggregator.server_th")
        print("3. Ejecutar agentes:")
        print("   python -m examples.tabular_ncd.binary_classification 1 50001 a1")
        print("   python -m examples.tabular_ncd.binary_classification 1 50002 a2")
        print("   python -m examples.tabular_ncd.binary_classification 1 50003 a3")
        print("   python -m examples.tabular_ncd.binary_classification 1 50004 a4")
        
    except Exception as e:
        logging.error(f"\n❌ ERROR: {e}")
        logging.error("\nVerifica que:")
        logging.error(f"  - Existan archivos data1.csv-data4.csv en: examples/tabular_ncd/data/")
        logging.error(f"  - Los archivos tengan la columna 'is_premature_ncd'")
        raise
