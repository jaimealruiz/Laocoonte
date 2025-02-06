from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os

# Lista de archivos a analizar
file_paths = [
    "./Datasets/v0/NormalTraffic_Training_dataset.csv",
]

# Diccionario para almacenar las columnas más importantes en cada parte
important_features_with_categorical = {}

for file_path in file_paths:
    try:
        # Cargar el dataset
        df_part = pd.read_csv(file_path, low_memory=False)
        
        # Identificar columnas categóricas
        categorical_cols = df_part.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Convertir columnas categóricas a numéricas usando Label Encoding
        for col in categorical_cols:
            df_part[col] = LabelEncoder().fit_transform(df_part[col].astype(str))
        
        # Seleccionar todas las columnas ahora numéricas
        all_numeric_cols = df_part.select_dtypes(include=[np.number]).columns.tolist()
        
        # Aplicar PCA con todas las características
        pca = PCA(n_components=15)
        df_pca = pca.fit_transform(df_part[all_numeric_cols])
        
        # Obtener la importancia de cada característica
        feature_importance = np.abs(pca.components_).sum(axis=0)
        
        # Ordenar las características por importancia
        important_features_idx = np.argsort(feature_importance)[-15:][::-1]
        important_features = [all_numeric_cols[i] for i in important_features_idx]
        
        # Almacenar los resultados
        important_features_with_categorical[os.path.basename(file_path)] = important_features
    
    except Exception as e:
        important_features_with_categorical[os.path.basename(file_path)] = f"Error: {str(e)}"

# Mostrar los resultados para cada archivo
important_features_with_categorical
print(important_features_with_categorical)
