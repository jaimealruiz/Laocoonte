import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Cargar el dataset preprocesado
file_path_preprocessed = "/mnt/data/DNN_Part0_preprocessed.csv"
df_processed = pd.read_csv(file_path_preprocessed)

# Verificar si hay valores no numéricos
non_numeric_cols = df_processed.select_dtypes(exclude=[np.number]).columns

# Si existen columnas no numéricas, eliminarlas antes del análisis
df_processed = df_processed.drop(columns=non_numeric_cols, errors='ignore')

# Calcular la matriz de correlación con las columnas preprocesadas
correlation_matrix_processed = df_processed.corr()

# Identificar columnas altamente correlacionadas (umbral > 0.9)
highly_correlated_features_processed = set()
correlation_threshold = 0.9

for i in range(len(correlation_matrix_processed.columns)):
    for j in range(i):
        if abs(correlation_matrix_processed.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix_processed.columns[i]
            highly_correlated_features_processed.add(colname)

# Aplicar PCA al dataset preprocesado
pca_processed = PCA(n_components=min(df_processed.shape[0], df_processed.shape[1]))
df_pca_processed = pca_processed.fit_transform(df_processed)

# Variabilidad explicada por los primeros componentes
explained_variance_ratio_processed = np.cumsum(pca_processed.explained_variance_ratio_)

# Mostrar características altamente correlacionadas y variabilidad explicada
highly_correlated_features_processed, explained_variance_ratio_processed
