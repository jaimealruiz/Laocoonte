import pandas as pd

# Cargar el CSV grande
input_file = "./Datasets/v0/NormalTraffic_Training_dataset_preprocessed.csv"
chunksize = 100000  # Número de filas por archivo

# Leer el CSV en partes y guardarlas
for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunksize)):
    chunk.to_csv(f"DNN_Part{i}.csv", index=False)
