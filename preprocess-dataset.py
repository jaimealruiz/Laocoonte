import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Dataset path
path = "./datasets/Wednesday-workingHours.pcap_ISCX.csv"

# Cargar dataset
df = pd.read_csv(path)

# Comprobar las columnas disponibles
print("Columnas disponibles:", df.columns.tolist())

# Nombre exacto de la columna de etiquetas (ajustado según salida anterior)
label_col = ' Label'  # probablemente con espacio inicial

if label_col in df.columns:
    labels = df[label_col]
    benign_mask = labels == 'BENIGN'
else:
    raise ValueError("La columna de etiquetas no existe en el dataset.")

# Columnas irrelevantes incluyendo la etiqueta
cols_to_drop = ['Flow ID', 'Source IP', 'Source Port', 
                'Destination IP', 'Destination Port', 
                'Timestamp', label_col]

for col in cols_to_drop:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

print("Características restantes tras filtrado:", df.columns.tolist())
print("Dimensiones tras filtrado de columnas:", df.shape)

# Separar datos normales y ataques
normal_data = df[benign_mask]
attack_data = df[~benign_mask]

print(f"Muestras benignas: {normal_data.shape[0]}, Muestras de ataque: {attack_data.shape[0]}")

# División entrenamiento/validación/prueba (70%-15%-15%)
X_train_norm, X_temp_norm = train_test_split(normal_data, test_size=0.30, random_state=42)
X_val_norm, X_test_norm = train_test_split(X_temp_norm, test_size=0.50, random_state=42)

# Combinar conjunto de prueba con ataques
X_test_full = pd.concat([X_test_norm, attack_data])
y_test_full = np.concatenate([
    np.zeros(len(X_test_norm)),  # benignos
    np.ones(len(attack_data))    # ataques
])

print("Tamaño entrenamiento (solo benignos):", X_train_norm.shape)
print("Tamaño validación (solo benignos):", X_val_norm.shape)
print("Tamaño prueba (benignos + ataques):", X_test_full.shape, 
      "| Ataques en prueba:", (y_test_full == 1).sum())

# Asegurar columnas numéricas
X_train_norm = X_train_norm.select_dtypes(include=["number"])
X_val_norm = X_val_norm.select_dtypes(include=[np.number])
X_test_full = X_test_full.select_dtypes(include=[np.number])

# Reemplazar infinitos por NaN y eliminarlos
X_train_norm.replace([np.inf, -np.inf], np.nan, inplace=True)
X_val_norm.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_full.replace([np.inf, -np.inf], np.nan, inplace=True)

# Eliminar filas con NaNs
X_train_norm.dropna(inplace=True)
X_val_norm.dropna(inplace=True)
X_test_full.dropna(inplace=True)

# Reiniciar índices
X_train_norm.reset_index(drop=True, inplace=True)
X_val_norm.reset_index(drop=True, inplace=True)
X_test_full.reset_index(drop=True, inplace=True)

# Reconstruir vector y_test_full tras eliminar NaNs
y_test_full = np.concatenate([
    np.zeros(len(X_test_full) - len(attack_data)),
    np.ones(len(attack_data))
])

# Escalado MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train_norm)
X_val_scaled   = scaler.transform(X_val_norm)
X_test_scaled  = scaler.transform(X_test_full)

# Convertir a float32
X_train_scaled = X_train_scaled.astype('float32')
X_val_scaled   = X_val_scaled.astype('float32')
X_test_scaled  = X_test_scaled.astype('float32')

# Añadir dimensión para conv1D
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_val_scaled   = X_val_scaled.reshape((X_val_scaled.shape[0], X_val_scaled.shape[1], 1))
X_test_scaled  = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

print("Shape final de X_train:", X_train_scaled.shape)

# Guardar arrays procesados
np.save("./datasets/processed/X_train.npy", X_train_scaled)
np.save("./datasets/processed/X_val.npy", X_val_scaled)
np.save("./datasets/processed/X_test.npy", X_test_scaled)
np.save("./datasets/processed/y_test.npy", y_test_full.astype('int'))


# ========================= #
#           JAR             #
# ========================= #