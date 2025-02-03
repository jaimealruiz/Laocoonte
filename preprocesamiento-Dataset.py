import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Cargar el dataset asegurando que frame.time se lea como string
file_path = "DNN-EdgeIIoT-dataset.csv"
df = pd.read_csv(file_path, low_memory=False, dtype={"frame.time": str})

# Extraer solo la parte de HH:MM:SS.Microsegundos
df["frame.time_extracted"] = df["frame.time"].apply(lambda x: x.split()[-1] if isinstance(x, str) else None)

# Filtrar valores incorrectos
df = df[df["frame.time_extracted"].str.match(r"^\d{2}:\d{2}:\d{2}\.\d+$", na=False)]

# Convertir a segundos flotantes
df["frame.time_seconds"] = df["frame.time_extracted"].apply(
    lambda x: sum(float(t) * f for t, f in zip(x.split(":"), [3600, 60, 1])) if pd.notna(x) else None
)

# Calcular delta_time y estadísticas adicionales
df["delta_time"] = df["frame.time_seconds"].diff().fillna(0)
df["delta_time_max"] = df["delta_time"].rolling(window=100, min_periods=1).max()
df["delta_time_min"] = df["delta_time"].rolling(window=100, min_periods=1).min()

# Seleccionar solo las columnas necesarias
columns_to_keep = [
    "delta_time", "delta_time_max", "delta_time_min",  # Tiempos entre paquetes
    "tcp.len", "tcp.flags", "tcp.ack", "tcp.seq",  # Propiedades de tráfico
    "udp.port", "dns.qry.type", "mqtt.msgtype", "mbtcp.len"  # Protocolos específicos
]
df = df[columns_to_keep]

# Eliminar filas con valores NaN (por seguridad)
df = df.dropna()

# Normalizar los datos
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Guardar el dataset final
df.to_csv("DNN-EdgeIIoT-processed.csv", index=False)

# Verificar estructura final del dataset
print("Dataset final listo para el modelo VAE:")
print(df.head())
print("Número total de filas en el dataset procesado:", len(df))
