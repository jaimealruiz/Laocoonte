import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from ipaddress import ip_address, AddressValueError
import re

# Cargar y procesar dataset en bloques
def preprocess_dataset(file_path, output_path, chunksize=100000):
    selected_columns = {'mbtcp.len', 'tcp.options', 'tcp.connection.synack', 'tcp.srcport', 'tcp.dstport',
                        'tcp.payload', 'http.tls_port', 'tcp.checksum', 'udp.port', 'tcp.ack_raw', 'frame.time_seconds',
                        'udp.stream', 'tcp.seq', 'tcp.ack', 'tcp.len', 'delta_time'}
    
    # Crear un archivo CSV vacío con solo los encabezados
    pd.DataFrame(columns=list(selected_columns)).to_csv(output_path, index=False)
    
    for chunk in pd.read_csv(file_path, low_memory=False, chunksize=chunksize):
        
        # Eliminar columnas irrelevantes
        chunk = chunk.drop(columns=['Attack_label', 'Attack_type'], errors='ignore')
        
        # Convertir direcciones IP en valores numéricos
        def convert_ip(ip):
            try:
                return int(ip_address(ip))
            except (ValueError, AddressValueError):
                return 0  # Asignar un valor predeterminado para valores inválidos
        
        for col in ['ip.src_host', 'ip.dst_host']:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype(str).apply(convert_ip)
        
        # Estandarizar frame.time y calcular delta_time
        if 'frame.time' in chunk.columns:
            chunk['frame.time'] = chunk['frame.time'].astype(str).str.strip()
            
            # Extraer solo la parte de HH:MM:SS.Microsegundos si tiene el formato correcto
            chunk['frame.time_extracted'] = chunk['frame.time'].apply(
                lambda x: x.split()[-1] if isinstance(x, str) and re.match(r'^\d{2}:\d{2}:\d{2}\.\d+$', x.split()[-1]) else None
            )
            
            # Filtrar valores incorrectos sin eliminar el bloque entero
            chunk = chunk.dropna(subset=['frame.time_extracted'])
            
            # Convertir a segundos flotantes
            if not chunk.empty:
                chunk['frame.time_seconds'] = chunk['frame.time_extracted'].apply(
                    lambda x: sum(float(t) * f for t, f in zip(x.split(':'), [3600, 60, 1])) if pd.notna(x) else None
                )
                
                # Normalizar frame.time_seconds y delta_time a rango [0,1] usando MinMaxScaler
                scaler = MinMaxScaler()
                chunk[['frame.time_seconds', 'delta_time']] = scaler.fit_transform(chunk[['frame.time_seconds', 'delta_time']])
                
            chunk = chunk.drop(columns=['frame.time_extracted', 'frame.time'], errors='ignore')
        
        # Conservar solo las 15 columnas más importantes
        chunk = chunk[list(selected_columns.intersection(chunk.columns))]
        
        # Si el bloque está vacío, saltarlo
        if chunk.empty:
            continue
        
        # Identificar columnas categóricas
        categorical_cols = chunk.select_dtypes(include=['object']).columns.tolist()
        
        # Aplicar Label Encoding a variables categóricas
        for col in categorical_cols:
            chunk[col] = LabelEncoder().fit_transform(chunk[col].astype(str))
        
        # Escalar valores numéricos (excepto frame.time_seconds y delta_time)
        scaler = StandardScaler()
        numeric_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['frame.time_seconds', 'delta_time']]
        
        # Verificar si hay suficientes filas para aplicar StandardScaler
        if len(chunk) > 0:
            chunk[numeric_cols] = scaler.fit_transform(chunk[numeric_cols])
        
        # Guardar el dataset procesado en un nuevo archivo CSV en modo append
        chunk.to_csv(output_path, mode='a', header=False, index=False)
    
    return output_path

# Uso del script
preprocess_dataset('DNN_Part0.csv', 'DNN_Part0_preprocessed.csv')
