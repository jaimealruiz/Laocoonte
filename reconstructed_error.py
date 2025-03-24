import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Cargar el modelo entrenado
autoencoder = tf.keras.models.load_model("./models/autoencoder_model_v4.keras", compile=False)

# Cargar conjunto de test y sus etiquetas (disponibles)
X_test = np.load("./datasets/processed/X_test.npy")
y_test = np.load("./datasets/processed/y_test.npy")

# Reconstrucciones usando el modelo
reconstructions = autoencoder.predict(X_test)

# Calcular error absoluto medio (MAE) por muestra
reconstruction_errors = np.mean(np.abs(reconstructions - X_test), axis=(1,2))

# Separar errores por clase
errors_normal = reconstruction_errors[y_test == 0]
errors_anomaly = reconstruction_errors[y_test == 1]

# Guardar errores para futuros análisis
np.save("./datasets/processed/errors_normal.npy", errors_normal)
np.save("./datasets/processed/errors_anomaly.npy", errors_anomaly)

# Visualizar histogramas
plt.figure(figsize=(10, 6))
plt.hist(errors_normal, bins=50, alpha=0.7, label='Normal')
plt.hist(errors_anomaly, bins=50, alpha=0.7, label='Anomalía')
plt.title("Histograma de errores de reconstrucción (Test set)")
plt.xlabel("Error de reconstrucción (MAE)")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.show()
