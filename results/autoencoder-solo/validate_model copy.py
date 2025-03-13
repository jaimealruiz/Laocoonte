import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_fscore_support,
    roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import time


# Cargar modelo entrenado
autoencoder = tf.keras.models.load_model("./models/autoencoder_model_v0.keras")

# Cargar datos de prueba
X_test = np.load("./datasets/processed/X_test.npy")
y_test = np.load("./datasets/processed/y_test.npy")

# Reconstruir datos con el modelo
reconstructions = autoencoder.predict(X_test)

# Calcular errores de reconstrucción (MAE por muestra)
mae_loss = np.mean(np.abs(reconstructions - X_test), axis=(1, 2))

# Determinar umbral automático (percentil 99 sobre errores benignos)
threshold = np.percentile(mae_loss[y_test == 0], 98)
print("Umbral de anomalía (percentil 98):", threshold)

# Clasificación de anomalías
anomaly_predictions = (reconstructions.mean(axis=(1,2)) > threshold).astype(int)

# Evaluar rendimiento
from sklearn.metrics import classification_report, roc_auc_score

precision, recall, f1, _ = precision_recall_fscore_support(y_test, anomaly_predictions, average='binary')

print("Precision:", precision_score(y_test, anomaly_predictions))
print("Recall:", recall_score(y_test, anomaly_predictions))
print("F1-Score:", f1_score(y_test, anomaly_predictions))

# AUC-ROC
from sklearn.metrics import roc_auc_score, roc_curve

auc = roc_auc_score(y_test, reconstructions.mean(axis=(1,2)))
print("AUC-ROC:", auc)

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, reconstructions.mean(axis=(1,2)))

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Histograma del error de reconstrucción
plt.hist(reconstructions[y_test==0].mean(axis=(1,2)), bins=50, alpha=0.7, label="Normal")
plt.hist(reconstructions.mean(axis=(1,2))[y_test==1], bins=50, alpha=0.7, label='Ataque')
plt.axvline(threshold, color='r', linestyle='--', label='Umbral (99%)')
plt.xlabel('Error de reconstrucción (MAE)')
plt.ylabel('Frecuencia')
plt.title('Distribución de errores de reconstrucción')
plt.legend()
plt.show()

# Medición de latencia de inferencia
import time
n_samples = 1000
start_time = time.perf_counter()
autoencoder.predict(X_test[:n_samples])
end_time = time.perf_counter()

inference_time_ms = ((end_time - start_time) / n_samples) * 1000
print(f"Latencia promedio por muestra: {inference_time_ms:.4f} ms")

# Matriz de confusión
tn, fp, fn, tp = confusion_matrix(y_test, anomaly_predictions).ravel()

# Imprimir tasas
print("\n--- Matriz de Confusión ---")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"True Negatives (TN): {tn}")
print(f"False Negatives (FN): {fn}")

# Opcionalmente, calcular tasas relativas (%)
total = tp + tn + fp + fn
print("\n--- Tasas relativas (%) ---")
print(f"TP rate: {100 * tp / total:.2f}%")
print(f"FP rate: {100 * fp / total:.2f}%")
print(f"TN rate: {100 * tn / total:.2f}%")
print(f"FN rate: {100 * fn / total:.2f}%")

