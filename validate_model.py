import numpy as np
import tensorflow as tf
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import time
import joblib

# Cargar modelo entrenado
autoencoder = tf.keras.models.load_model("./models/autoencoder_model_v2.keras")

# Modelo del espacio latente (extraer capa intermedia)
encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[5].output)

# Cargar datos
X_train = np.load("./datasets/processed/X_train.npy")
X_test = np.load("./datasets/processed/X_test.npy")
y_test = np.load("./datasets/processed/y_test.npy")

# Obtener embeddings latentes
print("\n--- Generando embeddings latentes ---")
X_train_latent = encoder.predict(X_train)
X_test_latent = encoder.predict(X_test)
print("Embeddings latentes generados.")

# Reconfigurar embeddings a 2D
print("\n--- Reconfigurando embeddings latentes a 2D ---")
X_train_latent_flat = X_train_latent.reshape(X_train_latent.shape[0], -1)
X_test_latent_flat = X_test_latent.reshape(X_test_latent.shape[0], -1)
print("--- Embeddings latentes reconfigurados ---")

# Entrenar One-Class SVM solo con tráfico normal (ajuste fino)
print("\n--- Entrenando One-Class SVM (ajuste fino) ---")
ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.03)  # Ajuste de nu
ocsvm.fit(X_train_latent_flat)
print("Modelo OCSVM entrenado.")

# Guardar el modelo entrenado
joblib.dump(ocsvm, "./models/ocsvm_model_v1.pkl")
print("Modelo OCSVM guardado en ./models/ocsvm_model.pkl")

# Predicciones
print("\n--- Realizando predicciones ---")
svm_predictions = ocsvm.predict(X_test_latent_flat)
svm_predictions = np.where(svm_predictions == -1, 1, 0)  # Convertir: -1 (anomalía) -> 1, 1 (normal) -> 0
print("Predicciones realizadas.")

# Evaluar desempeño combinado
print("\n--- Métricas de evaluación (Autoencoder + OCSVM) ---")
print(classification_report(y_test, svm_predictions, target_names=['Normal', 'Anomalía']))

# AUC-ROC
auc = roc_auc_score(y_test, svm_predictions)
print(f"AUC-ROC combinado: {auc:.3f}")

# Matriz de confusión
tn, fp, fn, tp = confusion_matrix(y_test, svm_predictions).ravel()

print("\n--- Matriz de Confusión ---")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"True Negatives (TN): {tn}")
print(f"False Negatives (FN): {fn}")

# Tasas relativas
print("\n--- Tasas relativas (%) ---")
print(f"TP rate: {tp / len(y_test) * 100:.2f}%")
print(f"FP rate: {fp / len(y_test) * 100:.2f}%")
print(f"TN rate: {tn / len(y_test) * 100:.2f}%")
print(f"FN rate: {fn / len(y_test) * 100:.2f}%")

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, svm_predictions)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Autoencoder + OCSVM)')
plt.legend()
plt.show()

# Medición de latencia
start_time = time.perf_counter()
ocsvm.predict(X_test_latent_flat[:1000])
end_time = time.perf_counter()
latency = ((end_time - start_time) / 1000) * 1000
print(f"Latencia promedio (OCSVM) por muestra: {latency:.4f} ms")
