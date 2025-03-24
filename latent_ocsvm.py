import numpy as np
import tensorflow as tf
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Cargar modelo autoencoder entrenado y su encoder
autoencoder = tf.keras.models.load_model("./models/autoencoder_model_v4.keras", compile=False)
encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[5].output)

# Cargar datasets
X_train = np.load("./datasets/processed/X_train.npy")
X_test = np.load("./datasets/processed/X_test.npy")
y_test = np.load("./datasets/processed/y_test.npy")

# Extraer embeddings latentes
latent_train = encoder.predict(X_train)
latent_test = encoder.predict(X_test)

# Aplanar embeddings
latent_train_flat = latent_train.reshape(latent_train.shape[0], -1)
latent_test_flat = latent_test.reshape(latent_test.shape[0], -1)

# Entrenar One-Class SVM (solo datos normales)
ocsvm = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale')
ocsvm.fit(latent_train_flat)

# Predecir anomalías en test
y_pred = ocsvm.predict(latent_test_flat)
y_pred = np.where(y_pred == -1, 1, 0)  # Convertir a (0 normal, 1 anomalía)

# Evaluar rendimiento
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomalía']))

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred)
print(f"AUC ROC: {roc_auc:.4f}")


# ========================= #
#           JAR             #
# ========================= #