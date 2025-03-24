import numpy as np
import tensorflow as tf
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import ParameterGrid

# Cargar el encoder entrenado previamente
autoencoder = tf.keras.models.load_model("./models/autoencoder_model_v4.keras", compile=False)
encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[5].output)

# Cargar datos
X_train = np.load("./datasets/processed/X_train.npy")
X_test = np.load("./datasets/processed/X_test.npy")
y_test = np.load("./datasets/processed/y_test.npy")

# Embeddings latentes
X_train_latent = encoder.predict(X_train).reshape(len(X_train), -1)
X_test_latent = encoder.predict(X_test).reshape(len(X_test), -1)

# Espacio de hiperparámetros a explorar
param_grid = {
    'nu': [0.001, 0.005, 0.01, 0.02, 0.05],
    'gamma': ['auto', 'scale', 0.1, 0.01, 0.001]
}

best_auc = 0
best_params = None
best_report = None
best_cm = None

print("Iniciando búsqueda de hiperparámetros...")

# Iterar sobre todas las combinaciones
for params in ParameterGrid(param_grid):
    print(f"Probando configuración: nu={params['nu']} | gamma={params['gamma']}")

    ocsvm = OneClassSVM(nu=params['nu'], kernel="rbf", gamma=params['gamma'])
    ocsvm.fit(X_train_latent)

    predictions = ocsvm.predict(X_test_latent)
    predictions = np.where(predictions == 1, 0, 1)

    auc = roc_auc_score(y_test, predictions)
    print(f"AUC ROC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_params = params
        best_report = classification_report(y_test, predictions, target_names=["Normal", "Anomalía"])
        best_cm = confusion_matrix(y_test, predictions)

print("\n--- Mejor configuración encontrada ---")
print(f"Mejores hiperparámetros: {best_params}")
print(f"AUC ROC del mejor modelo: {best_auc:.4f}")
print("\nReporte de clasificación:\n", best_report)
print("\nMatriz de confusión:\n", best_cm)
