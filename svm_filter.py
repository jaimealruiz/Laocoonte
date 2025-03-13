import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report

# Cargar datos procesados
X_train = np.load("./datasets/processed/X_train.npy").reshape(len(np.load("./datasets/processed/X_train.npy")), -1)
X_test = np.load("./datasets/processed/X_test.npy").reshape(len(np.load("./datasets/processed/X_test.npy")), -1)
y_test = np.load("./datasets/processed/y_test.npy")

# Entrenar One-Class SVM solo con datos normales
svm_model = OneClassSVM(kernel="rbf", gamma="auto", nu=0.01)  # Ajusta nu según la sensibilidad deseada
svm_model.fit(X_train)

# Predecir en datos de prueba
svm_predictions = svm_model.predict(X_test)

# Convertir salida (-1 es anomalía, 1 es normal) a formato binario (0=normal, 1=anomalía)
svm_predictions = (svm_predictions == -1).astype(int)

# Evaluar desempeño del SVM
print("--- Evaluación del One-Class SVM ---")
print(classification_report(y_test, svm_predictions))

# Guardar predicciones para análisis posterior
np.save("./datasets/processed/svm_predictions.npy", svm_predictions)
