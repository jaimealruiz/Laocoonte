import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, f1_score, confusion_matrix, make_scorer
import joblib
import tensorflow as tf

# Cargar modelo entrenado
autoencoder = tf.keras.models.load_model("./models/autoencoder_model.keras")
encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[5].output)

# Cargar datos
X_train = np.load("./datasets/processed/X_train.npy")
X_test = np.load("./datasets/processed/X_test.npy")
y_test = np.load("./datasets/processed/y_test.npy")

# Obtener embeddings latentes
X_train_latent = encoder.predict(X_train)
X_test_latent = encoder.predict(X_test)

# Embeddings 2D
X_train_latent_flat = X_train_latent.reshape(X_train_latent.shape[0], -1)
X_test_latent_flat = X_test_latent.reshape(X_test_latent.shape[0], -1)

# Parámetros para la búsqueda
param_grid = {
    'nu': [0.01, 0.03, 0.05],
    'gamma': ['auto', 'scale', 0.001],
    'kernel': ['rbf']
}

# Scorer compatible con GridSearchCV
def ocsvm_scorer(estimator, X, y=None):
    preds = estimator.predict(X)
    return np.mean(preds == 1)

custom_scorer = make_scorer(ocsvm_scorer, greater_is_better=True, needs_proba=False, needs_threshold=False)

# GridSearchCV con scorer corregido
grid_search = GridSearchCV(
    estimator=OneClassSVM(),
    param_grid=param_grid,
    scoring=custom_scorer,
    cv=3,
    n_jobs=-1,
    verbose=2
)

# Entrenar GridSearchCV
grid_search.fit(X_train_latent_flat)

# Guardar el mejor modelo OCSVM
best_ocsvm = grid_search.best_estimator_
joblib.dump(best_ocsvm, './models/ocsvm_finetuned.pkl')
print("Modelo OCSVM optimizado guardado en ./models/ocsvm_finetuned.pkl")

# Evaluación final en Test
preds_test = best_ocsvm.predict(X_test_latent_flat)
preds_test_labels = np.where(preds_test == -1, 1, 0)

print("\n--- Resultados finales en Test ---")
print(classification_report(y_test, preds_test_labels, target_names=['Normal', 'Anomalía']))
print(f"AUC-ROC final: {roc_auc_score(y_test, preds_test_labels):.3f}")
print(f"F1-score final: {f1_score(y_test, preds_test_labels):.3f}")
