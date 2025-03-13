from tensorflow.keras.models import load_model
import joblib

# Carga del Autoencoder
autoencoder = load_model("./models/autoencoder.keras")

# Generar embeddings latentes
X_nuevos_latentes = autoencoder.predict(X_nuevos)

# Aplanar embeddings latentes para OCSVM
X_nuevos_latentes_flat = X_nuevos_latentes.reshape((X_nuevos_latentes.shape[0], -1))

# Carga del modelo OCSVM
ocsvm = joblib.load("./models/ocsvm_model.pkl")

# Clasificación con OCSVM
predicciones = ocsvm.predict(X_nuevos_latentes_flat)
predicciones = np.where(predicciones == -1, 1, 0)
