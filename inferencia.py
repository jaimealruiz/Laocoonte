import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Definir la clase Sampling (necesaria para la capa personalizada en el encoder)
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Cargar el modelo entrenado
encoder = tf.keras.models.load_model(
    "encoder_v1.keras", 
    custom_objects={'Sampling': Sampling}
)
decoder = tf.keras.models.load_model("decoder_v1.keras")

# Cargar el dataset etiquetado para pruebas
# Este dataset debe contener la columna 'Attack_label'
test_df = pd.read_csv("labeled-DNN-EdgeIIoT-processed.csv")

# Separar tráfico normal y anómalo usando 'Attack_label'
normal_mask = test_df['Attack_label'] == 0
anomalous_mask = test_df['Attack_label'] == 1

X_test_normal = test_df[normal_mask].drop(columns=['Attack_label']).values
X_test_anomalous = test_df[anomalous_mask].drop(columns=['Attack_label']).values

# Reconstrucción del tráfico normal y anómalo
reconstructions_normal = decoder.predict(encoder.predict(X_test_normal)[0])
mse_normal = np.mean(np.power(X_test_normal - reconstructions_normal, 2), axis=1)

reconstructions_anomalous = decoder.predict(encoder.predict(X_test_anomalous)[0])
mse_anomalous = np.mean(np.power(X_test_anomalous - reconstructions_anomalous, 2), axis=1)

# Determinar el umbral de detección de anomalías basado en el percentil 95 del tráfico normal
threshold = np.percentile(mse_normal, 95)
print(f"Nuevo umbral de detección de anomalías: {threshold}")

# Generar etiquetas predichas para el dataset completo
X_test = test_df.drop(columns=['Attack_label']).values
true_labels = test_df['Attack_label'].values

reconstructions = decoder.predict(encoder.predict(X_test)[0])
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
predicted_labels = (mse > threshold).astype(int)

# Calcular métricas de evaluación
print("\nMatriz de Confusión:")
print(confusion_matrix(true_labels, predicted_labels))

print("\nReporte de Clasificación:")
print(classification_report(true_labels, predicted_labels))

# Visualizar la distribución de errores de reconstrucción
plt.hist(mse_normal, bins=50, alpha=0.7, label="Tráfico Normal (MSE)")
plt.hist(mse_anomalous, bins=50, alpha=0.7, label="Tráfico Anómalo (MSE)")
plt.axvline(threshold, color='red', linestyle='--', label="Umbral")
plt.title("Distribución del Error de Reconstrucción")
plt.xlabel("Error de Reconstrucción (MSE)")
plt.ylabel("Frecuencia")
plt.legend()
plt.show()
