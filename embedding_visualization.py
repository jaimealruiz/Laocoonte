import numpy as np
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.image import ssim
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Registra la función personalizada para poder cargar el modelo correctamente
@register_keras_serializable()
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(ssim(y_true, y_pred, max_val=1.0))

# Cargar autoencoder entrenado indicando la función personalizada
autoencoder = tf.keras.models.load_model(
    "./models/autoencoder_model_v4.keras",
    custom_objects={"ssim_loss": ssim_loss}
)

# Encoder para obtener embeddings latentes (ajusta el índice si cambias la arquitectura)
encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[5].output)

# Cargar datos
X_test = np.load("./datasets/processed/X_test.npy")
y_test = np.load("./datasets/processed/y_test.npy")

# Obtener embeddings latentes
print("Generando embeddings latentes...")
X_test_latent = encoder.predict(X_test)
X_test_latent_flat = X_test_latent.reshape(X_test_latent.shape[0], -1)

# PCA para visualización rápida
print("Aplicando PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test_latent_flat)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[y_test == 0, 0], X_pca[y_test == 0, 1], label='Normal', alpha=0.5)
plt.scatter(X_pca[y_test == 1, 0], X_pca[y_test == 1, 1], label='Anomalía', alpha=0.5)
plt.title("Visualización PCA del espacio latente")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.legend()
plt.show()

# t-SNE con subset por eficiencia computacional
print("Generando visualización t-SNE...")
tsne = TSNE(n_components=2, random_state=42)
subset_size = 5000
X_subset = X_test_latent_flat[:subset_size]
y_subset = y_test[:subset_size]

X_tsne = tsne.fit_transform(X_subset)

plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[y_subset == 0, 0], X_tsne[y_subset == 0, 1], label='Normal', alpha=0.5)
plt.scatter(X_tsne[y_subset == 1, 0], X_tsne[y_subset == 1, 1], label='Anomalía', alpha=0.5)
plt.title("Visualización t-SNE del espacio latente")
plt.legend()
plt.show()
