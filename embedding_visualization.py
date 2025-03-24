import numpy as np
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.image import ssim
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Registrar también la función 'hybrid_loss' o 'ssim_metric' si la usas como métrica
@register_keras_serializable(package="Custom", name="hybrid_loss")
def hybrid_loss(y_true, y_pred):
    alpha = 0.8
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_l = 1.0 - tf.reduce_mean(ssim(y_true, y_pred, max_val=1.0))
    return alpha * mae + (1 - alpha) * ssim_l

@register_keras_serializable(package="Custom", name="ssim_metric")
def ssim_metric(y_true, y_pred):
    return 1.0 - tf.reduce_mean(ssim(y_true, y_pred, max_val=1.0))

# Cargar con los custom_objects
autoencoder = tf.keras.models.load_model(
    "./models/autoencoder_model_v5.keras",
    custom_objects={
        "hybrid_loss": hybrid_loss,
        "ssim_metric": ssim_metric
    }
)

encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[5].output)

X_test = np.load("./datasets/processed/X_test.npy")
y_test = np.load("./datasets/processed/y_test.npy")

print("Generando embeddings latentes...")
X_test_latent = encoder.predict(X_test)
X_test_latent_flat = X_test_latent.reshape(X_test_latent.shape[0], -1)

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


# ========================= #
#           JAR             #
# ========================= #