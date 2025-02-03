import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar el dataset preprocesado
df = pd.read_csv("NormalTraffic-EdgeIIoT-processed.csv")

# Dividir en conjunto de entrenamiento y validación
X_train, X_val = train_test_split(df.values, test_size=0.2, random_state=42)

# Definir la dimensionalidad del espacio latente
latent_dim = 8  

# Definir la capa de muestreo
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Definir el encoder
def build_encoder(input_shape, latent_dim):
    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(32, activation="relu")(inputs)
    x = layers.Dense(16, activation="relu")(x)
    
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    
    z = Sampling()([z_mean, z_log_var])
    
    encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

# Definir el decoder
def build_decoder(latent_dim, output_shape):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(16, activation="relu")(latent_inputs)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(output_shape, activation="sigmoid")(x)
    
    decoder = keras.Model(latent_inputs, outputs, name="decoder")
    return decoder

# Construcción del VAE
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

# Función de pérdida del VAE
def vae_loss(y_true, y_pred):
    reconstruction_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    return reconstruction_loss

# Construcción del modelo
input_shape = (X_train.shape[1],)  # Convertimos a tupla
encoder = build_encoder(input_shape, latent_dim)
decoder = build_decoder(latent_dim, input_shape[0])  # Usamos input_shape[0] aquí
vae = VAE(encoder, decoder)

# Compilar el modelo
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=vae_loss)

# Entrenar el modelo
vae.fit(X_train, X_train, epochs=50, batch_size=64, validation_data=(X_val, X_val))

# Evaluación del error de reconstrucción en tráfico normal
reconstructions = vae.predict(X_val)
mse = np.mean(np.power(X_val - reconstructions, 2), axis=1)

# Definir umbral de detección de anomalías basado en percentil 95
threshold = np.percentile(mse, 95)
print(f"Umbral de detección de anomalías: {threshold}")

# Guardar el modelo entrenado
vae.encoder.save("vae_encoder.h5")
vae.decoder.save("vae_decoder.h5")

vae.encoder.save("vae_encoder_KERAS.keras")
vae.decoder.save("vae_decoder_KERAS.keras")


print("Entrenamiento finalizado. Modelo guardado.")
