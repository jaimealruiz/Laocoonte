import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar el dataset preprocesado
df = pd.read_csv("./Datasets/v0/NormalTraffic_Training_dataset_preprocessed.csv")

# Dividir en conjunto de entrenamiento y validación
X_train, X_val = train_test_split(df.values, test_size=0.2, random_state=42)

# Redimensionar para la capa Conv1D (agregar una dimensión de canal)
X_train = np.expand_dims(np.nan_to_num(X_train), axis=-1)
X_val = np.expand_dims(np.nan_to_num(X_val), axis=-1)

# Definir la dimensionalidad del espacio latente
latent_dim = 6  

# Definir la capa de muestreo
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Definir el encoder híbrido
def build_encoder(input_shape, latent_dim):
    inputs = keras.Input(shape=input_shape)
    
    # Capa convolucional 1D
    x = layers.Conv1D(32, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Aplanar y pasar a capas densas
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    
    z = Sampling()([z_mean, z_log_var])
    
    encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

# Definir el decoder
def build_decoder(latent_dim, output_shape):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(32, activation="relu")(latent_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(output_shape, activation="sigmoid")(x)
    
    decoder = keras.Model(latent_inputs, x, name="decoder")
    return decoder

# Construcción del VAE
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.z_mean = None
        self.z_log_var = None
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        self.z_mean, self.z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

# Función de pérdida del VAE
def vae_loss(y_true, y_pred):
    z_mean = vae.z_mean
    z_log_var = vae.z_log_var
    reconstruction_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    return reconstruction_loss + kl_loss
    return reconstruction_loss

# Construcción del modelo
input_shape = (X_train.shape[1], 1)  # Agregamos la dimensión de canal
encoder = build_encoder(input_shape, latent_dim)
decoder = build_decoder(latent_dim, X_train.shape[1])
vae = VAE(encoder, decoder)

# Compilar el modelo
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0), loss=vae_loss)

# Entrenar el modelo
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
vae.fit(X_train, X_train, epochs=50, batch_size=64, validation_data=(X_val, X_val), callbacks=[early_stopping])

# Evaluación del error de reconstrucción en tráfico normal
reconstructions = vae.predict(X_val)
mse = np.mean(np.power(np.squeeze(X_val) - reconstructions, 2), axis=1)

# Definir umbral de detección de anomalías basado en percentil 95
threshold = np.percentile(mse, 95)
print(f"Umbral de detección de anomalías: {threshold}")

# Guardar el modelo entrenado
vae.encoder.save("encoder_v1.keras")
vae.decoder.save("decoder_v1.keras")

print("Entrenamiento finalizado. Modelo guardado.")
