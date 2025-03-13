import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Conv1DTranspose, Dropout
from tensorflow.keras.optimizers import Adam

# Cargar datos preprocesados
X_train = np.load("./datasets/processed/X_train.npy")
X_val = np.load("./datasets/processed/X_val.npy")

# Dimensión de entrada
input_shape = X_train.shape[1:]

# Diseño del Autoencoder CNN 1D
input_layer = Input(shape=input_shape)

# Encoder
x = Conv1D(filters=32, kernel_size=7, activation='relu', padding='same', strides=2)(input_layer)
x = Dropout(0.2)(x)
x = Conv1D(filters=16, kernel_size=5, activation='relu', padding='same', strides=2)(x)
x = Conv1D(filters=8, kernel_size=3, activation='relu', padding='same', strides=2)(x)

# Latent space
latent = Conv1D(filters=4, kernel_size=3, activation='relu', padding='same')(x)

# Decoder
x = Conv1DTranspose(filters=8, kernel_size=3, activation='relu', padding='same', strides=2)(latent)
x = Conv1DTranspose(filters=16, kernel_size=5, activation='relu', padding='same', strides=2)(x)
x = Dropout(0.2)(x)
x = Conv1DTranspose(filters=32, kernel_size=7, activation='relu', padding='same', strides=2)(x)

# Capa de salida (reconstrucción)
output_layer = Conv1D(filters=1, kernel_size=3, activation='sigmoid', padding='same')(x)
# Ajustar dimensión exacta
output_layer = tf.keras.layers.Cropping1D(cropping=(0, output_layer.shape[1] - input_shape[0]))(output_layer)

# Definir modelo
autoencoder = Model(inputs=input_layer, outputs=output_layer)

autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss='mae')

autoencoder.summary()

# Entrenar el modelo con early stopping
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

history = autoencoder.fit(
    X_train, X_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_val, X_val),
    callbacks=callbacks,
    shuffle=True
)

# Guardar modelo entrenado
autoencoder.save("./models/autoencoder_model_v0.keras")
