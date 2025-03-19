import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Conv1DTranspose, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.image import ssim

# Cargar datos preprocesados
X_train = np.load("./datasets/processed/X_train.npy")
X_val = np.load("./datasets/processed/X_val.npy")

# Dimensión de entrada
input_shape = X_train.shape[1:]

# Diseño del Autoencoder CNN 1D
input_layer = Input(shape=input_shape)

# Encoder
x = Conv1D(64, 7, padding='same', strides=2, kernel_regularizer=l2(1e-4))(input_layer)
x = BatchNormalization()(x)
x = LeakyReLU(negative_slope=0.1)(x)
x = Dropout(0.3)(x)

x = Conv1D(32, 5, padding='same', strides=2, kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = LeakyReLU(negative_slope=0.1)(x)

x = Conv1D(16, 3, padding='same', strides=2, kernel_regularizer=l2(1e-4))(x)
x = BatchNormalization()(x)
x = LeakyReLU(negative_slope=0.1)(x)

# Espacio Latente (Reducido a 4 dimensiones)
latent = Conv1D(4, 3, padding='same')(x)

# Decoder
x = Conv1DTranspose(16, 3, padding='same', strides=2)(latent)
x = BatchNormalization()(x)
x = LeakyReLU(negative_slope=0.1)(x)

x = Conv1DTranspose(32, 5, padding='same', strides=2)(x)
x = BatchNormalization()(x)
x = LeakyReLU(negative_slope=0.1)(x)
x = Dropout(0.3)(x)

x = Conv1DTranspose(64, 7, padding='same', strides=2)(x)
x = BatchNormalization()(x)
x = LeakyReLU(negative_slope=0.1)(x)

# Capa de salida (reconstrucción)
output_layer = Conv1D(1, 3, activation='sigmoid', padding='same')(x)
output_layer = tf.keras.layers.Cropping1D((0, output_layer.shape[1] - input_shape[0]))(output_layer)

# Definir modelo
autoencoder = Model(input_layer, output_layer)

# Compilar modelo con SSIM como métrica adicional
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(ssim(y_true, y_pred, max_val=1.0))

autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mae',
    metrics=[ssim_loss]
)

autoencoder.summary()

# Callbacks adaptativos para control del aprendizaje
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-5),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
]

# Entrenar el modelo con los ajustes adaptativos
history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_val, X_val),
    callbacks=callbacks,
    shuffle=True
)

# Guardar modelo entrenado
autoencoder.save("./models/autoencoder_model_v4.keras")
print("Modelo autoencoder mejorado guardado en ./models/autoencoder_model_v4.keras")
