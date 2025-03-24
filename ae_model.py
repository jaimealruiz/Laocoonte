import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, Conv1DTranspose, Dropout,
                                     BatchNormalization, LeakyReLU, GaussianNoise)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.image import ssim
from tensorflow.keras.saving import register_keras_serializable

# ==============================
# Registrar la función de pérdida híbrida
# ==============================
@register_keras_serializable(package="Custom", name="hybrid_loss")
def hybrid_loss(y_true, y_pred):
    alpha = 0.8
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_l = 1.0 - tf.reduce_mean(ssim(y_true, y_pred, max_val=1.0))
    return alpha * mae + (1 - alpha) * ssim_l

@register_keras_serializable(package="Custom", name="ssim_metric")
def ssim_metric(y_true, y_pred):
    return 1.0 - tf.reduce_mean(ssim(y_true, y_pred, max_val=1.0))

# Cargar dataset
X_train = np.load("./datasets/processed/X_train.npy")
X_val = np.load("./datasets/processed/X_val.npy")

input_shape = X_train.shape[1:]

# ====== EJEMPLO de arquitectura (puedes ajustarla) ======
input_layer = Input(shape=input_shape)

x = GaussianNoise(0.01)(input_layer)
x = Conv1D(64, 7, padding='same', strides=2, kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.1)(x)
x = Dropout(0.3)(x)

x = Conv1D(32, 5, padding='same', strides=2, kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.1)(x)

x = Conv1D(16, 3, padding='same', strides=2, kernel_regularizer=l2(1e-5))(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.1)(x)

latent = Conv1D(4, 3, padding='same')(x)

x = Conv1DTranspose(16, 3, padding='same', strides=2)(latent)
x = BatchNormalization()(x)
x = LeakyReLU(0.1)(x)

x = Conv1DTranspose(32, 5, padding='same', strides=2)(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.1)(x)
x = Dropout(0.3)(x)

x = Conv1DTranspose(64, 7, padding='same', strides=2)(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.1)(x)

output_layer = Conv1D(1, 3, activation='sigmoid', padding='same')(x)
output_layer = tf.keras.layers.Cropping1D((0, output_layer.shape[1] - input_shape[0]))(output_layer)

autoencoder = Model(input_layer, output_layer)

# Compilar con la loss híbrida
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=hybrid_loss,
    metrics=[ssim_metric]
)

autoencoder.summary()

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-5),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
]

history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_val, X_val),
    callbacks=callbacks,
    shuffle=True
)

autoencoder.save("./models/autoencoder_model_v5.keras")
print("Modelo autoencoder guardado en ./models/autoencoder_model_v5.keras")


# ========================= #
#           JAR             #
# ========================= #