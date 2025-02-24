# 🐍🛡️Laocoonte🛡️🐍

## "Timeo Danaos et dona ferentes"

## VAE-Based Anomaly Detection in IIoT Networks

Este proyecto implementa un Autoencoder Variacional (VAE) para la detección de anomalías en redes IIoT (Industrial IoT). Utiliza tráfico de red preprocesado para entrenar un modelo que diferencia entre tráfico normal y anómalo. Diseñado para integrarse en firewall Open Source (ex. Suricata) para crear un **Firewall de Nueva Generación (NGFW) Inteligente (INGFW).** Se containerizará utilizando la tecnología **docker-compose/Kubernetes** para su despliegue en redes IIoT.

⚠️ **ADVERTENCIA:** TRABAJO AÚN EN DESARROLLO - WORK IN PROGRESS (WIP)

📌 Características

✅ Implementación de VAE híbrido (Conv1D + Dense) para detección de anomalías.

✅ Preprocesamiento de datos con eliminación de valores irrelevantes y normalización global.

✅ Inferencia en tiempo real sobre tráfico nuevo.

✅ Generación de métricas de evaluación como Matriz de Confusión, Curva ROC y MSE.

### ⚡ Instalación

#### Clona el repositorio y accede a la carpeta del proyecto:

git clone https://github.com/jaimealruiz/Laocoonte.git

cd Laocoonte

#### Crea y activa un entorno virtual:

python -m venv .venv

source .venv/bin/activate  # En Linux/Mac

.venv\Scripts\activate     # En Windows

#### Instala las dependencias:

pip install -r requirements.txt

### 🚀 Uso

#### 1️⃣ Preprocesamiento del dataset

Ejecuta el script para procesar los datos antes de entrenar el modelo:

python preprocess_dataset.py

Esto generará un archivo DNN_Tests_preprocessed.csv con los datos listos para entrenar.

#### 2️⃣ Entrenamiento del VAE

Entrena el modelo con:

python vae.py

Este script guardará el encoder y decoder en formato .keras y calculará el umbral de detección (vae_threshold.npy).

#### 3️⃣ Inferencia y pruebas

Para evaluar el modelo con tráfico nuevo, ejecuta:

python vae_testing.py

Esto cargará los modelos entrenados y generará métricas de evaluación.

### 📊 Evaluación del Modelo

Durante la inferencia, se generan:

Matriz de Confusión para evaluar la detección de anomalías.

Curva ROC y AUC para medir la capacidad de discriminación.

Histograma del Error de Reconstrucción para visualizar el comportamiento del modelo.

### 🛠 Estructura del Proyecto

📂 Laocoonte

│── 📜 preprocess_dataset.py    # Preprocesamiento del dataset

│── 📜 vae.py                   # Entrenamiento del VAE

│── 📜 vae_testing.py           # Inferencia y evaluación

│── 📜 requirements.txt         # Dependencias del proyecto

│── 📂 data                     # Carpeta para datasets

│── 📂 models                   # Modelos entrenados

│── 📜 README.md                # Documentación del proyecto


### 📝 Licencia

📌 Todos los derechos reservados por el autor. No se permite la distribución, modificación o uso sin consentimiento explícito.

### 📧 Contacto:

jaimealru99@gmail.com🔗

[LinkedIn](https://www.linkedin.com/in/jaimealonsoruiz/)
