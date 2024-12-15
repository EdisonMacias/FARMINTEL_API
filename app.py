from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import uuid

# Inicializar la app Flask y habilitar CORS
app = Flask(__name__)
CORS(app)

# Cargar el modelo preentrenado
model = load_model('./model/modelo_diatraea_vgg16_augmented.h5')

# Definir las clases según el modelo
CLASS_NAMES = ["Adulta", "Con daño", "Huevos", "Larvas", "Sin daño", "Cogollero"]

# Crear carpetas para cada clase si no existen
BASE_FOLDER = "data/train/"
for class_name in CLASS_NAMES:
    os.makedirs(os.path.join(BASE_FOLDER, class_name), exist_ok=True)

# Función para procesar la imagen (ajuste al modelo)
def preprocess_image(image):
    image = image.resize((150, 150))  # Ajustar al tamaño de entrada del modelo
    image = np.array(image)  # Convertir a array de numpy
    image = np.expand_dims(image, axis=0)  # Añadir dimensión para el lote
    image = image / 255.0  # Normalizar valores entre 0 y 1
    return image

# Ruta para realizar predicción y, opcionalmente, almacenar imágenes clasificadas
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No se encontró un archivo."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"success": False, "message": "No se seleccionó un archivo."}), 400

    # Guardar la imagen temporalmente
    filename = secure_filename(file.filename)
    temp_path = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(temp_path)

    # Abrir y preprocesar la imagen
    image = Image.open(temp_path)
    processed_image = preprocess_image(image)

    # Hacer predicción
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    predicted_probability = prediction[0][predicted_class_index]

    # Determinar si se debe almacenar la imagen en una carpeta
    store_image = request.args.get('store', 'false').lower() == 'true'

    if store_image:
        # Crear ruta de almacenamiento para la clase
        target_folder = os.path.join(BASE_FOLDER, predicted_class_name)

        # Renombrar la imagen con un identificador único
        new_filename = f"{predicted_class_name}_{uuid.uuid4().hex[:8]}.jpg"
        new_path = os.path.join(target_folder, new_filename)

        # Mover la imagen a su carpeta correspondiente
        os.rename(temp_path, new_path)
    else:
        # Eliminar la imagen temporal si no se almacena
        os.remove(temp_path)

    # Responder con el resultado
    return jsonify({
        "success": True,
        "class": predicted_class_name,
        "probability": float(predicted_probability),
        "stored": store_image
    }), 200

# Ruta para verificar el estado de la API
@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "success": True,
        "message": "La API está activa y funcional."
    }), 200

# Ruta para obtener métricas del modelo
@app.route('/metrics', methods=['GET'])
def metrics():
    metrics_info = {
        "success": True,
        "model_name": "modelo_diatraea_vgg16_augmented",
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "description": "Modelo para clasificación de plagas en 6 categorías."
    }
    return jsonify(metrics_info), 200

# Iniciar la app Flask
if __name__ == '__main__':
    app.run(debug=True)
