from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import io
import base64
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# ── load models at startup ──────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

models = {
    'vgg16':        {'path': os.path.join(MODEL_DIR, 'vgg16_bmi.keras'),         'uses_gender': False},
    'vggface':      {'path': os.path.join(MODEL_DIR, 'vggface_bmi.keras'),       'uses_gender': False},
}

loaded_models = {}
for name, cfg in models.items():
    if os.path.exists(cfg['path']):
        loaded_models[name] = load_model(cfg['path'])
        print(f"Loaded {name}")
    else:
        print(f"WARNING: {name} not found at {cfg['path']}")

# ── helpers ─────────────────────────────────────────────────────────────────
def preprocess_image(image_bytes, model_id):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
    x = img_to_array(img).astype('float32')
    if model_id == 'vggface':
        # VGG-Face mean subtraction (BGR order)
        x[..., 0] -= 129.1863
        x[..., 1] -= 104.7624
        x[..., 2] -= 93.5940
    else:
        x = preprocess_input(x)
    return np.expand_dims(x, axis=0)

def bmi_confidence_range(bmi, margin=3.5):
    """Simple ±margin confidence range based on model MAE (~6 BMI units)."""
    return round(bmi - margin, 1), round(bmi + margin, 1)

# ── routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/models', methods=['GET'])
def list_models():
    return jsonify({
        'models': [
            {'id': 'vgg16',      'label': 'VGG16 (ImageNet)',          'uses_gender': False},
            {'id': 'vggface',    'label': 'VGG-Face',                  'uses_gender': False},
            {'id': 'vggface_v2', 'label': 'VGG-Face + Gender (best)',  'uses_gender': True},
        ],
        'available': list(loaded_models.keys())
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    model_id = data.get('model', 'vggface_v2')
    gender   = data.get('gender', 'Male')   # 'Male' or 'Female'

    if model_id not in loaded_models:
        return jsonify({'error': f'Model {model_id} not loaded'}), 400

    # decode base64 image
    image_bytes = base64.b64decode(data['image'].split(',')[-1])
    x = preprocess_image(image_bytes, model_id)

    model     = loaded_models[model_id]
    uses_gender = models[model_id]['uses_gender']

    if uses_gender:
        gender_val = np.array([[1 if gender == 'Male' else 0]])
        bmi = float(model.predict([x, gender_val], verbose=0)[0][0])
    else:
        bmi = float(model.predict(x, verbose=0)[0][0])

    low, high = bmi_confidence_range(bmi)

    return jsonify({
        'bmi':        round(bmi, 1),
        'range_low':  low,
        'range_high': high,
        'model':      model_id,
        'gender':     gender,
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
