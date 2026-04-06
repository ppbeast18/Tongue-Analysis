import os
import json
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras import backend as K
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def dice_coef(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Load Models
config_path = os.path.join(BASE_DIR, 'config.json')
weights_path = os.path.join(BASE_DIR, 'model.weights.h5')
classifier_path = os.path.join(BASE_DIR, 'best_model_effnet_mask.h5')

with open(config_path, 'r') as f: unet_config = json.load(f)
segment_model = model_from_json(json.dumps(unet_config), custom_objects={'dice_coef': dice_coef})
segment_model.load_weights(weights_path)
classifier_model = load_model(classifier_path, custom_objects={'dice_coef': dice_coef}, compile=False)

CLASS_MAP = {
    0: {"label": "Light Yellow", "stats": "Indicates mild internal heat. May suggest mild dehydration."},
    1: {"label": "White", "stats": "Indicates cold or dampness. Focus on warming foods."},
    2: {"label": "Yellow", "stats": "Indicator of significant internal heat. Suggests cooling diet."}
}

@app.route('/')
def index(): return render_template('homepage.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filename = file.filename
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    # 1. Processing with High-Quality Interpolation
    img_bgr = cv2.imread(input_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Use INTER_AREA for more accurate downsampling
    img_256 = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    
    # 2. Segmentation
    seg_input = img_256.astype('float32') / 255.0
    mask = segment_model.predict(np.expand_dims(seg_input, axis=0), verbose=0)[0]
    mask_bin = (mask > 0.5).astype(np.uint8)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'mask_' + filename), mask_bin * 255)

    # 3. Classification
    img_class = eff_preprocess(img_256.copy().astype('float32'))
    mask_class = cv2.resize(mask_bin, (256, 256), interpolation=cv2.INTER_NEAREST).astype('float32')
    mask_class = np.expand_dims(mask_class, axis=-1)

    prediction = classifier_model.predict({
        "image_input": np.expand_dims(img_class, axis=0),
        "mask_input": np.expand_dims(mask_class, axis=0)
    }, verbose=0)[0]
    
    # --- LOGIC TUNING ---
    # If the model is conflicted between Index 0 (Light Yellow) and 2 (Yellow),
    # we use a small bias factor to favor the stronger category (Yellow) 
    # if its probability is significant (over 30%).
    if prediction[2] > 0.30 and prediction[0] > 0.40:
        class_idx = 2
    else:
        class_idx = np.argmax(prediction)

    print(f"--- ROBUST LOG --- File: {filename} | Predicted: {class_idx} | Raw: {prediction}")

    result_data = CLASS_MAP.get(class_idx)
    return render_template('result.html', input_img=filename, mask_img='mask_'+filename, 
                           label=result_data['label'], stats=result_data['stats'])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)