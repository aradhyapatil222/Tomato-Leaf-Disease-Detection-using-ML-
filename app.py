import os
os.environ["KERAS_BACKEND"] = "jax"
import jax
jax.config.update('jax_platform_name', 'cpu')

from flask import Flask, render_template, request, jsonify
import keras
from keras.utils import load_img, img_to_array
import numpy as np
import io

app = Flask(__name__)

print("Starting app...")
# Load the model you just downloaded
try:
    print("Loading model...")
    model = keras.models.load_model('model_inception.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Common 10-class PlantVillage Tomato dataset ordering (Healthy at the end)
classes = [
    'Bacterial Spot', 
    'Early Blight', 
    'Late Blight', 
    'Leaf Mold', 
    'Septoria Leaf Spot', 
    'Spider Mites', 
    'Target Spot', 
    'Mosaic Virus', 
    'Yellow Leaf Curl Virus',
    'Healthy'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file_storage = request.files['file']
        if file_storage.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        print(f"Received file: {file_storage.filename}")
        file_content = file_storage.read()
        print("Loading image...")
        # Use bilinear interpolation for better quality resizing
        img = load_img(io.BytesIO(file_content), target_size=(224, 224), interpolation='bilinear')
        print("Converting to array...")
        # Image preprocessing: ensure correct channel order (RGB) and range
        x = img_to_array(img) 
        
        # Diagnostic: print raw image stats before and after normalization
        # InceptionV3 models often expect [-1, 1] normalization
        x = (x / 127.5) - 1.0
        x = np.expand_dims(x, axis=0)
        print(f"Image stats (after normalization) - Min: {np.min(x)}, Max: {np.max(x)}, Mean: {np.mean(x)}")
        
        print("Running prediction...")
        prediction = model.predict(x)
        
        # DEBUG: Print all classes to see if the model is producing high confidence elsewhere
        top_k = 5
        top_indices = np.argsort(prediction[0])[-top_k:][::-1]
        print("Top 5 Predictions:")
        for idx in top_indices:
            print(f"  {classes[idx]}: {prediction[0][idx]*100:.2f}%")
            
        print("Prediction successful!")
        index = np.argmax(prediction)
        
        return jsonify({
            'disease': classes[index],
            'confidence': f"{np.max(prediction)*100:.2f}%"
        })
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)