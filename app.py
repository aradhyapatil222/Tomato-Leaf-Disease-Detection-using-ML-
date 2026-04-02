import os
os.environ["KERAS_BACKEND"] = "jax"
import jax
jax.config.update('jax_platform_name', 'cpu')

from flask import Flask, render_template, request, jsonify
import keras
from keras.utils import load_img, img_to_array
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
import io

app = Flask(__name__)

print("Starting app...")

# Load the plant leaf validation model (MobileNetV2 pre-trained on ImageNet)
try:
    print("Loading validation model...")
    validation_model = MobileNetV2(weights='imagenet')
    print("Validation model loaded successfully!")
except Exception as e:
    print(f"Error loading validation model: {e}")
    validation_model = None

# Load the tomato disease classification model
try:
    print("Loading classification model...")
    model = keras.models.load_model('model_inception.h5')
    print("Classification model loaded successfully!")
except Exception as e:
    print(f"Error loading classification model: {e}")
    model = None

# ImageNet class names that represent plants or leaves (refined)
PLANT_CLASSES = {
    'leaf', 'leafy', 'plant', 'flower', 'fruit', 'vegetable', 'tree', 'bush', 'shrub',
    'grass', 'tomato', 'potato', 'garden', 'greenhouse', 'orchid', 'pot',
    'herb', 'flora', 'botanical', 'nature', 'seed', 'sprout',
    'cucumber', 'pepper', 'corn', 'maize', 'wheat', 'rice', 'bean', 'pea',
    'daisy', 'sunflower', 'rose', 'tulip', 'lily', 'acorn', 'fig', 'pineapple',
    'banana', 'orange', 'lemon', 'pomegranate', 'strawberry', 'broccoli',
    'cauliflower', 'cabbage', 'artichoke', 'bell_pepper', 'chili', 'mushroom',
    'cardoon', 'artichoke', 'buckeye', 'conker', 'custard_apple', 'jackfruit',
    'zucchini', 'squash', 'gourd', 'kale', 'spinach', 'lettuce', 'basil', 'cilantro',
    'parsley', 'mint', 'thyme', 'rosemary', 'sage', 'oregano', 'potted_plant'
}

def is_mostly_botanical_color(img_array):
    """
    Checks if the image has colors consistent with plant leaves (greens, yellows, browns).
    """
    # img_array is (1, 224, 224, 3) and preprocess_input for MobileNetV2 
    # might have changed the range. Let's work with raw image array.
    # We'll calculate the percentage of green-ish pixels.
    # (Simplified version: Check if Green channel is dominant in many pixels)
    
    # Calculate average color
    avg_color = np.mean(img_array[0], axis=(0, 1))
    r, g, b = avg_color
    
    # In leaves, Green is usually higher than Blue and Red, or at least 
    # significant. For brown/dead leaves, R and G are high, B is low.
    # Most random art/posters have high R or complex mixes.
    
    # Heuristic: Plant leaves usually have G > B. 
    # Let's be a bit more flexible to include yellowing leaves.
    is_botanical = (g > b * 0.8) and (g > 30) # Even more flexible
    print(f"Color check: R={r:.2f}, G={g:.2f}, B={b:.2f}, is_botanical={is_botanical}")
    return is_botanical

def is_plant_leaf(img_bytes):
    """
    Validates if the image contains a plant or leaf using MobileNetV2 and color heuristics.
    """
    if validation_model is None:
        return True, "Validation skipped (model not loaded)"

    try:
        # Load for color heuristic first (using raw array)
        img_raw = load_img(io.BytesIO(img_bytes), target_size=(224, 224))
        x_raw = img_to_array(img_raw)
        
        # Color check
        if not is_mostly_botanical_color(np.expand_dims(x_raw, axis=0)):
            print("Image colors do not match a plant or leaf profile.")
            return False, "Image colors do not match a plant or leaf profile."

        # Preprocess for MobileNetV2
        x = preprocess_input(np.expand_dims(x_raw, axis=0))

        # Get top 5 predictions
        preds = validation_model.predict(x)
        decoded = decode_predictions(preds, top=5)[0]
        
        # Check if any of the top predictions match our plant classes
        is_plant = False
        top_labels = []
        highest_score = 0
        
        for _, label, score in decoded:
            top_labels.append(f"{label} ({score:.2f})")
            if score > highest_score:
                highest_score = score
            
            # Check for keywords in the label
            if any(plant_keyword in label.lower() for plant_keyword in PLANT_CLASSES):
                # Stricter confidence threshold: 0.15
                if score > 0.15:
                    is_plant = True
                    break
        
        print(f"Validation results: {top_labels}")
        
        # If the top result is very high confidence and NOT a plant, reject it
        # even if a lower result matched a keyword.
        first_label, first_score = decoded[0][1], decoded[0][2]
        if first_score > 0.7 and not any(pk in first_label.lower() for pk in PLANT_CLASSES):
            return False, f"Detected {first_label.replace('_', ' ')} with very high confidence. Not a plant."

        if is_plant:
            return True, "Plant detected"
        else:
            return False, f"Image content does not match a plant leaf. Detected: {first_label.replace('_', ' ')}"
            
    except Exception as e:
        print(f"Validation error: {e}")
        return True, f"Validation error: {e}" # Fallback to allow processing


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
        
        # Step 1: Validate image content
        print("Validating image content...")
        is_plant, message = is_plant_leaf(file_content)
        if not is_plant:
            return jsonify({'error': f'Image validation failed: {message}'}), 400
        print(f"Validation successful: {message}")

        # Step 2: Perform disease prediction
        print("Loading image for classification...")
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