import os
os.environ["KERAS_BACKEND"] = "jax"
import jax
jax.config.update('jax_platform_name', 'cpu')

import keras
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from keras.utils import load_img, img_to_array
import numpy as np

print("Loading test model...")
mobilenet_model = MobileNetV2(weights='imagenet')
print("Model loaded.")

def test_image(img_path):
    print(f"\nTesting {img_path}")
    try:
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        preds = mobilenet_model.predict(x)
        decoded = decode_predictions(preds, top=10)[0]
        
        print("Top 10 predictions:")
        for _, label, prob in decoded:
            print(f"  {label}: {prob:.4f}")
            
    except Exception as e:
        print(f"Error testing {img_path}: {e}")

# We don't have images easily available, let's just make sure the script runs and the model loads.
