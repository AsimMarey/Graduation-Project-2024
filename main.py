import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List
import numpy as np
import json
from io import BytesIO
from PIL import Image

# Initialize the FastAPI app
app = FastAPI()

# Load the model and class indices
model_path = 'my_model.h5'
class_indices_path = 'plantnet300K_species_id_2_name.json'


# Set up GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Try to load the model
try:
    model = load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")


# Load the original class indices from the JSON file
with open(class_indices_path, 'r') as f:
    original_class_indices = json.load(f)

# Create a new mapping from 0 to len(original_class_indices) - 1
class_indices = {str(i): label for i, (original_idx, label) in enumerate(original_class_indices.items())}

# Save the new mapping to a new JSON file (optional)
with open('new_class_indices.json', 'w') as f:
    json.dump(class_indices, f)

def preprocess_input(image_batch):
    # Preprocess the input image here
    return tf.keras.applications.mobilenet_v2.preprocess_input(image_batch)

def custom_decode_predictions(preds, class_indices, top=5):
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(class_indices[str(i)], float(pred[i])) for i in top_indices]
        results.append(result)
    return results

def process_images(model, images, size, preprocess_input, top_k=2):
    results = []
    for idx, image in enumerate(images):
        try:
            image = Image.open(BytesIO(image))
            image = image.resize(size)
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            image_batch = np.expand_dims(image_array, axis=0)
            image_batch = preprocess_input(image_batch)
            preds = model.predict(image_batch)
            decoded_preds = custom_decode_predictions(preds, class_indices, top=top_k)
            results.append(decoded_preds)
        except Exception as e:
            results.append(f"Error processing image {idx}: {e}")
    return results
    
@app.post('/predict', response_model=list)
async def predict(files: List[UploadFile] = File(...), top_k: int = 5):
    if model is None:
        raise HTTPException(status_code=500, detail="Model could not be loaded")

    images = [await file.read() for file in files]
    size = (256, 256)
    results = process_images(model, images, size, preprocess_input, top_k)
    return results

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 10034))
    uvicorn.run(app, host='0.0.0.0', port=port)
