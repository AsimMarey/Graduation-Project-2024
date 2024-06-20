import os
import tensorflow as tf
from fastapi.responses import JSONResponse, HTMLResponse
import numpy as np
import json
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, validator
from typing import Literal, List
import httpx

app = FastAPI()

# Constant
API_KEY = "2b10jRtH0kF5BARgBUnUk9eKdO" 
PROJECT = "all" 


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
    model = tf.keras.models.load_model('my_model.h5')
    print("Model loaded successfully")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Load the original class indices from the JSON file
with open('plantnet300K_species_id_2_name.json', 'r') as f:
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


class OrgansModel(BaseModel):
    organs: Literal['leaf', 'flower', 'fruit', 'auto']

    @validator('organs', pre=True)
    def check_organs(cls, v):
        if isinstance(v, list):
            if len(v) != 1:
                raise ValueError('Exactly one organ should be specified.')
            v = v[0]
        return v
        
# Define the endpoint for image upload and external API call
@app.post("/identify-plant")
async def identify_plant(
    organs: List[Literal['leaf', 'flower', 'fruit', 'auto']] = Form(...),
    image: UploadFile = File(...)
):
    # Define the external API URL with API key
    external_api_url = f"https://my-api.plantnet.org/v2/identify/{PROJECT}?api-key={API_KEY}"
    
    # Read the image file content
    image_content = await image.read()
    
    # Prepare the data for the external API request
    size = (256, 256)
    files = {"images": (image.filename, image_content, image.content_type)}
    data = {
        "organs": organs[0]  # Since organs should be exactly one item
    }
    
    try:
        # Make the request to the external API
        async with httpx.AsyncClient() as client:
            response = await client.post(external_api_url, files=files, data=data)
            response.raise_for_status()
            result = response.json()
        
        # Parse the external API response
        results = result.get("results", [])
        if not results:
            raise HTTPException(status_code=400, detail="No results found")

        parsed_results = []
        for res in results:
            score = res.get("score", 0)
            probability = round(score * 100, 2)
            if probability > 5:
                species_info = res.get("species", {})
                scientific_name = species_info.get("scientificNameWithoutAuthor", "N/A")
                common_names = species_info.get("commonNames", ["N/A"])
                common_name = common_names[0] if common_names else "N/A"
                parsed_results.append({
                    "probability": f"{probability}%",  # Attach % to the formatted probability string
                    "scientific_name": scientific_name,
                    "common_name": common_name
                })

        if not parsed_results:
            raise HTTPException(status_code=400, detail="No results found with probability higher than 5%")

        return {"results": parsed_results}

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"External API error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.head("/")
async def head_index():
    return {"message": "Service is running"}
    
if __name__ == '__main__':
    import uvicorn
    import os
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
