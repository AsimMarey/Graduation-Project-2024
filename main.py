from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, validator
from typing import Literal, List
import httpx

app = FastAPI()

# Constant
API_KEY = "2b10jRtH0kF5BARgBUnUk9eKdO" 
PROJECT = "all" 

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

if __name__ == '__main__':
    import uvicorn
    import os
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
