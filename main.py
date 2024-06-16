from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, conlist
from typing import List
import httpx

app = FastAPI()

# Constants
API_KEY = "2b10jRtH0kF5BARgBUnUk9eKdO" 
PROJECT = "all" 

class OrgansModel(BaseModel):
    organs: conlist(Literal['leaf', 'flower', 'fruit', 'auto'], min_items=1, max_items=1)


# Define the endpoint for image upload and external API call
@app.post("/identify-plant")
async def identify_plant(
    organs: Literal['leaf', 'flower', 'fruit', 'auto'] = Form(...),
    image: UploadFile = File(...)
):
    # Define the external API URL
    external_api_url = f"https://my-api.plantnet.org/v2/identify/{PROJECT}?api-key={API_KEY}"
    
    # Read the image file content
    image_content = await image.read()
    
    # Prepare the data for the external API request
    files = {"images": image_content}
    data = {
        "api-key": API_KEY,
        "organs": organs
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
            species_info = res.get("species", {})
            scientific_name = species_info.get("scientificNameWithoutAuthor", "N/A")
            common_names = species_info.get("commonNames", ["N/A"])
            common_name = common_names[0] if common_names else "N/A"
            parsed_results.append({
                "score": score,
                "scientific_name": scientific_name,
                "common_name": common_name
            })

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
