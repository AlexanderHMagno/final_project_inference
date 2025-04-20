from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import base64
import numpy as np
from inference import detect_people  # Import the detection function from your existing app

app = FastAPI(
    title="Person Detection API",
    description="API for detecting people in images using YOLOv11",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Endpoint to detect people in an uploaded image
    Returns both the patch analysis and final detection result as base64 encoded images
    """
    try:
        # Read and convert the uploaded file to PIL Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Process the image using your existing detection function
        patches_image, result_image = detect_people(image)
        
        # Convert both images to base64
        patches_buffer = io.BytesIO()
        result_buffer = io.BytesIO()
        
        patches_image.save(patches_buffer, format='PNG')
        result_image.save(result_buffer, format='PNG')
        
        patches_base64 = base64.b64encode(patches_buffer.getvalue()).decode('utf-8')
        result_base64 = base64.b64encode(result_buffer.getvalue()).decode('utf-8')
        
        return JSONResponse({
            "patches": f"data:image/png;base64,{patches_base64}",
            "result": f"data:image/png;base64,{result_base64}"
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 