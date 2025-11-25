import os
import shutil
import uuid
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.graph import app as workflow_app
from backend.models import InvoiceState

app = FastAPI(title="Multi-Agent Invoice System")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.post("/process")
async def process_invoice(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF and images are supported.")

    # Save the file
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # If PDF, we might need to convert to image first for the vision model, 
    # but for simplicity let's assume the user uploads images or we handle PDF-to-image here.
    # For this demo, let's assume the input is an image path for the vision model.
    # If it's a PDF, we would convert it. Let's add a basic check.
    
    final_image_path = file_path
    if file_ext.lower() == '.pdf':
        # TODO: Implement PDF to Image conversion if needed.
        # For now, let's assume the user uploads images as per the previous task context (converted pdfs are available but vision models work best on images).
        # Actually, Gemini 1.5 Pro/Flash can handle PDFs directly if passed correctly, but image is safer for "vision".
        # Let's assume for this MVP we accept images. If PDF, we'd need `pdf2image`.
        pass

    # Initial state
    initial_state = InvoiceState(image_path=final_image_path)

    # Run the graph
    # invoke returns the final state
    final_state = workflow_app.invoke(initial_state)
    
    return final_state

@app.get("/")
def read_root():
    return {"message": "Multi-Agent Invoice System API is running"}
