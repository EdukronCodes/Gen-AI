"""
Main FastAPI Application - Multi-Agent AI Framework Backend
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional
import os
from agents.orchestrator import OrchestratorAgent
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Multi-Agent AI Framework", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel('gemini-2.5-pro')

# Initialize Orchestrator
orchestrator = OrchestratorAgent(db_path="retail_banking.db", gemini_client=gemini_model)

# Request/Response Models
class QuestionRequest(BaseModel):
    question: str
    user_id: Optional[str] = None

class QuestionResponse(BaseModel):
    status: str
    message: str
    pdf_path: Optional[str] = None
    summary: Optional[str] = None
    query_results_count: Optional[int] = None
    error: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend interface"""
    frontend_path = os.path.join("frontend", "index.html")
    if os.path.exists(frontend_path):
        with open(frontend_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
        <html>
            <head><title>Multi-Agent AI Framework</title></head>
            <body style="font-family: Arial; padding: 40px; text-align: center;">
                <h1>Multi-Agent AI Framework API</h1>
                <p>Frontend not found. Please use the API endpoints:</p>
                <ul style="text-align: left; display: inline-block;">
                    <li><strong>POST /ask</strong> - Ask a question about the database</li>
                    <li><strong>GET /download/{filename}</strong> - Download generated PDF</li>
                    <li><strong>GET /health</strong> - Health check</li>
                    <li><strong>GET /agents</strong> - Get agent status</li>
                    <li><strong>GET /docs</strong> - API documentation</li>
                </ul>
            </body>
        </html>
        """)

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Multi-Agent AI Framework API",
        "version": "1.0.0",
        "endpoints": {
            "/ask": "POST - Ask a question about the database",
            "/download/{filename}": "GET - Download generated PDF",
            "/health": "GET - Health check",
            "/agents": "GET - Get agent status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": "connected" if os.path.exists("retail_banking.db") else "not found"
    }

@app.get("/agents")
async def get_agents():
    """Get agent status"""
    return orchestrator.get_agent_status()

@app.get("/ask")
async def ask_question_get():
    """GET handler for /ask - redirects to frontend"""
    return HTMLResponse(content="""
    <html>
        <head>
            <title>Method Not Allowed</title>
            <style>
                body { font-family: Arial; padding: 40px; text-align: center; }
                .container { max-width: 600px; margin: 0 auto; }
                .error { color: #d32f2f; font-size: 24px; margin-bottom: 20px; }
                .info { background: #e3f2fd; padding: 20px; border-radius: 8px; margin: 20px 0; }
                a { color: #1976d2; text-decoration: none; font-weight: bold; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="error">⚠️ Method Not Allowed</div>
                <p>The /ask endpoint only accepts POST requests.</p>
                <div class="info">
                    <p><strong>To use this API:</strong></p>
                    <ul style="text-align: left; display: inline-block;">
                        <li>Use the <a href="/">frontend interface</a> to ask questions</li>
                        <li>Or use POST requests to <code>/ask</code> endpoint</li>
                        <li>See <a href="/docs">API documentation</a> for details</li>
                    </ul>
                </div>
                <p><a href="/">← Go to Frontend</a> | <a href="/docs">View API Docs →</a></p>
            </div>
        </body>
    </html>
    """, status_code=405)

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Main endpoint to ask questions about the database
    The orchestrator will coordinate all agents to:
    1. Query the database
    2. Analyze results
    3. Generate PDF summary
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Execute orchestrated workflow
        result = await orchestrator.execute(request.question)
        
        if result.get("status") == "success":
            # Extract just the filename from the full path
            pdf_path = result.get("pdf_path", "")
            pdf_filename = os.path.basename(pdf_path) if pdf_path else None
            
            return QuestionResponse(
                status="success",
                message="Question processed successfully. PDF generated.",
                pdf_path=pdf_filename,
                summary=result.get("summary", ""),
                query_results_count=result.get("query_results_count", 0)
            )
        else:
            return QuestionResponse(
                status="error",
                message="Failed to process question",
                error=result.get("error", "Unknown error")
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/download/{filename}")
async def download_pdf(filename: str):
    """
    Download generated PDF file
    """
    file_path = os.path.join("pdf_outputs", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename=filename
    )

@app.get("/pdfs")
async def list_pdfs():
    """
    List all available PDF files
    """
    pdf_dir = "pdf_outputs"
    if not os.path.exists(pdf_dir):
        return {"pdfs": []}
    
    pdfs = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    return {"pdfs": sorted(pdfs, reverse=True)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

