from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import logging
import time
from typing import Dict, Any
from .inference import transcribe_audio
from .utils import validate_audio, cleanup_temp_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NeMo ASR Transcription Service",
    description="A FastAPI-based Automatic Speech Recognition service using NVIDIA NeMo",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global statistics for monitoring
stats = {
    "total_requests": 0,
    "successful_transcriptions": 0,
    "failed_transcriptions": 0,
    "average_processing_time": 0.0
}

@app.get("/")
async def root():
    """Root endpoint providing service information"""
    return {
        "message": "Welcome to the NeMo ASR Transcription Service!",
        "service": "NVIDIA NeMo Hindi ASR",
        "model": "stt_hi_conformer_ctc_medium",
        "supported_formats": ["audio/wav"],
        "max_duration": "10 seconds",
        "min_duration": "5 seconds",
        "sample_rate": "16kHz"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "NeMo ASR",
        "timestamp": time.time()
    }

@app.get("/stats")
async def get_stats():
    """Get service statistics"""
    return stats

@app.post("/transcribe")
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="WAV audio file (5-10 seconds, 16kHz)")
) -> JSONResponse:
    """
    Transcribe audio file to text using NeMo ASR model
    
    Args:
        file: WAV audio file upload
        
    Returns:
        JSON response with transcription text
        
    Raises:
        HTTPException: For invalid files or processing errors
    """
    start_time = time.time()
    temp_file_path = None
    
    try:
        # Update request statistics
        stats["total_requests"] += 1
        
        # Validate file type
        if not file.content_type or "audio" not in file.content_type.lower():
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Only audio files are supported."
            )
        
        if not file.filename or not file.filename.lower().endswith('.wav'):
            raise HTTPException(
                status_code=400,
                detail="Only .wav files are supported"
            )
        
        # Check file size (max 10MB)
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 10MB."
            )
        
        logger.info(f"Processing file: {file.filename}, size: {len(content)} bytes")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp.write(content)
            temp.flush()
            temp_file_path = temp.name
        
        # Validate audio duration
        duration_valid = await validate_audio(temp_file_path)
        if not duration_valid:
            raise HTTPException(
                status_code=400,
                detail="Audio duration must be between 5-10 seconds"
            )
        
        # Perform transcription
        transcription_text = await transcribe_audio(temp_file_path)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Update statistics
        stats["successful_transcriptions"] += 1
        stats["average_processing_time"] = (
            (stats["average_processing_time"] * (stats["successful_transcriptions"] - 1) + processing_time) 
            / stats["successful_transcriptions"]
        )
        
        logger.info(f"Transcription completed in {processing_time:.2f}s: {transcription_text}")
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_file_path)
        
        return JSONResponse(
            content={
                "transcription": transcription_text,
                "filename": file.filename,
                "processing_time": round(processing_time, 2),
                "model": "stt_hi_conformer_ctc_medium",
                "status": "success"
            }
        )
        
    except HTTPException:
        stats["failed_transcriptions"] += 1
        if temp_file_path:
            background_tasks.add_task(cleanup_temp_file, temp_file_path)
        raise
    except Exception as e:
        stats["failed_transcriptions"] += 1
        if temp_file_path:
            background_tasks.add_task(cleanup_temp_file, temp_file_path)
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during transcription: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )