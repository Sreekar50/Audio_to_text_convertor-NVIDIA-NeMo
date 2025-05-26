"""
NeMo ASR FastAPI Application Package

This package provides a production-ready FastAPI service for Hindi speech recognition
using NVIDIA NeMo's optimized ONNX models.

Modules:
    main: FastAPI application with REST endpoints
    inference: ONNX-based ASR inference engine
    utils: Audio processing and utility functions
    test: Comprehensive test client

Features:
    - Async-compatible processing
    - ONNX Runtime optimization
    - Comprehensive error handling
    - Production-ready monitoring
"""

__version__ = "1.0.0"
__author__ = "ASR Team"
__description__ = "NeMo ASR FastAPI Service"

# Package metadata
__all__ = [
    "main",
    "inference", 
    "utils",
    "test"
]