#!/usr/bin/env python3
"""
Setup validation script for FastAPI ASR Service
Checks if all required components are properly configured
"""

import os
import sys
import asyncio
from pathlib import Path


def check_file_exists(file_path, description):
    """Check if a required file exists"""
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        print(f"✓ {description}: {file_path} ({file_size:,} bytes)")
        return True
    else:
        print(f"✗ {description}: {file_path} - NOT FOUND")
        return False


def check_directory_structure():
    """Validate project directory structure"""
    print("=== Directory Structure Check ===")
    
    required_files = [
        ("app/__init__.py", "App package init"),
        ("app/main.py", "Main FastAPI application"),
        ("app/inference.py", "Model inference module"),
        ("app/utils.py", "Utility functions"),
        ("app/startup.py", "Startup configuration"),
        ("model/stt_hi_conformer_ctc_medium.onnx", "ONNX model file"),
        ("model/tokens.txt", "Token vocabulary"),
        ("deploy_requirements.txt", "Production dependencies"),
        ("Dockerfile", "Docker configuration"),
        ("README.md", "Documentation"),
        ("Description.md", "Development documentation"),
    ]
    
    all_good = True
    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            all_good = False
    
    return all_good


def check_python_imports():
    """Check if all required Python packages can be imported"""
    print("\n=== Python Dependencies Check ===")
    
    required_packages = [
        ("fastapi", "FastAPI framework"),
        ("uvicorn", "ASGI server"),
        ("onnxruntime", "ONNX Runtime"),
        ("librosa", "Audio processing"),
        ("numpy", "Numerical computing"),
        ("aiofiles", "Async file operations"),
        ("pydantic", "Data validation"),
    ]
    
    all_good = True
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}: {description}")
        except ImportError:
            print(f"✗ {package}: {description} - NOT AVAILABLE")
            all_good = False
    
    return all_good


def check_model_files():
    """Validate model files"""
    print("\n=== Model Files Check ===")
    
    model_dir = Path("model")
    if not model_dir.exists():
        print("✗ Model directory not found")
        return False
    
    onnx_file = model_dir / "stt_hi_conformer_ctc_medium.onnx"
    tokens_file = model_dir / "tokens.txt"
    
    all_good = True
    
    # Check ONNX model
    if onnx_file.exists():
        size_mb = onnx_file.stat().st_size / (1024 * 1024)
        print(f"✓ ONNX model: {size_mb:.1f} MB")
    else:
        print("✗ ONNX model file missing")
        all_good = False
    
    # Check tokens file
    if tokens_file.exists():
        with open(tokens_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"✓ Tokens file: {len(lines)} tokens")
    else:
        print("✗ Tokens file missing")
        all_good = False
    
    return all_good


async def test_audio_processing():
    """Test audio processing pipeline"""
    print("\n=== Audio Processing Test ===")
    
    try:
        from app.utils import preprocess_audio, load_token_map
        
        # Test token map loading
        if os.path.exists("model/tokens.txt"):
            token_map = await load_token_map()
            print(f"✓ Token map loaded: {len(token_map)} tokens")
        else:
            print("✗ Cannot test token loading - tokens.txt missing")
            return False
        
        # Test audio preprocessing (if sample audio exists)
        sample_audio = "audio/sample.wav"
        if os.path.exists(sample_audio):
            features = await preprocess_audio(sample_audio)
            print(f"✓ Audio preprocessing: shape {features.shape}")
        else:
            print("ℹ Sample audio not found - skipping preprocessing test")
        
        return True
        
    except Exception as e:
        print(f"✗ Audio processing test failed: {e}")
        return False


async def test_model_loading():
    """Test ONNX model loading"""
    print("\n=== Model Loading Test ===")
    
    try:
        from app.inference import initialize_model
        
        await initialize_model()
        print("✓ ONNX model loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False


def print_usage_instructions():
    """Print usage instructions"""
    print("\n=== Usage Instructions ===")
    print("If all checks passed, you can start the service:")
    print()
    print("Local development:")
    print("  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
    print()
    print("Docker deployment:")
    print("  docker build -t fastapi-asr .")
    print("  docker run -p 8000:8000 fastapi-asr")
    print()
    print("Test the service:")
    print("  python test.py")
    print()
    print("API endpoint:")
    print('  curl -X POST "http://localhost:8000/transcribe" \\')
    print('    -H "Content-Type: multipart/form-data" \\')
    print('    -F "file=@audio/sample.wav"')


async def main():
    """Run all validation checks"""
    print("FastAPI ASR Service - Setup Validation")
    print("=" * 50)
    
    checks = [
        ("Directory Structure", check_directory_structure()),
        ("Python Dependencies", check_python_imports()),
        ("Model Files", check_model_files()),
        ("Audio Processing", await test_audio_processing()),
        ("Model Loading", await test_model_loading()),
    ]
    
    print("\n" + "=" * 50)
    print("Validation Summary")
    print("=" * 50)
    
    all_passed = True
    for check_name, result in checks:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{check_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print(f"\n All validation checks passed!")
        print_usage_instructions()
    else:
        print(f"\n  Some validation checks failed. Please fix the issues above.")
        print("Refer to README.md for setup instructions.")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)