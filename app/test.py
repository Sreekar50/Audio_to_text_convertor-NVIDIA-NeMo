import httpx
import asyncio
import os
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASRTestClient:
    """Comprehensive test client for ASR service"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    async def test_health_check(self):
        """Test health check endpoint"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health")
                logger.info(f"Health Check - Status: {response.status_code}")
                logger.info(f"Response: {response.json()}")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    async def test_root_endpoint(self):
        """Test root endpoint"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/")
                logger.info(f"Root Endpoint - Status: {response.status_code}")
                logger.info(f"Response: {response.json()}")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Root endpoint test failed: {str(e)}")
            return False
    
    async def test_transcription(self, audio_file_path: str, expected_duration: float = None):
        """Test transcription endpoint"""
        try:
            if not os.path.exists(audio_file_path):
                logger.error(f"Audio file not found: {audio_file_path}")
                return False
            
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                with open(audio_file_path, "rb") as f:
                    files = {"file": (os.path.basename(audio_file_path), f, "audio/wav")}
                    response = await client.post(f"{self.base_url}/transcribe", files=files)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Transcription Test - File: {os.path.basename(audio_file_path)}")
            logger.info(f"Status Code: {response.status_code}")
            logger.info(f"Processing Time: {processing_time:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Transcription: {result.get('transcription', 'N/A')}")
                logger.info(f"Model Processing Time: {result.get('processing_time', 'N/A')}s")
                
                self.results.append({
                    "file": os.path.basename(audio_file_path),
                    "status": "success",
                    "transcription": result.get('transcription'),
                    "processing_time": processing_time,
                    "model_time": result.get('processing_time')
                })
                return True
            else:
                logger.error(f"Transcription failed: {response.text}")
                self.results.append({
                    "file": os.path.basename(audio_file_path),
                    "status": "failed",
                    "error": response.text,
                    "processing_time": processing_time
                })
                return False
                
        except Exception as e:
            logger.error(f"Transcription test failed: {str(e)}")
            return False
    
    async def test_invalid_file_type(self):
        """Test with invalid file type"""
        try:
            # Create a fake text file
            test_content = b"This is not an audio file"
            
            async with httpx.AsyncClient() as client:
                files = {"file": ("test.txt", test_content, "text/plain")}
                response = await client.post(f"{self.base_url}/transcribe", files=files)
            
            logger.info(f"Invalid File Test - Status: {response.status_code}")
            logger.info(f"Response: {response.text}")
            
            # Should return 400 for invalid file type
            return response.status_code == 400
            
        except Exception as e:
            logger.error(f"Invalid file test failed: {str(e)}")
            return False
    
    async def test_stats_endpoint(self):
        """Test statistics endpoint"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/stats")
                logger.info(f"Stats Endpoint - Status: {response.status_code}")
                logger.info(f"Response: {response.json()}")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Stats test failed: {str(e)}")
            return False
    
    async def run_comprehensive_tests(self, audio_files: list = None):
        """Run all tests"""
        logger.info("Starting comprehensive ASR service tests...")
        
        # Default test files
        if audio_files is None:
            audio_files = [
                "audio/sample.wav",
                "audio/test1.wav",
                "audio/test2.wav"
            ]
        
        test_results = {}
        
        # Test 1: Health check
        test_results["health_check"] = await self.test_health_check()
        
        # Test 2: Root endpoint
        test_results["root_endpoint"] = await self.test_root_endpoint()
        
        # Test 3: Valid transcriptions
        for audio_file in audio_files:
            if os.path.exists(audio_file):
                test_name = f"transcribe_{os.path.basename(audio_file)}"
                test_results[test_name] = await self.test_transcription(audio_file)
        
        # Test 4: Invalid file type
        test_results["invalid_file_type"] = await self.test_invalid_file_type()
        
        # Test 5: Stats endpoint
        test_results["stats_endpoint"] = await self.test_stats_endpoint()
        
        # Summary
        passed = sum(1 for result in test_results.values() if result)
        total = len(test_results)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"TEST SUMMARY: {passed}/{total} tests passed")
        logger.info(f"{'='*50}")
        
        for test_name, result in test_results.items():
            status = "PASS" if result else "FAIL"
            logger.info(f"{test_name}: {status}")
        
        if self.results:
            logger.info(f"\nTranscription Results:")
            for result in self.results:
                if result["status"] == "success":
                    logger.info(f"✓ {result['file']}: {result['transcription']}")
                else:
                    logger.info(f"✗ {result['file']}: {result.get('error', 'Unknown error')}")
        
        return test_results

async def test_with_curl_command():
    """Test with curl-like command"""
    logger.info("Testing with curl-like command...")
    
    # This simulates the curl command from the assignment
    audio_file = "audio/sample.wav"
    
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found: {audio_file}")
        return
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(audio_file, "rb") as f:
                files = {"file": (os.path.basename(audio_file), f, "audio/wav")}
                headers = {"accept": "application/json"}
                
                response = await client.post(
                    "http://localhost:8000/transcribe",
                    files=files,
                    headers=headers
                )
        
        logger.info(f"Curl Test Results:")
        logger.info(f"Status Code: {response.status_code}")
        logger.info(f"Headers: {dict(response.headers)}")
        logger.info(f"Response: {response.text}")
        
    except Exception as e:
        logger.error(f"Curl test failed: {str(e)}")

async def main():
    """Main test function"""
    # Initialize test client
    client = ASRTestClient()
    
    # Test server availability
    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get("http://localhost:8000/health", timeout=5.0)
            logger.info("Server is running and accessible")
    except Exception as e:
        logger.error(f"Server not accessible: {str(e)}")
        logger.error("Please ensure the FastAPI server is running on localhost:8000")
        return
    
    # Run comprehensive tests
    await client.run_comprehensive_tests()
    
    # Test curl command equivalent
    await test_with_curl_command()

if __name__ == "__main__":
    asyncio.run(main())