import onnxruntime as ort
import numpy as np
from .utils import preprocess_audio, load_token_map, decode_tokens
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Optional

logger = logging.getLogger(__name__)

class ASRInference:
    """Async-compatible ASR inference class using ONNX Runtime"""
    
    def __init__(self, model_path: str = "model/stt_hi_conformer_ctc_medium.onnx"):
        self.model_path = model_path
        self.session: Optional[ort.InferenceSession] = None
        self.token_map: Optional[dict] = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize ONNX model and token map"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Configure ONNX Runtime for optimal performance
            session_options = ort.SessionOptions()
            session_options.inter_op_num_threads = 2
            session_options.intra_op_num_threads = 2
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            providers = ['CPUExecutionProvider']
            if ort.get_available_providers():
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers:
                    providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )
            
            logger.info(f"Model loaded successfully with providers: {self.session.get_providers()}")
            logger.info(f"Model inputs: {[inp.name for inp in self.session.get_inputs()]}")
            logger.info(f"Model outputs: {[out.name for out in self.session.get_outputs()]}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
    
    async def _run_inference(self, input_tensor: np.ndarray, input_length: np.ndarray) -> np.ndarray:
        """Run inference asynchronously"""
        try:
            # Prepare inputs
            ort_inputs = {
                self.session.get_inputs()[0].name: input_tensor,
                self.session.get_inputs()[1].name: input_length,
            }
            
            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            ort_outputs = await loop.run_in_executor(
                self.executor, 
                self.session.run, 
                None, 
                ort_inputs
            )
            
            return ort_outputs[0]
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise
    
    async def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            input_tensor = await preprocess_audio(audio_path)
            input_length = np.array([input_tensor.shape[2]], dtype=np.int64)
            
            logger.info(f"Input tensor shape: {input_tensor.shape}")
            
            logits = await self._run_inference(input_tensor, input_length)
            
            # Decode predictions
            decoded_ids = np.argmax(logits, axis=-1).squeeze()
            
            if self.token_map is None:
                self.token_map = await load_token_map("model/tokens.txt")
            
            # Convert tokens to text
            transcription = decode_tokens(decoded_ids, self.token_map)
            
            # Post-process transcription
            transcription = self._post_process_text(transcription)
            
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise
    
    def _post_process_text(self, text: str) -> str:
        """Post-process transcribed text"""
        text = ' '.join(text.split())
        
        if not text.strip():
            return "[No speech detected]"
        
        return text.strip()
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


_inference_instance: Optional[ASRInference] = None

def get_inference_instance() -> ASRInference:
    """Get singleton inference instance"""
    global _inference_instance
    if _inference_instance is None:
        _inference_instance = ASRInference()
    return _inference_instance

async def transcribe_audio(audio_path: str) -> str:
    """
    Main transcription function
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Transcribed text
    """
    inference = get_inference_instance()
    return await inference.transcribe(audio_path)