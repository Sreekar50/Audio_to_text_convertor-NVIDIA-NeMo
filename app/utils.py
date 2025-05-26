import librosa
import numpy as np
import asyncio
import aiofiles
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Thread pool for CPU-intensive operations
_executor = ThreadPoolExecutor(max_workers=4)

async def validate_audio(path: str) -> bool:
    """
    Validate audio file duration and format
    
    Args:
        path: Path to audio file
        
    Returns:
        True if audio is valid (5-10 seconds), False otherwise
    """
    try:
        # Load audio file asynchronously
        y, sr = await asyncio.get_event_loop().run_in_executor(
            _executor, 
            lambda p: librosa.load(p, sr=16000), 
            path
        )
        
        duration = librosa.get_duration(y=y, sr=sr)
        logger.info(f"Audio duration: {duration:.2f} seconds")
        
        is_valid = 5.0 <= duration <= 10.0
        
        if not is_valid:
            logger.warning(f"Invalid audio duration: {duration:.2f}s (expected 5-10s)")
        
        return is_valid
        
    except Exception as e:
        logger.error(f"Audio validation failed: {str(e)}")
        return False

async def preprocess_audio(path: str) -> np.ndarray:
    """
    Preprocess audio file for ASR model inference
    
    Args:
        path: Path to audio file
        
    Returns:
        Preprocessed audio features as numpy array
    """
    try:
        sample_rate = 16000
        
        # Load audio asynchronously
        samples, _ = await asyncio.get_event_loop().run_in_executor(
            _executor, 
            lambda p: librosa.load(p, sr=sample_rate), 
            path
        )
        
        logger.info(f"Loaded audio: {len(samples)} samples at {sample_rate}Hz")
        
        # Normalize and scale samples
        samples = samples * 32768.0  
        
        # Extract mel-spectrogram features
        mel_spec = librosa.feature.melspectrogram(
            y=samples,
            sr=sample_rate,
            n_mels=80,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window='hann',
            center=True,
            pad_mode='constant'
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize features
        mean = log_mel_spec.mean(axis=1, keepdims=True)
        std = log_mel_spec.std(axis=1, keepdims=True)
        normalized_features = (log_mel_spec - mean) / (std + 1e-8)
        
        # Add batch dimension: (1, n_mels, time_steps)
        features = np.expand_dims(normalized_features, axis=0)
        
        logger.info(f"Preprocessed features shape: {features.shape}")
        
        return features.astype(np.float32)
        
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {str(e)}")
        raise

async def load_token_map(token_file: str = "model/tokens.txt") -> Dict[int, str]:
    """
    Load token mapping from file
    
    Args:
        token_file: Path to tokens file
        
    Returns:
        Dictionary mapping token IDs to strings
    """
    try:
        if not os.path.exists(token_file):
            logger.warning(f"Token file not found: {token_file}")
            return {}
        
        token_map = {}
        
        async with aiofiles.open(token_file, 'r', encoding='utf-8') as f:
            async for line in f:
                line = line.strip()
                if line:
                    try:
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            token, idx = parts
                            token_map[int(idx)] = token
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Skipping invalid token line: {line}")
        
        logger.info(f"Loaded {len(token_map)} tokens from {token_file}")
        return token_map
        
    except Exception as e:
        logger.error(f"Failed to load token map: {str(e)}")
        return {}

def decode_tokens(token_ids: np.ndarray, token_map: Dict[int, str]) -> str:
    """
    Decode token IDs to text using CTC decoding
    
    Args:
        token_ids: Array of token IDs
        token_map: Mapping from token IDs to strings
        
    Returns:
        Decoded text string
    """
    try:
        if len(token_map) == 0:
            logger.warning("Empty token map provided")
            return ""
        
        # Get blank token ID(usually the last token)
        blank_token_id = max(token_map.keys()) if token_map else 128
        
        # CTC decoding: remove blanks and consecutive duplicates
        decoded_tokens = []
        prev_token = None
        
        for token_id in token_ids:
            token_id = int(token_id)
            
            # Skip blank tokens
            if token_id == blank_token_id:
                prev_token = None
                continue
            
            # Skip consecutive duplicates
            if token_id == prev_token:
                continue
            
            # Add token if it exists in the map
            if token_id in token_map:
                decoded_tokens.append(token_map[token_id])
            else:
                logger.warning(f"Unknown token ID: {token_id}")
            
            prev_token = token_id
        
        # Join tokens to form text
        text = ''.join(decoded_tokens)
        
        logger.info(f"Decoded {len(token_ids)} tokens to: '{text}'")
        return text
        
    except Exception as e:
        logger.error(f"Token decoding failed: {str(e)}")
        return ""

async def cleanup_temp_file(file_path: str) -> None:
    """
    Cleanup temporary file asynchronously
    
    Args:
        file_path: Path to temporary file
    """
    try:
        if file_path and os.path.exists(file_path):
            await asyncio.get_event_loop().run_in_executor(
                _executor,
                os.unlink,
                file_path
            )
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {str(e)}")

def get_audio_info(file_path: str) -> Tuple[float, int]:
    """
    Get audio file information
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Tuple of (duration, sample_rate)
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        return duration, sr
    except Exception as e:
        logger.error(f"Failed to get audio info: {str(e)}")
        return 0.0, 0

# Cleanup executor on module unload
import atexit

def _cleanup_executor():
    """Cleanup thread pool executor"""
    global _executor
    if _executor:
        _executor.shutdown(wait=True)

atexit.register(_cleanup_executor)