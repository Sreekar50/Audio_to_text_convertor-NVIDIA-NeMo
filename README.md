# NeMo ASR FastAPI Service

A production-ready FastAPI-based Automatic Speech Recognition (ASR) service using NVIDIA NeMo's Hindi Conformer CTC model. This service provides high-performance, async-compatible speech-to-text transcription for Hindi audio files.


### Prerequisites

- Docker and Docker Compose
- Python 3.12+ (for local development)
- Audio files in WAV format (16kHz, 5-10 seconds)

### Using Docker

1. **Clone the repository**
```bash
git clone https://github.com/Sreekar50/Audio_to_text_convertor-NVIDIA-NeMo-.git
cd fastapi-asr-nemo
```

2. **Generate onxx file**
```bash
# first place the .nemo file inside model folder and then run following script from model folder
python export_to_onnx.py
```

3. **Build the Docker image**
```bash
docker build -t fastapi-asr .
```

4. **Run the container**
```bash
docker run -p 8000:8000 fastapi-asr
```

5. **Test the service**
```bash
python app/test.py
```
Or using curl request
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -F "file=@/path/to/your/audio/sample.wav;type=audio/wav"
```

## API Endpoints

### Core Endpoints

#### `POST /transcribe`
Transcribe audio file to text.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: WAV audio file (5-10 seconds, 16kHz)

**Response:**
```json
{
  "transcription": "transcribed text in Hindi",
  "filename": "sample.wav",
  "processing_time": 2.34,
  "model": "stt_hi_conformer_ctc_medium",
  "status": "success"
}
```

#### `GET /`
Service information and usage guide.

#### `GET /health`
Health check endpoint for monitoring.

#### `GET /stats`
Service statistics including request counts and processing times.


## Usage Examples

### Using cURL

```bash
# Basic transcription
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -F "file=@/path/to/your/audio/sample.wav;type=audio/wav"

# Health check
curl -X GET "http://localhost:8000/health"

# Service statistics
curl -X GET "http://localhost:8000/stats"
```

## Configuration

### Environment Variables

- `ONNX_THREADS`: Number of ONNX inference threads (default: 2)
- `MAX_FILE_SIZE`: Maximum upload file size in MB (default: 10)
- `LOG_LEVEL`: Logging level (default: INFO)

### Audio Requirements

- **Format**: WAV
- **Sample Rate**: 16kHz
- **Duration**: 5-10 seconds
- **Channels**: Mono preferred
- **Bit Depth**: 16-bit

## Performance Optimization

### ONNX Runtime Configuration
- CPU optimization with multi-threading
- Graph optimization enabled
- Memory pool optimization

### Async Processing
- Non-blocking I/O operations
- Thread pool for CPU-intensive tasks
- Efficient resource cleanup

### Docker Optimization
- Multi-stage build (if applicable)
- Minimal base image (python:3.12-slim)
- Optimized layer caching

## Troubleshooting

### Common Issues

1. **Model files not found**
   ```
   Error: Model file not found: model/stt_hi_conformer_ctc_medium.onnx
   ```
   - Ensure model files are present in the `model/` directory
   - Run `export_to_onnx.py` to generate ONNX model

2. **Audio duration validation fails**
   ```
   Error: Audio duration must be between 5-10 seconds
   ```
   - Check audio file duration with `ffprobe` or similar tool
   - Trim or pad audio to meet requirements

3. **Memory issues**
   ```
   Error: Out of memory during inference
   ```
   - Reduce ONNX thread count
   - Use smaller batch sizes
   - Increase Docker memory limits

4. **Port already in use**
   ```
   Error: Port 8000 is already in use
   ```
   - Use different port: `docker run -p 8001:8000 nemo-asr-service`
   - Kill existing processes using the port

### Logging

Logs are available at different levels:
- `INFO`: General service information
- `DEBUG`: Detailed processing information
- `ERROR`: Error conditions
- `WARNING`: Non-critical issues

View logs:
```bash
# Docker logs
docker logs <container-id>

```

## Testing

### Automated Tests

Run the comprehensive test suite:
```bash
python app/test.py
```

Test coverage includes:
- Health check endpoint
- Root endpoint
- Valid transcription requests
- Invalid file type handling
- Statistics endpoint
- Error handling

### Manual Testing

1. **Prepare test audio**: 5-10 second WAV files at 16kHz
2. **Start service**: `docker run -p 8000:8000 nemo-asr-service`
3. **Test transcription**: Use cURL or Python requests
4. **Verify response**: Check transcription quality and response format

## Production Deployment

### Docker Compose 

```yaml
version: '3.8'
services:
  nemo-asr:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nemo-asr-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nemo-asr
  template:
    metadata:
      labels:
        app: nemo-asr
    spec:
      containers:
      - name: nemo-asr
        image: nemo-asr-service:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### Load Balancing

For high-traffic scenarios:
- Use nginx or HAProxy for load balancing
- Deploy multiple container instances
- Implement horizontal pod autoscaling in Kubernetes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request


## Acknowledgments

- NVIDIA NeMo team for the pre-trained ASR models
- FastAPI community for the excellent web framework
- ONNX Runtime team for optimization tools
