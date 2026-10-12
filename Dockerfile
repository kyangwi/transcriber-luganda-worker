# Use PyTorch runtime image with CUDA 12.1 GPU support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install required system packages (ffmpeg and libsndfile are essential for audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies for RunPod, Whisper, and Diarization
RUN pip install --no-cache-dir \
    runpod==1.7.7 \
    faster-whisper \
    transformers \
    accelerate \
    funasr \
    modelscope \
    scikit-learn \
    soundfile \
    pydub \
    numpy \
    scipy \
    huggingface_hub

# Set working directory inside container
WORKDIR /app

# Copy serverless handler code
COPY rp_handler.py /app/rp_handler.py

# Default environment variables
ENV WHISPER_MODEL="allandclive/whisper-medium-luganda"
ENV FASTER_WHISPER_MODEL="medium"

# Run the RunPod Serverless handler
CMD ["python", "-u", "/app/rp_handler.py"]
