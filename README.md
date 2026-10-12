# Deploying Speech-to-Text & Diarization Models to RunPod Serverless

This guide walks you through building, pushing, and deploying the Whisper and CAM++ Diarization models to a **RunPod Serverless Endpoint**.

---

## Architecture Overview

- **RunPod Serverless**: Hosts the GPU-heavy models (`ElizabethMwangi/whisper-large-v3-luganda-nss-v22`, `faster-whisper`, and Alibaba `FunASR` / `CAM++` speaker voiceprint diarization). Scales to 0 when idle ($0/hr cost when not transcribing).
- **Railway**: Hosts the Django web application (fast, lightweight, connects via RunPod REST API).

---

## Step 1: Build the Docker Image

Open a terminal in the `runpod_deployment/` directory. Replace `yourusername` with your Docker Hub (or GitHub Packages) username:

```bash
docker build -t yourusername/luganda-whisper-serverless:latest .
```

---

## Step 2: Push the Image to Docker Hub (or GHCR)

Log in and push the image:

```bash
docker login
docker push yourusername/luganda-whisper-serverless:latest
```

*(Alternatively, push to GitHub Container Registry using the included `.github/workflows/build.yml` workflow).*

---

## Step 3: Create a RunPod Serverless Endpoint

1. Go to the [RunPod Console](https://www.runpod.io/console/serverless) and log in.
2. Click **Serverless** -> **Endpoints** -> **New Endpoint**.
3. Configure the endpoint:
   - **Endpoint Name**: `luganda-transcriber-whisper`
   - **Container Image**: `yourusername/luganda-whisper-serverless:latest`
   - **Container Registry Credentials**: Select if using a private registry, or leave empty if public.
   - **Select GPU Types**: 
     - Recommended: `NVIDIA RTX 4090` (24GB VRAM) or `NVIDIA RTX 3090` (24GB VRAM) or `NVIDIA A40` / `A10G`.
     - *Minimum VRAM*: 12GB (16GB+ recommended for Whisper-large-v3).
   - **Container Disk**: `20 GB` (sufficient for caching models).
   - **Volume Disk**: Optional (you can attach a network volume if you want permanent model caching across cold starts).
   - **Min Provisioned Workers**: `0` *(ensures $0 cost when idle)*.
   - **Max Workers**: `3` (or more based on desired concurrency).
   - **Idle Timeout**: `180` to `300` seconds *(keeps worker alive for consecutive transcriptions)*.
4. Click **Create Endpoint**.
5. Note your **Endpoint ID** (e.g., `abc123xyz456`) and your **RunPod API Key** (from Settings -> API Keys).

---

## Step 4: Test the Endpoint

Run the test script locally:

```bash
python runpod_deployment/test_runpod.py audio/sample_en.mp3
```

---

## Step 5: Connect to Django (Railway or Local)

In your Django `.env` (or Railway environment variables):

```env
RUNPOD_API_KEY="your_runpod_api_key_here"
RUNPOD_ENDPOINT_ID="your_runpod_endpoint_id_here"
```

The Django application will automatically route all transcription, diarization, and live streaming requests to your RunPod Serverless GPU worker!
