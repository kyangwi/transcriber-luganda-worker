import os
import sys
import time
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

if not API_KEY or not ENDPOINT_ID:
    print("❌ Error: RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID must be set in .env")
    sys.exit(1)

audio_file = sys.argv[1] if len(sys.argv) > 1 else "audio/sample_en.mp3"

if not os.path.exists(audio_file):
    print(f"❌ Error: Test audio file not found at '{audio_file}'. Pass a valid file path as an argument.")
    sys.exit(1)

print(f"🚀 Testing RunPod Serverless Endpoint [{ENDPOINT_ID}] with audio: {audio_file}")

with open(audio_file, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

payload = {
    "input": {
        "action": "transcribe",
        "audio_base64": audio_b64,
        "language": "Luganda" if "lg" in audio_file.lower() else "en",
        "diarize": True
    }
}

print(f"📡 Sending request to {url} ...")
t0 = time.time()
resp = requests.post(url, json=payload, headers=headers, timeout=120)
print(f"⏱️ Response received in {time.time() - t0:.2f}s (Status: {resp.status_code})")

if resp.status_code != 200:
    print(f"❌ Failed: {resp.text}")
    sys.exit(1)

data = resp.json()
job_status = data.get("status")
print(f"Job Status: {job_status}")

if job_status == "COMPLETED":
    out = data.get("output", {})
    print("\n✅ Transcription Output:")
    print("-" * 50)
    print(out.get("formatted") or out.get("text"))
    print("-" * 50)
elif job_status in ("IN_QUEUE", "IN_PROGRESS"):
    job_id = data.get("id")
    print(f"⏳ Job queued/in-progress (ID: {job_id}). Polling status...")
    status_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}"
    for _ in range(30):
        time.sleep(3)
        poll = requests.get(status_url, headers=headers).json()
        print(f"  Status: {poll.get('status')}")
        if poll.get("status") == "COMPLETED":
            out = poll.get("output", {})
            print("\n✅ Transcription Output:")
            print("-" * 50)
            print(out.get("formatted") or out.get("text"))
            print("-" * 50)
            break
        elif poll.get("status") in ("FAILED", "CANCELLED"):
            print("❌ Job failed:", poll)
            break
else:
    print("Response:", data)
