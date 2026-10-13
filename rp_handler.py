import os
import sys
import gc
import re
import time
import uuid
import base64
import tempfile
import urllib.request
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch

# ====== FIX AIOHTTP / BROTLI DECOMPRESSOR COMPATIBILITY ======
try:
    import aiohttp
    _orig_aiohttp_session_init = aiohttp.ClientSession.__init__

    def _safe_aiohttp_session_init(self, *args, **kwargs):
        headers = kwargs.get("headers")
        if headers is None:
            headers = {}
        elif not isinstance(headers, dict):
            headers = dict(headers)
        headers.setdefault("Accept-Encoding", "gzip, deflate")
        kwargs["headers"] = headers
        return _orig_aiohttp_session_init(self, *args, **kwargs)

    aiohttp.ClientSession.__init__ = _safe_aiohttp_session_init
except Exception:
    pass

try:
    import brotli
    if hasattr(brotli, "Decompressor") and not hasattr(brotli.Decompressor, "is_finished"):
        brotli.Decompressor.is_finished = lambda self: False
except Exception:
    pass

try:
    import brotlicffi
    if hasattr(brotlicffi, "Decompressor") and not hasattr(brotlicffi.Decompressor, "is_finished"):
        brotlicffi.Decompressor.is_finished = lambda self: False
except Exception:
    pass

try:
    import runpod
except ImportError:
    runpod = None

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s: %(message)s')
logger = logging.getLogger("runpod_transcriber")

# ====== CONFIGURATION ======
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
LUGANDA_MODEL_ID = os.getenv("WHISPER_MODEL", "allandclive/whisper-medium-luganda")
FASTER_WHISPER_SIZE = os.getenv("FASTER_WHISPER_MODEL", "medium")
HF_TOKEN = os.getenv("HF_TOKEN", None)

logger.info(f"Target Device: {DEVICE} | Torch DType: {TORCH_DTYPE}")

# Global model references
luganda_pipeline = None
faster_whisper_model = None
vad_model = None
cam_model = None


# ====== AUDIO HELPERS ======
def decode_input_audio(audio_base64: Optional[str] = None, audio_url: Optional[str] = None) -> str:
    """Decodes base64 audio or downloads audio from URL into a temporary file on disk."""
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    if audio_base64:
        if "," in audio_base64:
            audio_base64 = audio_base64.split(",", 1)[1]
        raw_bytes = base64.b64decode(audio_base64)
        with open(temp_path, "wb") as f:
            f.write(raw_bytes)
        return temp_path

    if audio_url:
        logger.info(f"Downloading audio from URL: {audio_url[:80]}...")
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(audio_url, headers=headers)
        with urllib.request.urlopen(req, timeout=180) as resp, open(temp_path, "wb") as f:
            shutil.copyfileobj(resp, f)
        logger.info(f"Audio downloaded successfully ({os.path.getsize(temp_path)/(1024*1024):.1f} MB).")
        return temp_path

    raise ValueError("Neither audio_base64 nor audio_url was provided.")


def load_audio_16k_mono(audio_path: str) -> Tuple[np.ndarray, int]:
    """Loads any audio file as 16kHz mono numpy array."""
    if not audio_path or not os.path.exists(audio_path):
        return np.array([], dtype=np.float32), 16000

    try:
        import soundfile as sf
        audio_data, sr = sf.read(audio_path)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        if sr != 16000:
            from scipy.signal import resample
            num_samples = int(len(audio_data) * 16000 / sr)
            audio_data = resample(audio_data, num_samples).astype(np.float32)
        return audio_data.astype(np.float32), 16000
    except Exception as e:
        logger.debug(f"soundfile read failed: {e}. Falling back to pydub...")

    try:
        from pydub import AudioSegment
        sound = AudioSegment.from_file(audio_path)
        sound = sound.set_frame_rate(16000).set_channels(1)
        samples = np.array(sound.get_array_of_samples()).astype(np.float32) / 32768.0
        return samples, 16000
    except Exception as e2:
        logger.error(f"Failed to load audio with pydub: {e2}")
        return np.array([], dtype=np.float32), 16000


def format_seconds_to_timestamp(seconds: float) -> str:
    total_sec = max(0, int(seconds))
    hrs = total_sec // 3600
    mins = (total_sec % 3600) // 60
    secs = total_sec % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def filter_streaming_hallucinations(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'[\u2669-\u266f\U0001f3b5-\U0001f3bc♪♫♬♭♮♯🎼🎶🎵]', '', text)
    cleaned = re.sub(r'\[\s*["\'].*?["\']\s*\]', '', cleaned)
    cleaned = re.sub(r'\[(?:music|applause|laughter|bgm|silence|cheering|singing|audio|sound|noise)\]', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\((?:music|applause|laughter|bgm|silence|cheering|singing|audio|sound|noise)\)', '', cleaned, flags=re.IGNORECASE)

    norm = re.sub(r'[^a-zA-Z\s]', '', cleaned).lower().strip()
    words = norm.split()
    PHANTOM_PATTERNS = {
        "yeah", "yeha", "yep", "yes", "mhm", "uhhuh", "uh huh", "hello", "hi", "hey",
        "thank you", "thanks", "thanks for watching", "bye", "bye bye", "you", "the", "so", "ok"
    }
    phrase = " ".join(words)
    if not phrase or phrase in PHANTOM_PATTERNS:
        return ""
    if len(words) <= 1 and (len(phrase) <= 3 or phrase in PHANTOM_PATTERNS):
        return ""

    filler_pattern = r'(?i)\b(u+h+|u+m+|u+n+h+|e+h+|e+r+|a+h+|h+m+|m+h+m+|u+h+m+|u+m+h+)\b'
    cleaned = re.sub(filler_pattern, '', cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip()


LANGUAGE_CODE_MAP = {
    "english": "en",
    "swahili": "sw",
    "french": "fr",
    "spanish": "es",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "russian": "ru",
    "chinese": "zh",
    "mandarin": "zh",
    "japanese": "ja",
    "korean": "ko",
    "arabic": "ar",
    "hindi": "hi",
    "lingala": "ln",
    "yoruba": "yo",
    "somali": "so",
    "amharic": "am",
    "hausa": "ha",
    "oromo": "om",
    "dutch": "nl",
    "turkish": "tr",
    "polish": "pl",
    "ukrainian": "uk",
    "vietnamese": "vi",
}


def normalize_faster_whisper_lang(lang: Optional[str]) -> Optional[str]:
    if not lang or str(lang).lower().strip() in ("auto", "none", ""):
        return None
    cleaned = str(lang).lower().strip()
    return LANGUAGE_CODE_MAP.get(cleaned, cleaned)


# ====== MODEL LOADERS ======
def get_faster_whisper():
    global faster_whisper_model
    if faster_whisper_model is None:
        from faster_whisper import WhisperModel
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Faster Whisper ({FASTER_WHISPER_SIZE}) on {dev} ({compute_type})...")
        faster_whisper_model = WhisperModel(FASTER_WHISPER_SIZE, device=dev, compute_type=compute_type)
        logger.info("Faster Whisper loaded successfully.")
    return faster_whisper_model


def get_luganda_pipeline():
    global luganda_pipeline
    if luganda_pipeline is None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        logger.info(f"Loading Luganda Whisper model [{LUGANDA_MODEL_ID}] on {DEVICE}...")
        load_kwargs = {
            "torch_dtype": TORCH_DTYPE,
            "low_cpu_mem_usage": True,
        }
        if HF_TOKEN:
            load_kwargs["token"] = HF_TOKEN

        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(LUGANDA_MODEL_ID, use_safetensors=True, **load_kwargs)
        except Exception:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(LUGANDA_MODEL_ID, **load_kwargs)
        try:
            processor = AutoProcessor.from_pretrained(
                LUGANDA_MODEL_ID,
                token=HF_TOKEN,
                language="swahili",
                task="transcribe"
            )
        except Exception:
            processor = AutoProcessor.from_pretrained(LUGANDA_MODEL_ID, token=HF_TOKEN)

        if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "set_prefix_tokens"):
            try:
                processor.tokenizer.set_prefix_tokens(language="swahili", task="transcribe")
            except Exception:
                pass

        luganda_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=256,
            chunk_length_s=30,
            stride_length_s=10,
            batch_size=4,
            return_timestamps=True,
            torch_dtype=TORCH_DTYPE,
            device=DEVICE,
        )
        logger.info("Luganda Whisper pipeline ready.")
    return luganda_pipeline


def get_diarization_models():
    global vad_model, cam_model
    if vad_model is None or cam_model is None:
        from funasr import AutoModel
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading FSMN-VAD & CAM++ Diarization models on {dev}...")
        try:
            vad_model = AutoModel(model="fsmn-vad", device=dev, disable_update=True)
            cam_model = AutoModel(model="cam++", device=dev, disable_update=True)
            logger.info("Diarization models loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed loading diarization on {dev}: {e}. Retrying CPU...")
            vad_model = AutoModel(model="fsmn-vad", device="cpu", disable_update=True)
            cam_model = AutoModel(model="cam++", device="cpu", disable_update=True)
    return vad_model, cam_model


# ====== DIARIZATION ENGINE ======
def diarize_chunks(audio_path: str, chunks: List[Dict], num_speakers: Optional[int] = None) -> Tuple[str, List[Dict]]:
    """Acoustic speaker clustering using CAM++ voiceprints on audio slices."""
    if not chunks or not os.path.exists(audio_path):
        return "", []

    _, cam = get_diarization_models()
    audio_data, sr = load_audio_16k_mono(audio_path)
    if len(audio_data) == 0:
        return "", []

    total_sec = len(audio_data) / sr
    chunk_embeddings = []
    valid_indices = []

    for idx, c in enumerate(chunks):
        text = c.get("text", "").strip()
        if not text:
            continue
        ts = c.get("timestamp", [0.0, 0.0]) or [0.0, 0.0]
        s_sec = max(0.0, float(ts[0]) if ts[0] is not None else 0.0)
        e_sec = min(total_sec, float(ts[1]) if len(ts) > 1 and ts[1] is not None else s_sec + 2.0)

        s_idx = max(0, int((s_sec - 0.1) * sr))
        e_idx = min(len(audio_data), int((e_sec + 0.1) * sr))
        wave = audio_data[s_idx:e_idx]

        if len(wave) < int(0.5 * sr):
            wave = np.pad(wave, (0, int(0.5 * sr) - len(wave)))

        try:
            out = cam.generate(input=wave, fs=sr)
            emb = None
            if isinstance(out, list) and len(out) > 0:
                item = out[0]
                emb = item.get("spk_embedding", item.get("embedding")) if isinstance(item, dict) else item
            elif isinstance(out, dict):
                emb = out.get("spk_embedding", out.get("embedding"))

            if emb is not None:
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()
                emb = np.squeeze(emb).astype(np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                chunk_embeddings.append(emb)
                valid_indices.append(idx)
        except Exception as e:
            logger.debug(f"CAM++ embedding error: {e}")

    if not chunk_embeddings:
        portions = []
        for c in chunks:
            text = c.get("text", "").strip()
            if not text:
                continue
            ts = c.get("timestamp", [0.0, 0.0])
            portions.append({
                "time": format_seconds_to_timestamp(ts[0] if ts else 0.0),
                "speaker": "Speaker 1",
                "content": text
            })
        formatted = "\n\n".join([f"[{p['time']}] {p['speaker']}: {p['content']}" for p in portions])
        return formatted, portions

    emb_matrix = np.array(chunk_embeddings)
    n_samples = len(emb_matrix)

    from sklearn.cluster import AgglomerativeClustering
    if n_samples == 1:
        cluster_labels = [0]
    else:
        try:
            if num_speakers and num_speakers > 0:
                clusterer = AgglomerativeClustering(n_clusters=min(num_speakers, n_samples), metric="cosine", linkage="average")
                cluster_labels = clusterer.fit_predict(emb_matrix)
            else:
                clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=0.68, metric="cosine", linkage="average")
                cluster_labels = clusterer.fit_predict(emb_matrix)
        except Exception:
            cluster_labels = [0] * n_samples

    label_to_spk = {}
    speaker_cnt = 1
    for label in cluster_labels:
        if label not in label_to_spk:
            label_to_spk[label] = f"Speaker {speaker_cnt}"
            speaker_cnt += 1

    portions = []
    for i, c_idx in enumerate(valid_indices):
        chunk = chunks[c_idx]
        ts = chunk.get("timestamp", [0.0, 0.0])
        time_str = format_seconds_to_timestamp(ts[0] if ts else 0.0)
        spk = label_to_spk[cluster_labels[i]]
        portions.append({
            "time": time_str,
            "speaker": spk,
            "content": chunk.get("text", "").strip()
        })

    formatted = "\n\n".join([f"[{p['time']}] {p['speaker']}: {p['content']}" for p in portions])
    return formatted, portions


# ====== STREAMING SPEAKER TRACKER ======
class StreamingSpeakerTracker:
    def __init__(self, similarity_threshold: float = 0.62):
        self.similarity_threshold = similarity_threshold
        self.sessions = {}

    def identify_speaker(self, session_id: str, audio_path: str, text: str = "") -> dict:
        now = time.time()
        # Clean stale sessions
        stale = [k for k, v in self.sessions.items() if now - v.get("last_active", 0) > 3600]
        for k in stale:
            self.sessions.pop(k, None)

        if session_id not in self.sessions:
            self.sessions[session_id] = {"speakers": [], "last_speaker": 1, "last_active": now}
        sess = self.sessions[session_id]
        sess["last_active"] = now

        words = [w for w in re.split(r'\s+', (text or "").strip()) if w]
        is_single_word = len(words) <= 1

        _, cam = get_diarization_models()
        audio_data, sr = load_audio_16k_mono(audio_path)
        dur = len(audio_data) / float(sr) if sr > 0 else 0.0

        if not cam or dur < 0.35:
            spk_id = sess.get("last_speaker", 1)
            return {"speaker": f"Speaker {spk_id}", "speaker_id": spk_id, "is_new_speaker": False}

        out = cam.generate(input=audio_data, fs=sr)
        emb = None
        if isinstance(out, list) and len(out) > 0:
            item = out[0]
            emb = item.get("spk_embedding", item.get("embedding")) if isinstance(item, dict) else item
        elif isinstance(out, dict):
            emb = out.get("spk_embedding", out.get("embedding"))

        if emb is None:
            spk_id = sess.get("last_speaker", 1)
            return {"speaker": f"Speaker {spk_id}", "speaker_id": spk_id, "is_new_speaker": False}

        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
        emb = np.squeeze(emb).astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        speakers = sess["speakers"]
        if not speakers:
            speakers.append({"id": 1, "name": "Speaker 1", "centroid": emb, "count": 1})
            sess["last_speaker"] = 1
            return {"speaker": "Speaker 1", "speaker_id": 1, "is_new_speaker": True}

        best_sim = -1.0
        best_spk = None
        for s in speakers:
            sim = float(np.dot(emb, s["centroid"]))
            if sim > best_sim:
                best_sim = sim
                best_spk = s

        if is_single_word or dur < 0.75:
            last_id = sess.get("last_speaker", 1)
            tgt = next((s for s in speakers if s["id"] == last_id), speakers[0])
            return {"speaker": tgt["name"], "speaker_id": tgt["id"], "is_new_speaker": False}

        if best_sim >= self.similarity_threshold and best_spk is not None:
            best_spk["centroid"] = 0.8 * best_spk["centroid"] + 0.2 * emb
            b_norm = np.linalg.norm(best_spk["centroid"])
            if b_norm > 0:
                best_spk["centroid"] /= b_norm
            best_spk["count"] += 1
            sess["last_speaker"] = best_spk["id"]
            return {"speaker": best_spk["name"], "speaker_id": best_spk["id"], "is_new_speaker": False}
        else:
            new_id = len(speakers) + 1
            speakers.append({"id": new_id, "name": f"Speaker {new_id}", "centroid": emb, "count": 1})
            sess["last_speaker"] = new_id
            return {"speaker": f"Speaker {new_id}", "speaker_id": new_id, "is_new_speaker": True}


stream_tracker = StreamingSpeakerTracker()


# ====== MAIN HANDLER ======
def handle_transcribe(job_input: dict, job: dict = None) -> dict:
    audio_path = decode_input_audio(
        audio_base64=job_input.get("audio_base64"),
        audio_url=job_input.get("audio_url")
    )

    language = (job_input.get("language") or "auto").strip()
    diarize = bool(job_input.get("diarize", True))
    num_speakers = job_input.get("num_speakers")
    if num_speakers and str(num_speakers).isdigit():
        num_speakers = int(num_speakers)
    else:
        num_speakers = None

    is_luganda = language.lower() in ("luganda", "lg")
    chunks = []
    clean_raw_text = ""

    t0 = time.time()

    if runpod is not None and job is not None:
        try:
            runpod.serverless.progress_update(job, {
                "stage": "transcribing",
                "progress_pct": 35,
                "message": f"Neural Speech Recognition ({language})...",
                "subtext": "Transcribing audio segments on Cloud GPU..."
            })
        except Exception:
            pass

    try:
        if is_luganda:
            logger.info(f"Executing Luganda Whisper model ({LUGANDA_MODEL_ID}) with Swahili tokenizer...")
            pipe = get_luganda_pipeline()
            with torch.inference_mode():
                # Luganda Whisper models use Swahili (<|sw|>) token prefix
                gen_kwargs = {"task": "transcribe", "language": "swahili"}
                res = pipe(audio_path, return_timestamps=True, generate_kwargs=gen_kwargs)
            chunks = res.get("chunks", [])
            if chunks:
                clean_raw_text = "\n".join(c["text"].strip() for c in chunks if c.get("text"))
            else:
                clean_raw_text = res.get("text", "").strip()
        else:
            logger.info(f"Executing Faster Whisper for language: {language}...")
            fw = get_faster_whisper()
            target_lang = normalize_faster_whisper_lang(language)
            segs, info = fw.transcribe(
                audio_path,
                language=target_lang,
                beam_size=5,
                vad_filter=True,
                temperature=0.0
            )
            for s in segs:
                txt = filter_streaming_hallucinations(s.text)
                if txt:
                    chunks.append({
                        "timestamp": [round(s.start, 2), round(s.end, 2)],
                        "text": txt
                    })
            clean_raw_text = " ".join(c["text"] for c in chunks).strip()

        # Build basic formatted lines
        timestamped_lines = []
        for c in chunks:
            ts = c.get("timestamp", [0.0, 0.0])
            t_str = format_seconds_to_timestamp(ts[0] if ts else 0.0)
            timestamped_lines.append(f"[{t_str}] {c.get('text', '').strip()}")
        formatted_text = "\n".join(timestamped_lines) if timestamped_lines else clean_raw_text

        # Diarization
        if diarize and chunks:
            if runpod is not None and job is not None:
                try:
                    runpod.serverless.progress_update(job, {
                        "stage": "diarizing",
                        "progress_pct": 72,
                        "message": "Acoustic Speaker Diarization...",
                        "subtext": "Extracting voiceprints with FSMN-VAD & Alibaba CAM++..."
                    })
                except Exception:
                    pass
            logger.info("Executing Acoustic Speaker Diarization...")
            ac_formatted, _ = diarize_chunks(audio_path, chunks, num_speakers=num_speakers)
            if ac_formatted:
                formatted_text = ac_formatted

        elapsed = round(time.time() - t0, 2)
        return {
            "status": "success",
            "text": clean_raw_text,
            "formatted": formatted_text,
            "chunks": chunks,
            "language": language,
            "processing_time": elapsed
        }
    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass


def handle_diarize(job_input: dict) -> dict:
    audio_path = decode_input_audio(
        audio_base64=job_input.get("audio_base64"),
        audio_url=job_input.get("audio_url")
    )
    portions = job_input.get("portions", [])
    num_speakers = job_input.get("num_speakers")
    if num_speakers and str(num_speakers).isdigit():
        num_speakers = int(num_speakers)
    else:
        num_speakers = None

    try:
        chunks = []
        for p in portions:
            time_str = p.get("time", "00:00").replace("[", "").replace("]", "").strip()
            parts = time_str.split(":")
            sec = 0.0
            if len(parts) == 2:
                try: sec = int(parts[0]) * 60 + float(parts[1])
                except: sec = 0.0
            elif len(parts) == 3:
                try: sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                except: sec = 0.0

            chunks.append({
                "timestamp": [sec, sec + 2.0],
                "text": p.get("content", "")
            })

        formatted, updated_portions = diarize_chunks(audio_path, chunks, num_speakers=num_speakers)
        return {
            "status": "success",
            "formatted": formatted,
            "portions": updated_portions
        }
    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass


def handle_stream_chunk(job_input: dict) -> dict:
    audio_path = decode_input_audio(
        audio_base64=job_input.get("audio_base64"),
        audio_url=job_input.get("audio_url")
    )
    session_id = job_input.get("session_id", "default_stream")
    language = (job_input.get("language") or "auto").strip().lower()
    target_lang = "en" if language in ("auto", "") else language
    t0 = time.time()

    try:
        fw = get_faster_whisper()
        segs, _ = fw.transcribe(audio_path, language=target_lang, beam_size=1, vad_filter=False, temperature=0.0)
        text_parts = [s.text.strip() for s in segs]
        raw_text = " ".join(text_parts).strip()
        cleaned = filter_streaming_hallucinations(raw_text)

        elapsed_ms = int((time.time() - t0) * 1000)

        if not cleaned:
            return {"status": "success", "text": "", "speaker": "", "is_speech": False, "latency_ms": elapsed_ms}

        spk_info = stream_tracker.identify_speaker(session_id, audio_path, text=cleaned)
        return {
            "status": "success",
            "text": cleaned,
            "speaker": spk_info.get("speaker", "Speaker 1"),
            "speaker_id": spk_info.get("speaker_id", 1),
            "is_speech": True,
            "latency_ms": elapsed_ms
        }
    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass


def warmup_streaming_models():
    """
    Pre-loads Faster Whisper and CAM++ speaker diarization models into GPU VRAM
    and executes a short dummy tensor pass so PyTorch & CTranslate2 CUDA contexts,
    kernels, and memory allocations are 100% warm before any user request arrives.
    """
    logger.info("--> [WARMUP] Pre-loading Faster Whisper and CAM++ models in GPU memory...")
    t0 = time.time()
    try:
        fw = get_faster_whisper()
        dummy_pcm = np.zeros(3200, dtype=np.float32)  # 0.2s silence
        list(fw.transcribe(dummy_pcm, language="en", beam_size=1, vad_filter=False)[0])
        logger.info(f"--> [WARMUP] Faster Whisper ({FASTER_WHISPER_SIZE}) ready & CUDA context hot.")
    except Exception as e:
        logger.warning(f"--> [WARMUP] Faster Whisper pre-warm warning: {e}")

    try:
        _, cam = get_diarization_models()
        if cam is not None:
            dummy_pcm = np.zeros(16000, dtype=np.float32)  # 1.0s silence
            cam.generate(input=dummy_pcm, fs=16000)
            logger.info("--> [WARMUP] CAM++ speaker diarization model ready & hot.")
    except Exception as e:
        logger.warning(f"--> [WARMUP] Diarization pre-warm warning: {e}")

    logger.info(f"--> [WARMUP] All streaming models fully warmed up in {time.time() - t0:.2f}s.")


def handler(job):
    """RunPod Serverless Master Handler"""
    job_input = job.get("input", {})
    action = job_input.get("action", "transcribe")

    try:
        if action == "transcribe":
            return handle_transcribe(job_input, job=job)
        elif action == "diarize":
            return handle_diarize(job_input)
        elif action == "stream_chunk":
            return handle_stream_chunk(job_input)
        elif action in ("ping", "health", "warmup"):
            warmup_streaming_models()
            return {
                "status": "ready",
                "device": DEVICE,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            }
        else:
            return {"status": "error", "error": f"Unknown action '{action}'"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    warmup_streaming_models()
    if runpod is not None:
        runpod.serverless.start({"handler": handler})
    else:
        logger.info("Runpod package not detected. Running verification tests locally...")
        print("Device:", DEVICE)
        print("Models can be verified on container start.")

