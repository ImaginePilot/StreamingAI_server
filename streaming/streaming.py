#!/usr/bin/env python3
"""
Unified Audio/Video Streaming Orchestrator

- Runs a combined Flask server for audio (/audio) and video (/video) ingestion.
- Warms models (Pyannote + FunASR, YOLO11, CompreFace) before serving.
- Continuously processes recorded fragments to:
  * Diarize + transcribe audio fragments.
  * Run YOLO detection and CompreFace face recognition on matching video frames.
  * Align conversations (5-10s windows, or full conversation if >10s) and crop
    synchronized audio+video clips that cover the entire dialogue without mid-cut.
  * Append per-clip metadata to main/cache/data.json (dialogue text, speakers,
    detections, face info, timings, source fragments).

NOTE: This script builds on existing components in audiostream/ and videostream/.
It does not modify them; it orchestrates their caches. Minimal assumptions are:
- Audio fragments saved in audiostream/cache with metadata in fragments.json.
- Video fragments saved in videostream/cache with metadata in fragments.json.
- Fragment filenames carry timestamps consistent with metadata timestamps.
- YOLO11 weights (yolo11n.pt) exist in videostream/models or will auto-download.
- CompreFace is reachable at configured host/port.

Run:
  python3 streaming.py --host 0.0.0.0 --port 8080

This will:
  1) Warm all models.
  2) Start the ingestion server.
  3) Start background workers that watch fragment metadata and produce clips.
"""

import os
import json
import time
import threading
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from flask import Flask, request, jsonify, Response
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

from werkzeug.serving import make_server

# Rolling data limits
MAX_CLIPS = 50

# Local module imports (reuse existing components)
from audiostream.audio_to_text import AudioTranscriber, get_pyannote_pipeline, get_funasr_model, get_whisper_model
from videostream.opencv import YOLO11Detector
from videostream.facial_recognition import CompreFaceProcessor

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audiostream"
VIDEO_DIR = BASE_DIR / "videostream"
MAIN_CACHE = BASE_DIR / "cache"
AUDIO_CACHE = AUDIO_DIR / "cache"
VIDEO_CACHE = VIDEO_DIR / "cache"
DATA_JSON = MAIN_CACHE / "data.json"

# Ensure caches exist
MAIN_CACHE.mkdir(parents=True, exist_ok=True)
(AUDIO_CACHE).mkdir(parents=True, exist_ok=True)
(VIDEO_CACHE).mkdir(parents=True, exist_ok=True)

video_app = Flask(__name__)
audio_app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model warmup
# ---------------------------------------------------------------------------

def warm_models():
    """Load all heavy models once before serving."""
    print("[WARM] Loading diarization pipeline...")
    get_pyannote_pipeline()
    print("[WARM] Loading FunASR/Whisper...")
    cfg = json.load(open(AUDIO_DIR / "audio_env.json"))
    backend = cfg.get("models", {}).get("asr_backend", "funasr")
    if backend == "funasr":
        get_funasr_model()
    else:
        get_whisper_model(cfg.get("models", {}).get("whisper_model", "base"))
    print("[WARM] Loading YOLO11...")
    _ = YOLO11Detector("yolo11n")
    print("[WARM] Connecting CompreFace...")
    _ = CompreFaceProcessor()
    print("[WARM] All models ready")

# ---------------------------------------------------------------------------
# Ingestion server (lightweight)
# ---------------------------------------------------------------------------

# We accept raw audio bytes and raw video frames (JPEG) via two endpoints.
# Frames/bytes are appended to the existing caches and metadata files so that
# existing receivers stay usable.

AUDIO_FRAGMENT_SECONDS = 5
VIDEO_FRAGMENT_SECONDS = 10

_audio_buffer = bytearray()
_audio_meta = []
_audio_last_flush = time.time()
_audio_idx = 0

_video_frames: List[np.ndarray] = []
_video_last_flush = time.time()
_video_idx = 0


def _flush_audio():
    global _audio_buffer, _audio_idx, _audio_last_flush
    if not _audio_buffer:
        return None
    ts = datetime.utcnow()
    fname = f"fragment_{ts.strftime('%Y%m%d_%H%M%S')}_{_audio_idx:04d}.wav"
    fpath = AUDIO_CACHE / fname
    data = bytes(_audio_buffer)
    # Save wav
    import wave
    with wave.open(str(fpath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(data)
    duration = len(data) / 2 / 44100.0
    meta_entry = {
        "filename": fname,
        "filepath": str(fpath),
        "start_time": ts.isoformat(),
        "end_time": (ts + timedelta(seconds=duration)).isoformat() if duration else ts.isoformat(),
        "duration_seconds": duration,
        "timestamp": ts.timestamp(),
    }
    _audio_meta.append(meta_entry)
    _audio_buffer = bytearray()
    _audio_idx += 1
    _audio_last_flush = time.time()
    _append_audio_fragment_metadata(meta_entry)
    return meta_entry


def _append_audio_fragment_metadata(entry: Dict):
    meta_path = AUDIO_CACHE / "fragments.json"
    try:
        if meta_path.exists():
            data = json.load(open(meta_path))
        else:
            data = {"updated": None, "total_fragments": 0, "max_fragments": 200, "fragments": []}
    except Exception:
        data = {"updated": None, "total_fragments": 0, "max_fragments": 200, "fragments": []}
    data.setdefault("fragments", []).append(entry)
    data["latest"] = entry
    data["total_fragments"] = len(data["fragments"])
    data["updated"] = datetime.utcnow().isoformat()
    with open(meta_path, "w") as f:
        json.dump(data, f, indent=2)


def _flush_video():
    global _video_frames, _video_idx, _video_last_flush
    if not _video_frames:
        return None
    ts = datetime.utcnow()
    fname = f"fragment_{ts.strftime('%Y%m%d_%H%M%S')}_{_video_idx:04d}.avi"
    fpath = VIDEO_CACHE / fname
    height, width = _video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = max(1, int(len(_video_frames) / max(1e-3, time.time() - _video_last_flush)))
    writer = cv2.VideoWriter(str(fpath), fourcc, fps, (width, height))
    for f in _video_frames:
        writer.write(f)
    writer.release()
    duration = len(_video_frames) / float(fps)
    meta_entry = {
        "filename": fname,
        "filepath": str(fpath),
        "frames": len(_video_frames),
        "duration": duration,
        "timestamp": ts.timestamp(),
        "created": ts.isoformat(),
    }
    _video_frames = []
    _video_idx += 1
    _video_last_flush = time.time()
    _append_video_fragment_metadata(meta_entry)
    return meta_entry


def _append_video_fragment_metadata(entry: Dict):
    meta_path = VIDEO_CACHE / "fragments.json"
    try:
        if meta_path.exists():
            data = json.load(open(meta_path))
        else:
            data = {"updated": None, "total_fragments": 0, "max_fragments": 200, "fragments": []}
    except Exception:
        data = {"updated": None, "total_fragments": 0, "max_fragments": 200, "fragments": []}
    data.setdefault("fragments", []).append(entry)
    data["latest"] = entry
    data["total_fragments"] = len(data["fragments"])
    data["updated"] = datetime.utcnow().isoformat()
    with open(meta_path, "w") as f:
        json.dump(data, f, indent=2)


@audio_app.route("/", methods=["GET"])
def audio_index():
    return jsonify({
        'service': 'Audio Stream Receiver',
        'status': 'running',
        'version': '1.0',
        'endpoints': {
            'health': '/health',
            'stream_start': '/stream/start',
            'stream_stop': '/stream/stop',
            'audio_chunk': '/audio/chunk',
            'fragments': '/fragments'
        }
    })


@audio_app.route("/health", methods=["GET"])
def audio_health():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.utcnow().isoformat(),
        'active_session': True,
        'chunks_received': _audio_idx,
    })


@audio_app.route("/fragments", methods=["GET"])
def audio_fragments():
    meta_path = AUDIO_CACHE / "fragments.json"
    if meta_path.exists():
        return jsonify(json.load(open(meta_path)))
    return jsonify({"fragments": []})


@audio_app.route("/stream/start", methods=["POST"])
def audio_stream_start():
    global _audio_buffer, _audio_idx, _audio_last_flush
    data = request.get_json(silent=True) or {}
    sample_rate = data.get('sample_rate', 44100)
    channels = data.get('channels', 1)
    sample_width = data.get('sample_width', 2)
    session_id = data.get('session_id', datetime.utcnow().strftime('%Y%m%d_%H%M%S'))
    _audio_buffer = bytearray()
    _audio_last_flush = time.time()
    return jsonify({
        'status': 'started',
        'session_id': session_id,
        'sample_rate': sample_rate,
        'channels': channels,
        'sample_width': sample_width,
        'fragment_duration': AUDIO_FRAGMENT_SECONDS,
        'message': 'Audio stream session started'
    })


@audio_app.route("/stream/stop", methods=["POST"])
def audio_stream_stop():
    entry = _flush_audio()
    return jsonify({
        'status': 'stopped',
        'session_id': entry.get('filename', '') if entry else '',
        'chunks_received': _audio_idx,
        'bytes_received': 0,
        'fragments_saved': _audio_idx
    })


@audio_app.route("/audio/chunk", methods=["POST"])
def ingest_audio():
    global _audio_buffer
    chunk = request.data
    if not chunk:
        return jsonify({'error': 'No data received'}), 400
    _audio_buffer.extend(chunk)
    flushed = False
    if len(_audio_buffer) > 2 * 44100 * AUDIO_FRAGMENT_SECONDS:
        _flush_audio()
        flushed = True
    return jsonify({
        'status': 'received',
        'bytes': len(chunk),
        'chunk_count': _audio_idx,
        'fragment': _audio_idx,
        'queued': True,
        'flushed': flushed
    })


@video_app.route("/", methods=["GET"])
def video_index():
    return jsonify({
        'service': 'Video Stream Receiver',
        'status': 'running',
        'version': '1.0',
        'endpoints': {
            'health': '/health',
            'stream_start': '/stream/start',
            'stream_stop': '/stream/stop',
            'stream_chunk': '/stream/chunk',
            'stream_data': '/stream/data',
            'camera_frame': '/camera/frame',
            'fragments': '/fragments'
        }
    })


@video_app.route("/health", methods=["GET"])
def video_health():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.utcnow().isoformat(),
        'active_session': True,
        'frames_received': len(_video_frames),
    })


@video_app.route("/fragments", methods=["GET"])
def video_fragments():
    meta_path = VIDEO_CACHE / "fragments.json"
    if meta_path.exists():
        return jsonify(json.load(open(meta_path)))
    return jsonify({"fragments": []})


@video_app.route("/stream/start", methods=["POST"])
def video_stream_start():
    global _video_frames, _video_last_flush, _video_idx
    data = request.get_json(silent=True) or {}
    session_id = data.get('session_id', datetime.utcnow().strftime('%Y%m%d_%H%M%S'))
    _video_frames = []
    _video_last_flush = time.time()
    return jsonify({
        'status': 'started',
        'session_id': session_id,
        'fragment_duration': VIDEO_FRAGMENT_SECONDS
    })


@video_app.route("/stream/stop", methods=["POST"])
def video_stream_stop():
    entry = _flush_video()
    return jsonify({
        'status': 'stopped',
        'session_id': entry.get('filename', '') if entry else '',
        'frames_received': _video_idx,
        'fragments_saved': _video_idx
    })


@video_app.route("/stream/data", methods=["POST"])
@video_app.route("/stream/chunk", methods=["POST"])
@video_app.route("/camera/frame", methods=["POST"])
def ingest_video():
    global _video_frames
    # Handle multipart form or raw body
    file_data = None
    for field_name in ['video', 'chunk', 'frame', 'image']:
        if field_name in request.files:
            file_data = request.files[field_name].read()
            break
    if file_data is None:
        file_data = request.data
    if not file_data:
        return jsonify({'error': 'No video data found'}), 400
    nparr = np.frombuffer(file_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Could not decode frame'}), 400
    _video_frames.append(frame)
    flushed = False
    if time.time() - _video_last_flush >= VIDEO_FRAGMENT_SECONDS:
        _flush_video()
        flushed = True
    return jsonify({
        'status': 'received',
        'bytes': len(file_data),
        'fragment': _video_idx,
        'flushed': flushed
    })


# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------

class ClipMetadata:
    def __init__(self, clip_path: Path, start_ts: float, end_ts: float, text: str,
                 speakers: List[str], yolo: List[Dict], faces: List[Dict], src_audio: str, src_video: List[str]):
        self.clip_path = clip_path
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.text = text
        self.speakers = speakers
        self.yolo = yolo
        self.faces = faces
        self.src_audio = src_audio
        self.src_video = src_video

    def to_dict(self):
        return {
            "clip_path": str(self.clip_path),
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "duration": self.end_ts - self.start_ts,
            "text": self.text,
            "speakers": self.speakers,
            "yolo": self.yolo,
            "faces": self.faces,
            "source_audio": self.src_audio,
            "source_video": self.src_video,
        }


def load_data_json():
    if DATA_JSON.exists():
        try:
            return json.load(open(DATA_JSON))
        except Exception:
            return {"clips": []}
    return {"clips": []}


def save_data_json(data: Dict):
    """Save data.json and enforce rolling limit of MAX_CLIPS entries."""
    clips = data.get("clips", [])
    
    # Enforce rolling limit
    while len(clips) > MAX_CLIPS:
        oldest = clips.pop(0)
        # Delete the oldest clip file
        try:
            old_path = Path(oldest.get("clip_path", ""))
            if old_path.exists():
                old_path.unlink()
                print(f"[CLEANUP] Removed old clip: {old_path.name}")
        except Exception as e:
            print(f"[WARN] Failed to cleanup old clip: {e}")
    
    data["clips"] = clips
    data["updated"] = datetime.utcnow().isoformat()
    data["total_clips"] = len(clips)
    data["max_clips"] = MAX_CLIPS
    with open(DATA_JSON, "w") as f:
        json.dump(data, f, indent=2)


def read_metadata(meta_path: Path):
    try:
        return json.load(open(meta_path))
    except Exception:
        return None


# Track when program started - only process fragments created after this
_program_start_ts: float = 0.0


def set_program_start_ts():
    global _program_start_ts
    _program_start_ts = time.time()


def list_unprocessed_audio(processed_hashes: set):
    meta = read_metadata(AUDIO_CACHE / "fragments.json") or {}
    items = meta.get("fragments", [])
    new_items = []
    for it in items:
        fname = it.get("filename")
        if not fname:
            continue
        if fname in processed_hashes:
            continue
        # Skip historic data (created before program started)
        frag_ts = it.get("timestamp", 0)
        if frag_ts < _program_start_ts:
            continue
        new_items.append(it)
    return new_items


def list_unprocessed_video(min_ts: float = None):
    meta = read_metadata(VIDEO_CACHE / "fragments.json") or {}
    items = meta.get("fragments", [])
    if min_ts is None:
        min_ts = _program_start_ts
    # Only return fragments created after program start
    return [v for v in items if v.get("timestamp", 0) >= min_ts]


def transcribe_fragment(transcriber: AudioTranscriber, audio_entry: Dict):
    path = AUDIO_CACHE / audio_entry["filename"]
    if not path.exists():
        return None
    res = transcriber.process_file(str(path), force=True)
    if not res:
        return None
    start_ts = datetime.fromisoformat(audio_entry.get("start_time")).timestamp() if audio_entry.get("start_time") else audio_entry.get("timestamp", time.time())
    enriched_segments = []
    for seg in res.segments:
        enriched_segments.append({
            "abs_start": start_ts + seg.start_time,
            "abs_end": start_ts + seg.end_time,
            "speaker": seg.speaker_id,
            "text": seg.text,
            "language": seg.language,
            "emotion": seg.emotion,
        })
    return enriched_segments, res


def group_conversations(segments: List[Dict], min_len=5.0, max_gap=1.5):
    segments = sorted(segments, key=lambda s: s["abs_start"])
    conversations = []
    if not segments:
        return conversations
    cur = [segments[0]]
    for seg in segments[1:]:
        if seg["abs_start"] - cur[-1]["abs_end"] <= max_gap:
            cur.append(seg)
        else:
            conversations.append(cur)
            cur = [seg]
    conversations.append(cur)
    filtered = []
    for conv in conversations:
        start = conv[0]["abs_start"]
        end = conv[-1]["abs_end"]
        if end - start < min_len:
            end = start + min_len
        filtered.append((start, end, conv))
    return filtered


def slice_audio(audio_path: Path, abs_start: float, abs_end: float, frag_meta: Dict, out_path: Path, target_duration: float = None) -> Tuple[Path, float]:
    """
    Slice audio to match time window.
    If target_duration provided, pad/trim to exactly match video duration.
    Returns (output_path, actual_duration).
    """
    import torchaudio
    import torch
    wav, sr = torchaudio.load(str(audio_path))
    frag_start_ts = datetime.fromisoformat(frag_meta.get("start_time")).timestamp()
    offset = max(0.0, abs_start - frag_start_ts)
    end_off = max(0.0, abs_end - frag_start_ts)
    start_sample = int(offset * sr)
    end_sample = int(end_off * sr)
    start_sample = max(0, min(start_sample, wav.shape[1]))
    end_sample = max(start_sample, min(end_sample, wav.shape[1]))
    seg = wav[:, start_sample:end_sample]
    
    # If target_duration specified, pad or trim to match exactly
    if target_duration is not None and target_duration > 0:
        target_samples = int(target_duration * sr)
        current_samples = seg.shape[1]
        if current_samples < target_samples:
            # Pad with silence
            padding = torch.zeros((seg.shape[0], target_samples - current_samples), dtype=seg.dtype)
            seg = torch.cat([seg, padding], dim=1)
        elif current_samples > target_samples:
            # Trim
            seg = seg[:, :target_samples]
    
    torchaudio.save(str(out_path), seg, sr)
    actual_duration = seg.shape[1] / sr
    return out_path, actual_duration


def slice_video(video_files: List[Dict], abs_start: float, abs_end: float, out_path: Path) -> Tuple[Optional[Path], float, float]:
    """
    Slice video fragments to match time window.
    Returns (output_path, actual_duration, fps) for sync.
    """
    writer = None
    total_frames = 0
    out_fps = 15.0
    for vf in video_files:
        vpath = Path(vf["filepath"])
        if not vpath.exists():
            continue
        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur = frame_count / fps if fps else 0
        frag_ts = vf.get("timestamp", 0)
        frag_start = frag_ts
        frag_end = frag_ts + dur
        # Skip if no overlap
        if frag_end < abs_start or frag_start > abs_end:
            cap.release()
            continue
        # Compute frame indices
        start_off = max(0.0, abs_start - frag_start)
        end_off = max(0.0, min(abs_end, frag_end) - frag_start)
        start_idx = int(start_off * fps)
        end_idx = int(end_off * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out_fps = fps
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        cur = start_idx
        while cur <= end_idx:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            total_frames += 1
            cur += 1
        cap.release()
    if writer:
        writer.release()
    actual_duration = total_frames / out_fps if out_fps > 0 else 0.0
    return (out_path if writer else None, actual_duration, out_fps)


def run_detectors(yolo: YOLO11Detector, compreface: CompreFaceProcessor, video_path: Path, sample_every=5):
    dets = []
    faces = []
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            objects = yolo.detect(frame)
            dets.extend(objects)
            try:
                # Convert numpy frame to JPEG bytes for CompreFace
                _, img_encoded = cv2.imencode('.jpg', frame)
                img_bytes = img_encoded.tobytes()
                face_results = compreface.detect_faces_from_bytes(img_bytes)
                faces.extend(face_results)
            except Exception as e:
                # Only log if it's not a common "no faces" situation
                if "result" not in str(e).lower():
                    print(f"[WARN] Face detection error: {e}")
        frame_idx += 1
    cap.release()
    return dets, faces


def mux_audio_video(video_path: Path, audio_path: Path, output_path: Path, video_fps: float = 15.0) -> Optional[Path]:
    """
    Merge video (avi) and audio (wav) into a single mp4 file using ffmpeg.
    Uses precise timing for maximum A/V synchronization.
    Returns output_path on success, None on failure.
    """
    import subprocess
    if not video_path.exists():
        print(f"[WARN] Video file not found for muxing: {video_path}")
        return None
    if not audio_path.exists():
        print(f"[WARN] Audio file not found for muxing: {audio_path}")
        # If no audio, just copy video to output
        try:
            import shutil
            shutil.copy(str(video_path), str(output_path.with_suffix('.avi')))
            return output_path.with_suffix('.avi')
        except Exception as e:
            print(f"[ERROR] Failed to copy video: {e}")
            return None
    
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-i', str(audio_path),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-af', 'aresample=async=1:first_pts=0',  # Resample audio to sync with video
            '-vsync', 'cfr',  # Constant frame rate for better sync
            '-r', str(video_fps),  # Match output fps to source
            '-shortest',  # End when shortest stream ends
            '-map', '0:v:0',  # First input video
            '-map', '1:a:0',  # Second input audio
            str(output_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"[ERROR] ffmpeg failed: {result.stderr}")
            return None
        # Clean up intermediate files
        try:
            video_path.unlink()
            audio_path.unlink()
        except Exception:
            pass
        print(f"[INFO] Merged clip (synced): {output_path}")
        return output_path
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found. Install ffmpeg to merge audio+video.")
        return None
    except subprocess.TimeoutExpired:
        print("[ERROR] ffmpeg timed out")
        return None
    except Exception as e:
        print(f"[ERROR] mux failed: {e}")
        return None


def process_single_conversation(args_tuple):
    """
    Process a single conversation clip. Called in parallel.
    Returns ClipMetadata dict or None.
    """
    (conv_start, conv_end, conv_segments, candidates, audio_entry, 
     clip_video_tmp, clip_audio_tmp, clip_path, yolo, compreface) = args_tuple
    
    try:
        # Slice video first to get actual duration
        video_result, video_duration, video_fps = slice_video(candidates, conv_start, conv_end, clip_video_tmp)
        
        # Slice audio to exactly match video duration for sync
        audio_result, audio_duration = slice_audio(
            AUDIO_CACHE / audio_entry["filename"], conv_start, conv_end, audio_entry, clip_audio_tmp,
            target_duration=video_duration
        )
        
        # Merge audio + video into single mp4 with sync options
        merged = mux_audio_video(clip_video_tmp, clip_audio_tmp, clip_path, video_fps=video_fps)
        final_clip = merged if merged else clip_video_tmp
        detection_path = final_clip if (final_clip and Path(final_clip).exists()) else clip_video_tmp
        
        # Run detectors
        yolo_dets, face_dets = ([], [])
        if detection_path and Path(detection_path).exists():
            yolo_dets, face_dets = run_detectors(yolo, compreface, detection_path)
        
        text = " ".join([s["text"] for s in conv_segments])
        speakers = list({s["speaker"] for s in conv_segments})
        
        return {
            "clip_path": str(final_clip) if final_clip else str(clip_video_tmp),
            "start_ts": conv_start,
            "end_ts": conv_end,
            "duration": conv_end - conv_start,
            "text": text,
            "speakers": speakers,
            "yolo": yolo_dets,
            "faces": face_dets,
            "source_audio": audio_entry["filename"],
            "source_video": [c.get("filename") for c in candidates],
        }
    except Exception as e:
        print(f"[ERROR] process_single_conversation failed: {e}")
        return None


def processing_loop(stop_event: threading.Event):
    transcriber = AudioTranscriber()
    yolo = YOLO11Detector("yolo11n")
    compreface = CompreFaceProcessor()
    processed_audio = set()
    data = load_data_json()
    
    # Use thread pool for parallel processing
    max_workers = min(4, (os.cpu_count() or 2))
    
    while not stop_event.is_set():
        new_audio = list_unprocessed_audio(processed_audio)
        if not new_audio:
            time.sleep(2.0)
            continue
        
        for entry in new_audio:
            try:
                out = transcribe_fragment(transcriber, entry)
                if not out:
                    processed_audio.add(entry.get("filename"))
                    continue
                segments, res = out
                conversations = group_conversations(segments)
                
                if not conversations:
                    processed_audio.add(entry.get("filename"))
                    continue
                
                # Gather relevant video fragments that overlap
                video_meta = list_unprocessed_video()
                
                # Prepare tasks for parallel processing
                tasks = []
                for conv_start, conv_end, conv_segments in conversations:
                    clip_id = f"clip_{int(conv_start)}_{int(conv_end)}"
                    clip_video_tmp = MAIN_CACHE / f"{clip_id}_tmp.avi"
                    clip_audio_tmp = MAIN_CACHE / f"{clip_id}_tmp.wav"
                    clip_path = MAIN_CACHE / f"{clip_id}.mp4"
                    
                    candidates = [v for v in video_meta 
                                  if (v.get("timestamp",0) + v.get("duration",0)) >= conv_start 
                                  and v.get("timestamp",0) <= conv_end]
                    
                    tasks.append((
                        conv_start, conv_end, conv_segments, candidates, entry,
                        clip_video_tmp, clip_audio_tmp, clip_path, yolo, compreface
                    ))
                
                # Process conversations in parallel
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(process_single_conversation, t) for t in tasks]
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            data.setdefault("clips", []).append(result)
                            save_data_json(data)
                
                processed_audio.add(entry.get("filename"))
                
            except Exception as e:
                print(f"[ERROR] pipeline failed for {entry.get('filename')}: {e}")
                import traceback
                traceback.print_exc()
                continue


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

from argparse import ArgumentParser


class ServerThread(threading.Thread):
    def __init__(self, app, host, port):
        super().__init__(daemon=True)
        self.server = make_server(host, port, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()


def clear_all_data():
    """
    Clear all cached data for a fresh start:
      - main/cache/ (clips and data.json)
      - main/transcripts/ (voices.json, conversation_history.json - runtime location)
      - main/audiostream/cache/ (audio fragments)
      - main/videostream/cache/ (video fragments)
      - main/audiostream/transcripts/ (voices.json, conversation_history.json)
      - main/videostream/face_cache/ (face database and images)
    """
    import shutil
    
    # Define paths based on existing constants
    # Note: AudioTranscriber uses relative paths from CWD, so main/transcripts/ is also used
    main_transcript_dir = BASE_DIR / "transcripts"
    transcript_dir = AUDIO_DIR / "transcripts"
    face_cache_dir = VIDEO_DIR / "face_cache"
    face_img_dir = face_cache_dir / "face_images"
    
    # Files in main/transcripts/ (runtime location when running from main/)
    main_voices_file = main_transcript_dir / "voices.json"
    main_conversation_file = main_transcript_dir / "conversation_history.json"
    
    # Files in audiostream/transcripts/ (module default location)
    voices_file = transcript_dir / "voices.json"
    conversation_file = transcript_dir / "conversation_history.json"
    
    current_faces_file = face_cache_dir / "current_faces.json"
    all_faces_file = face_cache_dir / "all_faces.json"
    
    dirs_to_clear = [
        MAIN_CACHE,                         # main/cache/
        main_transcript_dir,                # main/transcripts/ (runtime)
        AUDIO_CACHE,                        # main/audiostream/cache/
        VIDEO_CACHE,                        # main/videostream/cache/
        transcript_dir,                     # main/audiostream/transcripts/
        face_cache_dir,                     # main/videostream/face_cache/
    ]
    
    print("[CLEAR] Clearing all cached data for fresh start...")
    for d in dirs_to_clear:
        d_str = str(d)
        if os.path.exists(d_str):
            # Remove all contents but keep the directory
            for item in os.listdir(d_str):
                item_path = os.path.join(d_str, item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.remove(item_path)
                        print(f"[CLEAR] Removed file: {item_path}")
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                        print(f"[CLEAR] Removed directory: {item_path}")
                except Exception as e:
                    print(f"[CLEAR] Error removing {item_path}: {e}")
        else:
            os.makedirs(d_str, exist_ok=True)
            print(f"[CLEAR] Created directory: {d_str}")
    
    # Recreate required subdirectories
    os.makedirs(str(face_img_dir), exist_ok=True)
    print(f"[CLEAR] Recreated: {face_img_dir}")
    
    # Initialize empty JSON files
    with open(DATA_JSON, "w") as f:
        json.dump({"conversations": []}, f)
    print(f"[CLEAR] Initialized: {DATA_JSON}")
    
    # Initialize main/transcripts/ (runtime location)
    with open(main_voices_file, "w") as f:
        json.dump({}, f)
    print(f"[CLEAR] Initialized: {main_voices_file}")
    
    with open(main_conversation_file, "w") as f:
        json.dump([], f)
    print(f"[CLEAR] Initialized: {main_conversation_file}")
    
    # Initialize audiostream/transcripts/ (module location)
    with open(voices_file, "w") as f:
        json.dump({}, f)
    print(f"[CLEAR] Initialized: {voices_file}")
    
    with open(conversation_file, "w") as f:
        json.dump([], f)
    print(f"[CLEAR] Initialized: {conversation_file}")
    
    with open(current_faces_file, "w") as f:
        json.dump([], f)
    print(f"[CLEAR] Initialized: {current_faces_file}")
    
    with open(all_faces_file, "w") as f:
        json.dump({}, f)
    print(f"[CLEAR] Initialized: {all_faces_file}")
    
    print("[CLEAR] All data cleared successfully.")


def main():
    parser = ArgumentParser(description="Unified streaming orchestrator")
    parser.add_argument("--video-host", default="0.0.0.0")
    parser.add_argument("--video-port", type=int, default=5000)
    parser.add_argument("--audio-host", default="0.0.0.0")
    parser.add_argument("--audio-port", type=int, default=5001)
    parser.add_argument("--clear", action="store_true",
                        help="Clear all cache, face, voice, and conversation data before starting")
    args = parser.parse_args()

    if args.clear:
        clear_all_data()

    warm_models()
    
    # Mark program start time - only process new data from now on
    set_program_start_ts()
    print(f"[INFO] Program started at {datetime.utcnow().isoformat()} - ignoring historic fragments")

    stop_event = threading.Event()
    worker = threading.Thread(target=processing_loop, args=(stop_event,), daemon=True)
    worker.start()

    video_server = ServerThread(video_app, args.video_host, args.video_port)
    audio_server = ServerThread(audio_app, args.audio_host, args.audio_port)
    video_server.start()
    audio_server.start()

    print(f"[SERVE] Video endpoints on http://{args.video_host}:{args.video_port} (compatible: /stream/start, /stream/chunk, /stream/data, /camera/frame)")
    print(f"[SERVE] Audio endpoints on http://{args.audio_host}:{args.audio_port} (compatible: /stream/start, /audio/chunk, /stream/stop)")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        worker.join(timeout=2.0)
        video_server.shutdown()
        audio_server.shutdown()


if __name__ == "__main__":
    main()
