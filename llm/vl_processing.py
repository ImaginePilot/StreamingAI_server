#!/usr/bin/env python3
"""
VL Processing Module - Fast Vision-Language Processing for Continuous Streaming

Processes video clips from the streaming cache using Ollama's vision models
and stores results in vl_processed.json for downstream decision-making.

Features:
- Fast mode optimized for continuous processing (2 frames, 512 tokens)
- Stores structured VLM analysis results in vl_processed.json
- Continuous processing function for real-time pipeline
- Ollama backend only (qwen3-vl:30b)

Usage:
    python vl_processing.py --process              # Process all unprocessed clips
    python vl_processing.py --continuous           # Watch and process new clips
    python vl_processing.py --testrun              # Test with latest clip
"""

import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
STREAMING_DIR = BASE_DIR / "streaming"
CACHE_DIR = STREAMING_DIR / "cache"
DATA_JSON = CACHE_DIR / "data.json"
VL_PROCESSED_JSON = BASE_DIR / "llm" / "vl_processed.json"

# Ollama Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "192.168.2.6")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")
OLLAMA_MODEL = os.getenv("OLLAMA_VL_MODEL", "qwen3-vl:30b")

# Fast mode settings (optimized for continuous processing)
FAST_MAX_TOKENS = int(os.getenv("VL_MAX_TOKENS", "512"))
FAST_NUM_FRAMES = int(os.getenv("VL_NUM_FRAMES", "2"))
FAST_IMAGE_QUALITY = int(os.getenv("VL_IMAGE_QUALITY", "10"))
FAST_RESIZE_WIDTH = int(os.getenv("VL_RESIZE_WIDTH", "512"))
TEMPERATURE = float(os.getenv("VL_TEMPERATURE", "0.7"))


@dataclass
class ClipData:
    """Represents a processed clip with all metadata."""
    clip_path: str
    start_ts: float
    end_ts: float
    duration: float
    text: str
    speakers: List[str] = field(default_factory=list)
    yolo_detections: List[Dict[str, Any]] = field(default_factory=list)
    faces: List[Dict[str, Any]] = field(default_factory=list)
    source_audio: str = ""
    source_video: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClipData":
        return cls(
            clip_path=data.get("clip_path", ""),
            start_ts=data.get("start_ts", 0.0),
            end_ts=data.get("end_ts", 0.0),
            duration=data.get("duration", 0.0),
            text=data.get("text", ""),
            speakers=data.get("speakers", []),
            yolo_detections=data.get("yolo", []),
            faces=data.get("faces", []),
            source_audio=data.get("source_audio", ""),
            source_video=data.get("source_video", []),
        )


@dataclass
class VLResult:
    """Result from VL model processing."""
    clip_path: str
    start_ts: float
    end_ts: float
    duration: float
    speakers: List[str]
    transcription: str
    vl_analysis: str  # Raw VLM output
    scene: str        # Extracted scene description
    speech: str       # Extracted speech analysis
    summary: str      # Extracted summary
    processed_at: str
    model: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VLResult":
        return cls(**data)


def load_data_json() -> Dict[str, Any]:
    """Load the streaming data.json file."""
    if not DATA_JSON.exists():
        return {"clips": [], "conversations": []}
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def get_clips() -> List[ClipData]:
    """Get all clips from data.json."""
    data = load_data_json()
    return [ClipData.from_dict(clip) for clip in data.get("clips", [])]


def load_vl_processed() -> Dict[str, VLResult]:
    """Load already processed clips from vl_processed.json."""
    if not VL_PROCESSED_JSON.exists():
        return {}
    try:
        with open(VL_PROCESSED_JSON, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            data = json.loads(content)
        return {k: VLResult.from_dict(v) for k, v in data.get("results", {}).items()}
    except (json.JSONDecodeError, KeyError):
        return {}


def save_vl_processed(results: Dict[str, VLResult]) -> None:
    """Save processed results to vl_processed.json."""
    data = {
        "last_updated": datetime.now().isoformat(),
        "count": len(results),
        "results": {k: v.to_dict() for k, v in results.items()}
    }
    VL_PROCESSED_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(VL_PROCESSED_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_clip_id(clip: ClipData) -> str:
    """Generate unique ID for a clip."""
    return f"{clip.clip_path}:{clip.start_ts}"


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Encode an image file to base64."""
    path = Path(image_path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_frames_from_video(video_path: str) -> List[str]:
    """
    Extract frames from video for VL processing.
    Uses fast mode settings: 2 frames, compressed, resized.
    """
    # Get video duration
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return []

    try:
        duration = float(result.stdout.strip())
    except ValueError:
        duration = 1.0

    if duration <= 0:
        duration = 1.0

    # Extract frames at start and middle (2 frames for fast mode)
    frame_times = [0.0, duration / 2]
    
    frame_paths = []
    for i, timestamp in enumerate(frame_times):
        output_path = tempfile.mktemp(suffix=f"_frame_{i}.jpg")
        cmd = [
            "ffmpeg", "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", str(FAST_IMAGE_QUALITY),
            "-vf", f"scale={FAST_RESIZE_WIDTH}:-1",
            "-y", output_path
        ]
        subprocess.run(cmd, capture_output=True, check=False)
        if os.path.exists(output_path):
            frame_paths.append(output_path)

    return frame_paths


def build_compact_context(clip: ClipData) -> str:
    """Build compact context for the VL model."""
    # Compact face info
    identities = {}
    for face in clip.faces:
        subjects = face.get("subjects", [])
        if subjects:
            best = max(subjects, key=lambda x: x.get("similarity", 0))
            if best.get("similarity", 0) >= 0.5:
                identity_id = best.get("subject", "unknown")
                if identity_id not in identities:
                    age = face.get("age", {})
                    gender = face.get("gender", {}).get("value", "?")
                    identities[identity_id] = f"{gender},{age.get('low','?')}-{age.get('high','?')}"
    
    face_str = "; ".join([f"{k}({v})" for k, v in identities.items()]) if identities else "none"
    
    # Compact YOLO
    yolo_counts = {}
    for det in clip.yolo_detections:
        cls = det.get("class", "unknown")
        yolo_counts[cls] = yolo_counts.get(cls, 0) + 1
    yolo_str = ", ".join([f"{v}x {k}" for k, v in yolo_counts.items()]) if yolo_counts else "none"
    
    return f"""[CLIP {clip.duration:.0f}s | {','.join(clip.speakers) or 'no speaker'}]
Speech: {clip.text or '(none)'}
Objects: {yolo_str}
Faces: {face_str}"""


def get_system_prompt() -> str:
    """System prompt for fast VL processing."""
    return """You are a real-time video analyst. You receive metadata AND actual video frames.

Your response format (keep it brief):
**Scene**: [What you SEE in the frames - environment, actions, expressions, body language, objects NOT in metadata]
**Speech**: [Who said what, tone/emotion if apparent]
**Summary**: [1 sentence: what's happening]

IMPORTANT: The "Scene" section must be YOUR OWN visual observations from the frames, not just restating the metadata. Look for details the automated detection missed."""


def parse_vl_response(response: str) -> tuple[str, str, str]:
    """Parse the VL model response into scene, speech, summary sections."""
    scene = ""
    speech = ""
    summary = ""
    
    lines = response.strip().split("\n")
    current_section = None
    current_content = []
    
    for line in lines:
        line_lower = line.lower().strip()
        if line_lower.startswith("**scene**"):
            if current_section and current_content:
                if current_section == "scene":
                    scene = " ".join(current_content)
                elif current_section == "speech":
                    speech = " ".join(current_content)
                elif current_section == "summary":
                    summary = " ".join(current_content)
            current_section = "scene"
            # Get content after the header
            content = line.split(":", 1)[-1].strip() if ":" in line else ""
            current_content = [content] if content else []
        elif line_lower.startswith("**speech**"):
            if current_section and current_content:
                if current_section == "scene":
                    scene = " ".join(current_content)
            current_section = "speech"
            content = line.split(":", 1)[-1].strip() if ":" in line else ""
            current_content = [content] if content else []
        elif line_lower.startswith("**summary**"):
            if current_section and current_content:
                if current_section == "scene":
                    scene = " ".join(current_content)
                elif current_section == "speech":
                    speech = " ".join(current_content)
            current_section = "summary"
            content = line.split(":", 1)[-1].strip() if ":" in line else ""
            current_content = [content] if content else []
        elif current_section and line.strip():
            current_content.append(line.strip())
    
    # Handle last section
    if current_section and current_content:
        if current_section == "scene":
            scene = " ".join(current_content)
        elif current_section == "speech":
            speech = " ".join(current_content)
        elif current_section == "summary":
            summary = " ".join(current_content)
    
    return scene, speech, summary


def check_ollama_health() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def process_clip_with_vl(clip: ClipData, verbose: bool = True) -> Optional[VLResult]:
    """
    Process a single clip with the VL model.
    
    Args:
        clip: ClipData object to process
        verbose: Whether to print progress
        
    Returns:
        VLResult or None if processing failed
    """
    if verbose:
        print(f"[VL] Processing: {clip.clip_path}")
    
    # Check video exists
    if not clip.clip_path or not Path(clip.clip_path).exists():
        if verbose:
            print(f"[WARN] Video not found: {clip.clip_path}")
        return None
    
    # Extract frames
    frame_paths = extract_frames_from_video(clip.clip_path)
    if not frame_paths:
        if verbose:
            print(f"[WARN] Could not extract frames from: {clip.clip_path}")
        return None
    
    try:
        # Encode frames
        images = []
        for frame_path in frame_paths:
            img_b64 = encode_image_to_base64(frame_path)
            if img_b64:
                images.append(img_b64)
        
        if not images:
            return None
        
        # Build prompt
        context = build_compact_context(clip)
        prompt = f"{context}\n\nBriefly summarize what's happening."
        
        # Build messages
        messages = [
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": prompt, "images": images}
        ]
        
        # Send to Ollama
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "think": False,  # Disable thinking for speed
            "options": {
                "num_predict": FAST_MAX_TOKENS,
                "temperature": TEMPERATURE,
            }
        }
        
        url = f"{OLLAMA_BASE_URL}/api/chat"
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        result_text = response.json().get("message", {}).get("content", "")
        
        if verbose:
            print(f"[VL] Response: {result_text[:100]}...")
        
        # Parse response
        scene, speech, summary = parse_vl_response(result_text)
        
        return VLResult(
            clip_path=clip.clip_path,
            start_ts=clip.start_ts,
            end_ts=clip.end_ts,
            duration=clip.duration,
            speakers=clip.speakers,
            transcription=clip.text,
            vl_analysis=result_text,
            scene=scene,
            speech=speech,
            summary=summary,
            processed_at=datetime.now().isoformat(),
            model=OLLAMA_MODEL,
        )
        
    finally:
        # Cleanup temp frames
        for frame_path in frame_paths:
            try:
                os.remove(frame_path)
            except OSError:
                pass


def process_new_clips(verbose: bool = True) -> int:
    """
    Process all unprocessed clips from data.json.
    
    Returns:
        Number of clips processed
    """
    # Load existing processed results
    processed = load_vl_processed()
    
    # Get all clips
    clips = get_clips()
    
    if verbose:
        print(f"[VL] Found {len(clips)} clips, {len(processed)} already processed")
    
    new_count = 0
    for clip in clips:
        clip_id = get_clip_id(clip)
        
        if clip_id in processed:
            continue
        
        result = process_clip_with_vl(clip, verbose=verbose)
        if result:
            processed[clip_id] = result
            save_vl_processed(processed)
            new_count += 1
            if verbose:
                print(f"[VL] Saved result for: {clip_id}")
    
    return new_count


def continuous_process(interval: float = 2.0, verbose: bool = True) -> None:
    """
    Continuously watch for and process new clips.
    
    Args:
        interval: Polling interval in seconds
        verbose: Whether to print progress
    """
    print(f"[VL] Starting continuous processing (interval={interval}s)")
    print(f"[VL] Model: {OLLAMA_MODEL}")
    print(f"[VL] Output: {VL_PROCESSED_JSON}")
    print("[VL] Press Ctrl+C to stop\n")
    
    # Check Ollama health
    if not check_ollama_health():
        print(f"[ERROR] Cannot connect to Ollama at {OLLAMA_BASE_URL}")
        sys.exit(1)
    
    processed = load_vl_processed()
    
    try:
        while True:
            clips = get_clips()
            
            for clip in clips:
                clip_id = get_clip_id(clip)
                
                if clip_id in processed:
                    continue
                
                if verbose:
                    print(f"\n[VL] New clip detected: {Path(clip.clip_path).name}")
                
                result = process_clip_with_vl(clip, verbose=verbose)
                if result:
                    processed[clip_id] = result
                    save_vl_processed(processed)
                    if verbose:
                        print(f"[VL] ✓ Processed and saved")
                        print(f"[VL] Summary: {result.summary}")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n[VL] Stopped continuous processing")
        print(f"[VL] Total processed: {len(processed)}")


def run_testrun(verbose: bool = True) -> None:
    """Run a single test with the latest clip."""
    print("=" * 60)
    print("VL Processing Test Run")
    print("=" * 60)
    
    # Check Ollama
    print("\n[1/3] Checking Ollama...")
    if not check_ollama_health():
        print(f"[ERROR] Cannot connect to Ollama at {OLLAMA_BASE_URL}")
        sys.exit(1)
    print(f"[OK] Ollama healthy at {OLLAMA_BASE_URL}")
    print(f"[OK] Model: {OLLAMA_MODEL}")
    
    # Get latest clip
    print("\n[2/3] Loading latest clip...")
    clips = get_clips()
    if not clips:
        print("[ERROR] No clips found in data.json")
        sys.exit(1)
    
    clip = clips[-1]
    print(f"[OK] Clip: {Path(clip.clip_path).name}")
    print(f"     Duration: {clip.duration:.1f}s")
    print(f"     Speakers: {', '.join(clip.speakers) or 'none'}")
    print(f"     Transcription: {clip.text[:50]}..." if len(clip.text) > 50 else f"     Transcription: {clip.text}")
    
    # Process
    print("\n[3/3] Processing with VL model...")
    start_time = time.time()
    result = process_clip_with_vl(clip, verbose=False)
    elapsed = time.time() - start_time
    
    if result:
        print(f"\n[OK] Processing completed in {elapsed:.2f}s")
        print("-" * 40)
        print(f"Scene: {result.scene}")
        print(f"Speech: {result.speech}")
        print(f"Summary: {result.summary}")
        print("-" * 40)
        
        # Save result
        processed = load_vl_processed()
        clip_id = get_clip_id(clip)
        processed[clip_id] = result
        save_vl_processed(processed)
        print(f"\n[OK] Saved to {VL_PROCESSED_JSON}")
    else:
        print("[ERROR] Processing failed")
        sys.exit(1)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="VL Processing - Fast Vision-Language Processing for Streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vl_processing.py --testrun        # Test with latest clip
  python vl_processing.py --process        # Process all unprocessed clips  
  python vl_processing.py --continuous     # Watch and process new clips
  python vl_processing.py --continuous --interval 1.0  # Faster polling
""",
    )
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--testrun",
        action="store_true",
        help="Run a single test with the latest clip",
    )
    mode_group.add_argument(
        "--process",
        action="store_true",
        help="Process all unprocessed clips once",
    )
    mode_group.add_argument(
        "--continuous",
        action="store_true",
        help="Continuously watch for and process new clips",
    )
    
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Polling interval for continuous mode (seconds, default: 2.0)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity",
    )
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    if args.testrun:
        run_testrun(verbose=verbose)
    elif args.process:
        count = process_new_clips(verbose=verbose)
        print(f"\n[VL] Processed {count} new clips")
    elif args.continuous:
        continuous_process(interval=args.interval, verbose=verbose)


if __name__ == "__main__":
    main()
