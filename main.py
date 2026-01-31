#!/usr/bin/env python3
"""
Main Orchestrator - Unified Pipeline for Streaming, VL Processing, and LLM

This module orchestrates the complete pipeline:
1. Streaming: Audio/Video ingestion and clip generation (streaming.py)
2. VL Processing: Vision-Language model analysis (vl_processing.py)
3. LLM Processing: Context-aware response generation (llm_processing.py)
4. TTS Output: (placeholder for future implementation)

Usage:
    python main.py                    # Start full pipeline
    python main.py --clear            # Clear all data and start fresh
    python main.py --no-streaming     # Run processors without starting streaming servers
    python main.py --vl-only          # Only run VL processing
    python main.py --llm-only         # Only run LLM processing
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent dirs to path for imports
BASE_DIR = Path(__file__).resolve().parent
LLM_DIR = BASE_DIR / "llm"
STREAMING_DIR = BASE_DIR / "streaming"
sys.path.insert(0, str(LLM_DIR))
sys.path.insert(0, str(STREAMING_DIR))

# Import processing modules
from vl_processing import (
    continuous_process as vl_continuous,
    check_ollama_health,
    load_vl_processed,
    get_clips,
    get_clip_id,
    process_clip_with_vl,
    save_vl_processed,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    VL_PROCESSED_JSON,
)

from llm_processing import (
    VLDataReader,
    ContextRetriever,
    LLMProcessor,
    OutputManager,
    LLM_OUTPUT_JSON,
)

# Try to import memory module
try:
    from memory import get_memory, MemoryManager
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

# Configuration
VL_PROCESS_INTERVAL = float(os.getenv("VL_INTERVAL", "2.0"))
LLM_PROCESS_INTERVAL = float(os.getenv("LLM_INTERVAL", "5.0"))
STREAMING_VIDEO_PORT = int(os.getenv("STREAMING_VIDEO_PORT", "5000"))
STREAMING_AUDIO_PORT = int(os.getenv("STREAMING_AUDIO_PORT", "5001"))

# Models to preload
PRELOAD_MODELS = [
    "qwen3-vl:30b",         # LLM processing (with visual frames)
    "qwen3-vl:2b",          # Fast VL pre-processing on secondary server
    "qwen3-embedding:0.6b", # Embeddings
]


def preload_ollama_models(ollama_url: str = OLLAMA_BASE_URL):
    """
    Preload models into Ollama memory with keep_alive=-1.
    This prevents model reload delays during processing.
    """
    import requests
    
    print("[PRELOAD] Warming up Ollama models...")
    
    for model in PRELOAD_MODELS:
        try:
            print(f"[PRELOAD] Loading {model}...", end=" ", flush=True)
            
            # For embedding models, use /api/embeddings
            if "embedding" in model:
                response = requests.post(
                    f"{ollama_url}/api/embeddings",
                    json={"model": model, "prompt": "warmup", "keep_alive": -1},
                    timeout=120
                )
            else:
                # For chat models, use /api/chat with minimal prompt
                response = requests.post(
                    f"{ollama_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False,
                        "keep_alive": -1,
                        "options": {"num_predict": 1}
                    },
                    timeout=120
                )
            
            if response.status_code == 200:
                print("✓")
            else:
                print(f"✗ (status {response.status_code})")
        except Exception as e:
            print(f"✗ ({e})")
    
    print("[PRELOAD] Model warmup complete")


@dataclass
class PipelineStats:
    """Track pipeline statistics."""
    started_at: float
    vl_processed: int = 0
    llm_processed: int = 0
    llm_spoken: int = 0
    llm_silent: int = 0
    errors: int = 0
    
    def to_dict(self):
        return {
            "started_at": datetime.fromtimestamp(self.started_at).isoformat(),
            "uptime_seconds": time.time() - self.started_at,
            "vl_processed": self.vl_processed,
            "llm_processed": self.llm_processed,
            "llm_spoken": self.llm_spoken,
            "llm_silent": self.llm_silent,
            "errors": self.errors,
        }


class Pipeline:
    """
    Main pipeline orchestrator.
    
    Manages:
    - Streaming subprocess (optional)
    - VL processing thread
    - LLM processing thread
    - Memory management (LanceDB + LightRAG)
    """
    
    def __init__(
        self,
        run_streaming: bool = True,
        run_vl: bool = True,
        run_llm: bool = True,
        clear_data: bool = False,
        verbose: bool = True,
        vl_interval: float = VL_PROCESS_INTERVAL,
        llm_interval: float = LLM_PROCESS_INTERVAL,
        preload_models: bool = True,
    ):
        self.run_streaming = run_streaming
        self.run_vl = run_vl
        self.run_llm = run_llm
        self.clear_data = clear_data
        self.verbose = verbose
        self.vl_interval = vl_interval
        self.llm_interval = llm_interval
        self.preload_models = preload_models
        
        self.stop_event = threading.Event()
        self.streaming_process: Optional[subprocess.Popen] = None
        self.vl_thread: Optional[threading.Thread] = None
        self.llm_thread: Optional[threading.Thread] = None
        
        self.stats = PipelineStats(started_at=time.time())
        self.memory: Optional[MemoryManager] = None
        
        # Components
        self.vl_reader = VLDataReader()
        self.context_retriever = ContextRetriever()
        self.llm_processor = LLMProcessor()
        self.output_manager = OutputManager()
    
    def _log(self, msg: str, component: str = "MAIN"):
        """Log with timestamp."""
        if self.verbose:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] [{component}] {msg}")
    
    def _init_memory(self):
        """Initialize memory system."""
        if not MEMORY_AVAILABLE:
            self._log("Memory module not available", "WARN")
            return
        
        try:
            self.memory = get_memory()
            stats = self.memory.get_stats()
            self._log(f"Memory initialized: {stats['short_term']['active_count']} short-term entries")
            if stats['long_term']['available']:
                self._log("LightRAG connected for long-term memory")
            else:
                self._log("LightRAG not available", "WARN")
        except Exception as e:
            self._log(f"Memory init failed: {e}", "ERROR")
    
    def _start_streaming(self):
        """Start streaming subprocess."""
        if not self.run_streaming:
            return
        
        streaming_script = STREAMING_DIR / "streaming.py"
        if not streaming_script.exists():
            self._log(f"Streaming script not found: {streaming_script}", "ERROR")
            return
        
        cmd = [
            sys.executable, str(streaming_script),
            "--video-port", str(STREAMING_VIDEO_PORT),
            "--audio-port", str(STREAMING_AUDIO_PORT),
        ]
        
        if self.clear_data:
            cmd.append("--clear")
        
        self._log(f"Starting streaming servers on ports {STREAMING_VIDEO_PORT}/{STREAMING_AUDIO_PORT}")
        
        try:
            self.streaming_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            # Start thread to forward streaming output
            def forward_output():
                for line in self.streaming_process.stdout:
                    if self.verbose:
                        print(f"[STREAM] {line.rstrip()}")
            
            threading.Thread(target=forward_output, daemon=True).start()
            
        except Exception as e:
            self._log(f"Failed to start streaming: {e}", "ERROR")
    
    def _vl_processing_loop(self):
        """VL processing loop - runs in thread."""
        self._log("VL processing started", "VL")
        self._log(f"Model: {OLLAMA_MODEL}", "VL")
        self._log(f"Interval: {self.vl_interval}s", "VL")
        
        # Check Ollama
        if not check_ollama_health():
            self._log(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}", "VL-ERROR")
            return
        
        vl_processed = load_vl_processed()
        failed_clips = set()  # Track clips that failed to avoid retry loops
        
        while not self.stop_event.is_set():
            try:
                clips = get_clips()
                
                # Find first unprocessed clip (process one at a time for responsiveness)
                new_clip = None
                for clip in clips:
                    clip_id = get_clip_id(clip)
                    if clip_id not in vl_processed and clip_id not in failed_clips:
                        # Check if file exists
                        if not Path(clip.clip_path).exists():
                            self._log(f"Skipping missing file: {Path(clip.clip_path).name}", "VL-WARN")
                            failed_clips.add(clip_id)
                            continue
                        new_clip = clip
                        break  # Process newest unprocessed clip
                
                if new_clip and not self.stop_event.is_set():
                    clip_id = get_clip_id(new_clip)
                    self._log(f"Processing: {Path(new_clip.clip_path).name}", "VL")
                    
                    result = process_clip_with_vl(new_clip, verbose=False)
                    if result:
                        vl_processed[clip_id] = result
                        save_vl_processed(vl_processed)
                        self.stats.vl_processed += 1
                        
                        self._log(f"Scene: {result.scene[:60]}..." if len(result.scene) > 60 else f"Scene: {result.scene}", "VL")
                        
                        # Save to memory
                        if self.memory:
                            try:
                                self.memory.add_vl_result(
                                    clip_path=result.clip_path,
                                    transcription=result.transcription,
                                    scene=result.scene,
                                    speech=result.speech,
                                    summary=result.summary,
                                    speakers=result.speakers,
                                    duration=result.duration,
                                    model=result.model,
                                )
                            except Exception as e:
                                self._log(f"Memory save failed: {e}", "VL-WARN")
                        
                        # Immediately check for more clips (no wait)
                        continue
                    else:
                        self.stats.errors += 1
                        failed_clips.add(clip_id)  # Don't retry failed processing
                
                # Only wait when no new clips to process
                self.stop_event.wait(self.vl_interval)
                
            except Exception as e:
                self._log(f"Error: {e}", "VL-ERROR")
                self.stats.errors += 1
                self.stop_event.wait(self.vl_interval)
        
        self._log("VL processing stopped", "VL")
    
    def _llm_processing_loop(self):
        """LLM processing loop - runs in thread."""
        self._log("LLM processing started", "LLM")
        self._log(f"Interval: {self.llm_interval}s", "LLM")
        
        while not self.stop_event.is_set():
            try:
                # Get unprocessed VL results
                unprocessed = self.vl_reader.get_unprocessed()
                
                if unprocessed:
                    self._log(f"Processing {len(unprocessed)} VL result(s)", "LLM")
                    
                    # Get combined context
                    context = self.context_retriever.get_combined_context(unprocessed)
                    
                    # Process with LLM
                    result = self.llm_processor.process(context)
                    
                    if result:
                        self.output_manager.add_output(result)
                        self.vl_reader.mark_processed(unprocessed)
                        self.stats.llm_processed += 1
                        
                        if result.should_speak:
                            self.stats.llm_spoken += 1
                            self._log(f"Reply: {result.reply}", "LLM")
                            self._log(f"Emotion: {result.emotion} | Priority: {result.priority}", "LLM")
                            
                            # TODO: Send to TTS here
                            # self._send_to_tts(result)
                        else:
                            self.stats.llm_silent += 1
                            self._log(f"(silent) Think: {result.think[:50]}..." if len(result.think) > 50 else f"(silent) Think: {result.think}", "LLM")
                    else:
                        self.stats.errors += 1
                
            except Exception as e:
                self._log(f"Error: {e}", "LLM-ERROR")
                self.stats.errors += 1
            
            # Wait for next interval
            self.stop_event.wait(self.llm_interval)
        
        self._log("LLM processing stopped", "LLM")
    
    def start(self):
        """Start the pipeline."""
        self._log("=" * 50)
        self._log("Starting Pipeline")
        self._log("=" * 50)
        
        # Preload models to avoid reload delays
        if self.preload_models:
            preload_ollama_models()
        
        # Initialize memory
        self._init_memory()
        
        # Start streaming (subprocess)
        if self.run_streaming:
            self._start_streaming()
            time.sleep(2)  # Give streaming time to initialize
        
        # Start VL processing thread
        if self.run_vl:
            self.vl_thread = threading.Thread(target=self._vl_processing_loop, daemon=True)
            self.vl_thread.start()
        
        # Start LLM processing thread
        if self.run_llm:
            self.llm_thread = threading.Thread(target=self._llm_processing_loop, daemon=True)
            self.llm_thread.start()
        
        self._log("Pipeline started. Press Ctrl+C to stop.")
        self._log("=" * 50)
    
    def stop(self):
        """Stop the pipeline."""
        self._log("Stopping pipeline...")
        
        self.stop_event.set()
        
        # Stop VL thread
        if self.vl_thread and self.vl_thread.is_alive():
            self.vl_thread.join(timeout=3.0)
        
        # Stop LLM thread
        if self.llm_thread and self.llm_thread.is_alive():
            self.llm_thread.join(timeout=3.0)
        
        # Stop streaming subprocess
        if self.streaming_process:
            self.streaming_process.terminate()
            try:
                self.streaming_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.streaming_process.kill()
        
        self._log("Pipeline stopped")
        self._print_stats()
    
    def _print_stats(self):
        """Print final statistics."""
        stats = self.stats.to_dict()
        print("\n" + "=" * 50)
        print("Pipeline Statistics")
        print("=" * 50)
        print(f"  Uptime: {stats['uptime_seconds']:.1f}s")
        print(f"  VL Processed: {stats['vl_processed']}")
        print(f"  LLM Processed: {stats['llm_processed']}")
        print(f"    - Spoken: {stats['llm_spoken']}")
        print(f"    - Silent: {stats['llm_silent']}")
        print(f"  Errors: {stats['errors']}")
        print("=" * 50)
    
    def run(self):
        """Run the pipeline until interrupted."""
        self.start()
        
        try:
            while not self.stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def print_status():
    """Print current status of all components."""
    print("\n" + "=" * 50)
    print("Pipeline Status")
    print("=" * 50)
    
    # Check Ollama
    if check_ollama_health():
        print(f"✓ Ollama: {OLLAMA_BASE_URL}")
    else:
        print(f"✗ Ollama: {OLLAMA_BASE_URL} (not reachable)")
    
    # Check VL processed
    if VL_PROCESSED_JSON.exists():
        with open(VL_PROCESSED_JSON) as f:
            data = json.load(f)
        print(f"✓ VL Results: {data.get('count', 0)} processed")
    else:
        print("- VL Results: No data yet")
    
    # Check LLM output
    if LLM_OUTPUT_JSON.exists():
        with open(LLM_OUTPUT_JSON) as f:
            data = json.load(f)
        outputs = data.get("outputs", [])
        print(f"✓ LLM Outputs: {len(outputs)} generated")
    else:
        print("- LLM Outputs: No data yet")
    
    # Check memory
    if MEMORY_AVAILABLE:
        try:
            memory = get_memory()
            stats = memory.get_stats()
            print(f"✓ Memory: {stats['short_term']['active_count']} short-term entries")
            if stats['long_term']['available']:
                print("✓ LightRAG: Connected")
            else:
                print("- LightRAG: Not available")
        except Exception as e:
            print(f"✗ Memory: Error - {e}")
    else:
        print("- Memory: Module not available")
    
    # Check streaming data
    data_json = STREAMING_DIR / "cache" / "data.json"
    if data_json.exists():
        with open(data_json) as f:
            data = json.load(f)
        clips = data.get("conversations", [])
        print(f"✓ Streaming: {len(clips)} clips in cache")
    else:
        print("- Streaming: No clips yet")
    
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Main Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start full pipeline
  python main.py --clear            # Clear all data and start fresh
  python main.py --no-streaming     # Run processors only (streaming already running)
  python main.py --vl-only          # Only run VL processing
  python main.py --llm-only         # Only run LLM processing
  python main.py --status           # Show current status
        """
    )
    
    parser.add_argument("--clear", action="store_true",
                        help="Clear all data before starting")
    parser.add_argument("--no-streaming", action="store_true",
                        help="Don't start streaming servers (assume already running)")
    parser.add_argument("--vl-only", action="store_true",
                        help="Only run VL processing")
    parser.add_argument("--llm-only", action="store_true",
                        help="Only run LLM processing")
    parser.add_argument("--status", action="store_true",
                        help="Show current pipeline status and exit")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Reduce output verbosity")
    parser.add_argument("--vl-interval", type=float, default=VL_PROCESS_INTERVAL,
                        help=f"VL processing interval in seconds (default: {VL_PROCESS_INTERVAL})")
    parser.add_argument("--llm-interval", type=float, default=LLM_PROCESS_INTERVAL,
                        help=f"LLM processing interval in seconds (default: {LLM_PROCESS_INTERVAL})")
    parser.add_argument("--no-preload", action="store_true",
                        help="Skip model preloading (faster startup, slower first inference)")
    
    args = parser.parse_args()
    
    # Status mode
    if args.status:
        print_status()
        return
    
    # Update intervals from args
    vl_interval = args.vl_interval
    llm_interval = args.llm_interval
    
    # Determine what to run
    run_streaming = not args.no_streaming
    run_vl = True
    run_llm = True
    
    if args.vl_only:
        run_streaming = False
        run_llm = False
    elif args.llm_only:
        run_streaming = False
        run_vl = False
    
    # Create and run pipeline
    pipeline = Pipeline(
        run_streaming=run_streaming,
        run_vl=run_vl,
        run_llm=run_llm,
        clear_data=args.clear,
        verbose=not args.quiet,
        vl_interval=vl_interval,
        llm_interval=llm_interval,
        preload_models=not args.no_preload,
    )
    
    # Handle signals
    def signal_handler(sig, frame):
        pipeline.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    pipeline.run()


if __name__ == "__main__":
    main()
