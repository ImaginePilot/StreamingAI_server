#!/usr/bin/env python3
"""
FunASR Paraformer-Large Online Streaming ASR Module
====================================================

Real-time speech recognition with timestamps using FunASR's Paraformer-Large Online model.
Provides timestamps that can be used as markers to cut videos for VL processing.

Streaming Model: iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online
Offline Model: paraformer-zh (with fsmn-vad + ct-punc)

Paraformer is a non-autoregressive end-to-end speech recognition model:
- Parallel output of entire sentence for high GPU efficiency
- Matches SOTA autoregressive model performance
- CIF-based Predictor for accurate character count prediction
- Sampler module for enhanced semantic understanding
- Bidirectional Decoder for better context modeling

Architecture:
- Encoder: Self-attention / Conformer / SAN-M
- Predictor: Two-layer FFN based on CIF (Continuous Integrate-and-Fire)
- Sampler: Parameter-free module for semantic vector transformation
- Decoder: Bidirectional attention (unlike unidirectional autoregressive)
- Loss: CE + MWER + MAE for Predictor

Features:
- Real-time streaming with ~600ms latency
- Sentence-based segmentation (5-15 seconds)
- VAD for voice activity detection
- Punctuation restoration for proper sentences
- Lookback attention (encoder=4, decoder=1)

Usage:
    from streaming_asr import StreamingASR
    
    asr = StreamingASR()
    
    # Process audio chunks in real-time
    for chunk in audio_stream:
        results = asr.process_chunk(chunk)
        for result in results:
            print(f\"[{result.start_time:.2f}-{result.end_time:.2f}] {result.text}\")
    
    # Finalize and get complete results with VAD + punctuation
    final_results = asr.finalize()
"""

import os
import sys
import time
import json
import threading
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator
from dataclasses import dataclass, asdict
from datetime import datetime
import queue

# Configuration
SCRIPT_DIR = Path(__file__).parent
CACHE_DIR = SCRIPT_DIR / "cache"
TRANSCRIPTS_DIR = SCRIPT_DIR / "transcripts"

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True)
TRANSCRIPTS_DIR.mkdir(exist_ok=True)


@dataclass
class ASRSegment:
    """A single ASR segment with timestamp."""
    text: str
    start_time: float  # seconds from stream start
    end_time: float    # seconds from stream start
    confidence: float = 1.0
    is_final: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ASRResult:
    """Complete ASR result with segments."""
    segments: List[ASRSegment]
    full_text: str
    duration: float
    timestamp: float  # absolute timestamp when processed
    
    def to_dict(self) -> Dict:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "full_text": self.full_text,
            "duration": self.duration,
            "timestamp": self.timestamp
        }


class StreamingASR:
    """
    Streaming ASR processor using FunASR Paraformer-Large Online model.
    
    Model: iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online
    
    Uses Paraformer-Large for real-time recognition with:
    - Non-autoregressive parallel decoding for GPU efficiency
    - CIF-based predictor for accurate character count prediction
    - Bidirectional decoder for better context modeling
    - Sampler module for enhanced semantic understanding
    
    For finalization, uses paraformer-zh with:
    - VAD (fsmn-vad) for voice activity detection
    - Punctuation restoration (ct-punc)
    
    Processes audio in real-time with ~600ms latency and provides
    timestamps for video cutting.
    """
    
    # Paraformer-Large Online model from ModelScope
    STREAMING_MODEL = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
    
    def __init__(
        self,
        model_name: str = None,  # Will default to streaming model
        sample_rate: int = 16000,
        chunk_ms: int = 600,  # chunk size in milliseconds (600ms = 0, 10, 5 config)
        encoder_chunk_look_back: int = 4,  # number of chunks to lookback for encoder self-attention
        decoder_chunk_look_back: int = 1,  # number of encoder chunks to lookback for decoder cross-attention
    ):
        """
        Initialize streaming ASR with Paraformer-Large Online.
        
        Args:
            model_name: FunASR model name (defaults to paraformer-zh-streaming)
            sample_rate: Audio sample rate (16000 for FunASR models)
            chunk_ms: Chunk size in milliseconds for streaming
            encoder_chunk_look_back: Number of chunks for encoder lookback (default 4)
            decoder_chunk_look_back: Number of chunks for decoder lookback (default 1)
        """
        # Default to streaming model
        self.model_name = model_name or self.STREAMING_MODEL
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.encoder_chunk_look_back = encoder_chunk_look_back
        self.decoder_chunk_look_back = decoder_chunk_look_back
        
        # Calculate chunk parameters
        # chunk_size = [0, 10, 5] means 600ms granularity (10*60=600ms)
        # chunk_stride = chunk_size[1] * 960 samples for 16kHz
        if chunk_ms == 600:
            self.chunk_size = [0, 10, 5]  # 600ms streaming
        elif chunk_ms == 480:
            self.chunk_size = [0, 8, 4]   # 480ms streaming
        else:
            # Default to 600ms
            self.chunk_size = [0, 10, 5]
        
        self.chunk_stride = self.chunk_size[1] * 960  # samples per chunk at 16kHz
        
        # Model and cache
        self.model = None
        self.cache = {}
        
        # Stream state
        self.stream_start_time = None
        self.samples_processed = 0
        self.pending_segments: List[ASRSegment] = []
        self.all_segments: List[ASRSegment] = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        print(f"[Paraformer] Initialized with chunk_size={self.chunk_size}, chunk_stride={self.chunk_stride}")
        print(f"[Paraformer] Lookback: encoder={encoder_chunk_look_back}, decoder={decoder_chunk_look_back}")
        print(f"[Paraformer] Streaming model: {self.model_name}")
    
    def _load_model(self):
        """Lazy load the FunASR Paraformer streaming model."""
        if self.model is not None:
            return
        
        print(f"[Paraformer] Loading streaming model: {self.model_name}")
        from funasr import AutoModel
        
        # Load Paraformer streaming model
        self.model = AutoModel(model=self.model_name)
        print(f"[Paraformer] Streaming model loaded successfully")
    
    def start_stream(self, start_timestamp: float = None):
        """
        Start a new streaming session.
        
        Args:
            start_timestamp: Absolute timestamp when stream started (for sync with video)
        """
        with self.lock:
            self._load_model()
            self.cache = {}
            self.stream_start_time = start_timestamp or time.time()
            self.samples_processed = 0
            self.pending_segments = []
            self.all_segments = []
            print(f"[Paraformer] Stream started at {datetime.fromtimestamp(self.stream_start_time)}")
    
    def process_chunk(
        self,
        audio_data: np.ndarray,
        is_final: bool = False
    ) -> List[ASRSegment]:
        """
        Process an audio chunk and return any recognized segments.
        
        Args:
            audio_data: Audio samples as int16 numpy array
            is_final: Whether this is the last chunk of the stream
            
        Returns:
            List of ASRSegment with timestamps
        """
        with self.lock:
            if self.model is None:
                self._load_model()
                if self.stream_start_time is None:
                    self.start_stream()
            
            # Convert to float32 normalized [-1, 1] for FunASR
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data.astype(np.float32)
            
            # Resample if needed (FunASR expects 16kHz)
            # For now, assume input is already at correct sample rate or handle externally
            
            # Process through model with lookback parameters for better accuracy
            try:
                result = self.model.generate(
                    input=audio_float,
                    cache=self.cache,
                    is_final=is_final,
                    chunk_size=self.chunk_size,
                    encoder_chunk_look_back=self.encoder_chunk_look_back,
                    decoder_chunk_look_back=self.decoder_chunk_look_back,
                )
            except Exception as e:
                print(f"[Paraformer] Error processing chunk: {e}")
                return []
            
            # Parse results into segments
            new_segments = []
            
            if result and len(result) > 0:
                for res in result:
                    # FunASR streaming returns text with optional timestamps
                    text = res.get("text", "")
                    
                    if text and text.strip():
                        # Calculate time based on samples processed
                        current_time = self.samples_processed / self.sample_rate
                        
                        # Get timestamps if available
                        timestamp_info = res.get("timestamp", [])
                        
                        if timestamp_info and len(timestamp_info) > 0:
                            # Parse word-level timestamps
                            for ts_item in timestamp_info:
                                if isinstance(ts_item, (list, tuple)) and len(ts_item) >= 3:
                                    start_ms, end_ms, word = ts_item[0], ts_item[1], ts_item[2] if len(ts_item) > 2 else text
                                    seg = ASRSegment(
                                        text=str(word),
                                        start_time=start_ms / 1000.0,
                                        end_time=end_ms / 1000.0,
                                        is_final=is_final
                                    )
                                    new_segments.append(seg)
                        else:
                            # No detailed timestamps, use chunk timing
                            chunk_duration = len(audio_float) / self.sample_rate
                            seg = ASRSegment(
                                text=text,
                                start_time=current_time,
                                end_time=current_time + chunk_duration,
                                is_final=is_final
                            )
                            new_segments.append(seg)
            
            # Update samples count
            self.samples_processed += len(audio_float)
            
            # Store segments
            self.all_segments.extend(new_segments)
            
            return new_segments
    
    def finalize(self) -> ASRResult:
        """
        Finalize the stream and return complete result.
        
        Returns:
            ASRResult with all segments and full text
        """
        with self.lock:
            # Process any remaining audio as final
            if self.cache:
                try:
                    result = self.model.generate(
                        input=np.array([], dtype=np.float32),
                        cache=self.cache,
                        is_final=True,
                        chunk_size=self.chunk_size,
                        encoder_chunk_look_back=self.encoder_chunk_look_back,
                        decoder_chunk_look_back=self.decoder_chunk_look_back,
                    )
                    if result:
                        for res in result:
                            text = res.get("text", "")
                            if text and text.strip():
                                current_time = self.samples_processed / self.sample_rate
                                seg = ASRSegment(
                                    text=text,
                                    start_time=current_time,
                                    end_time=current_time,
                                    is_final=True
                                )
                                self.all_segments.append(seg)
                except Exception as e:
                    print(f"[Paraformer] Error finalizing: {e}")
            
            # Build full text
            full_text = " ".join(seg.text for seg in self.all_segments)
            
            # Calculate duration
            duration = self.samples_processed / self.sample_rate
            
            return ASRResult(
                segments=self.all_segments.copy(),
                full_text=full_text,
                duration=duration,
                timestamp=self.stream_start_time or time.time()
            )
    
    def get_timestamps_for_video_cutting(
        self,
        min_segment_gap: float = 1.5,
        min_segment_duration: float = 3.0
    ) -> List[Tuple[float, float, str]]:
        """
        Get timestamp markers suitable for video cutting.
        
        Groups consecutive speech into segments based on silence gaps.
        
        Args:
            min_segment_gap: Minimum silence gap (seconds) to split segments
            min_segment_duration: Minimum segment duration (seconds)
            
        Returns:
            List of (start_time, end_time, text) tuples
        """
        with self.lock:
            if not self.all_segments:
                return []
            
            # Sort by start time
            sorted_segs = sorted(self.all_segments, key=lambda s: s.start_time)
            
            # Group by gaps
            groups = []
            current_group = [sorted_segs[0]]
            
            for seg in sorted_segs[1:]:
                gap = seg.start_time - current_group[-1].end_time
                if gap > min_segment_gap:
                    groups.append(current_group)
                    current_group = [seg]
                else:
                    current_group.append(seg)
            groups.append(current_group)
            
            # Build result
            results = []
            for group in groups:
                start = group[0].start_time
                end = group[-1].end_time
                
                # Enforce minimum duration
                if end - start < min_segment_duration:
                    end = start + min_segment_duration
                
                text = " ".join(s.text for s in group)
                results.append((start, end, text))
            
            return results


class AudioStreamProcessor:
    """
    Integrates StreamingASR with the audio receiver buffer.
    
    Continuously processes audio from the buffer and maintains
    ASR state with timestamps.
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        target_sample_rate: int = 16000,
        chunk_duration: float = 0.6,  # 600ms chunks
    ):
        """
        Initialize audio stream processor.
        
        Args:
            sample_rate: Input audio sample rate (from audioreciever)
            target_sample_rate: ASR model sample rate (16kHz for FunASR)
            chunk_duration: Chunk duration in seconds
        """
        self.sample_rate = sample_rate
        self.target_sample_rate = target_sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        
        # ASR engine
        self.asr = StreamingASR(sample_rate=target_sample_rate)
        
        # Processing state
        self.buffer = np.array([], dtype=np.int16)
        self.stream_start_time = None
        self.is_running = False
        self.results_queue = queue.Queue()
        
        # Results storage
        self.all_results: List[ASRResult] = []
        self.video_markers: List[Tuple[float, float, str]] = []
        
        # Thread
        self.process_thread = None
        self.lock = threading.Lock()
    
    def _resample(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio from source to target sample rate."""
        if self.sample_rate == self.target_sample_rate:
            return audio
        
        try:
            import scipy.signal as signal
            num_samples = int(len(audio) * self.target_sample_rate / self.sample_rate)
            return signal.resample(audio, num_samples).astype(np.int16)
        except ImportError:
            # Simple linear interpolation fallback
            ratio = self.target_sample_rate / self.sample_rate
            indices = np.arange(0, len(audio), 1/ratio).astype(int)
            indices = np.clip(indices, 0, len(audio) - 1)
            return audio[indices]
    
    def start(self, start_timestamp: float = None):
        """Start processing audio stream."""
        with self.lock:
            self.stream_start_time = start_timestamp or time.time()
            self.is_running = True
            self.buffer = np.array([], dtype=np.int16)
            self.all_results = []
            self.video_markers = []
            self.asr.start_stream(self.stream_start_time)
            print(f"[AudioStreamProcessor] Started at {datetime.fromtimestamp(self.stream_start_time)}")
    
    def add_audio(self, audio_data: np.ndarray) -> List[ASRSegment]:
        """
        Add audio data and process if enough samples available.
        
        Args:
            audio_data: Audio samples as int16 numpy array
            
        Returns:
            List of new ASR segments (may be empty)
        """
        with self.lock:
            if not self.is_running:
                self.start()
            
            # Add to buffer
            self.buffer = np.concatenate([self.buffer, audio_data])
            
            new_segments = []
            chunks_processed = 0
            
            # Process full chunks
            while len(self.buffer) >= self.chunk_samples:
                chunk = self.buffer[:self.chunk_samples]
                self.buffer = self.buffer[self.chunk_samples:]
                
                # Resample to 16kHz
                resampled = self._resample(chunk)
                
                # Process through ASR
                segments = self.asr.process_chunk(resampled, is_final=False)
                new_segments.extend(segments)
                chunks_processed += 1
            
            # Debug: show progress
            if chunks_processed > 0:
                total_time = self.asr.samples_processed / self.asr.sample_rate
                print(f"[Paraformer] Processed {chunks_processed} chunks, total: {total_time:.1f}s, segments: {len(self.asr.all_segments)}, buffer: {len(self.buffer)} samples")
            
            if new_segments:
                for seg in new_segments:
                    print(f"[Paraformer] >> [{seg.start_time:.2f}-{seg.end_time:.2f}] {seg.text}")
            
            return new_segments
    
    def stop(self) -> ASRResult:
        """
        Stop processing and return final results.
        
        Returns:
            Complete ASRResult with all segments
        """
        with self.lock:
            self.is_running = False
            
            # Process remaining buffer
            if len(self.buffer) > 0:
                resampled = self._resample(self.buffer)
                self.asr.process_chunk(resampled, is_final=True)
            
            # Finalize
            result = self.asr.finalize()
            
            # Update video markers
            self.video_markers = self.asr.get_timestamps_for_video_cutting()
            
            return result
    
    def get_video_markers(self) -> List[Tuple[float, float, str]]:
        """
        Get current video cutting markers.
        
        Returns:
            List of (start_time, end_time, text) tuples
        """
        with self.lock:
            return self.asr.get_timestamps_for_video_cutting()


# Singleton instance for easy integration
_stream_processor: Optional[AudioStreamProcessor] = None
_stream_processor_lock = threading.Lock()

# Offline model for finalization (paraformer-zh with VAD + punctuation)
_offline_model = None
_offline_model_lock = threading.Lock()


def get_stream_processor() -> AudioStreamProcessor:
    """Get or create the singleton stream processor."""
    global _stream_processor
    with _stream_processor_lock:
        if _stream_processor is None:
            print("[Paraformer] Creating singleton AudioStreamProcessor (44100Hz -> 16000Hz)")
            _stream_processor = AudioStreamProcessor()
            # Pre-load the model
            print("[Paraformer] Pre-loading Paraformer-Large Online streaming model...")
            _stream_processor.asr._load_model()
            print("[Paraformer] Streaming model ready")
        return _stream_processor


def get_offline_model():
    """Get or create the singleton offline model for finalization."""
    global _offline_model
    with _offline_model_lock:
        if _offline_model is None:
            print("[Paraformer] Pre-loading offline model (Paraformer-Large PyTorch + fsmn-vad + ct-punc)...")
            from funasr import AutoModel
            _offline_model = AutoModel(
                model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                vad_model="fsmn-vad",
                punc_model="ct-punc",
            )
            print("[Paraformer] Offline model ready")
        return _offline_model


def preload_all_models():
    """Preload all ASR models at startup."""
    print("[Paraformer] ════════════════════════════════════════════════════")
    print("[Paraformer] Preloading all ASR models...")
    get_stream_processor()  # Load streaming model
    get_offline_model()     # Load offline model with VAD + punctuation
    print("[Paraformer] All ASR models preloaded!")
    print("[Paraformer] ════════════════════════════════════════════════════")


def reset_stream():
    """
    Reset the streaming ASR state for a new audio stream.
    
    This should be called when starting a new audio stream session
    to clear any pending state from the previous session.
    """
    processor = get_stream_processor()
    processor.start()
    print("[Paraformer] Stream reset for new session")


def process_audio_realtime(audio_data: np.ndarray) -> List[ASRSegment]:
    """
    Process audio data in real-time and return any new segments.
    
    This is the main integration point for audioreciever.py.
    
    Args:
        audio_data: Audio samples as int16 numpy array
        
    Returns:
        List of new ASR segments with timestamps
    """
    processor = get_stream_processor()
    
    # Debug: show input info periodically
    if not hasattr(process_audio_realtime, '_chunk_count'):
        process_audio_realtime._chunk_count = 0
    process_audio_realtime._chunk_count += 1
    
    if process_audio_realtime._chunk_count % 50 == 1:  # Every 50 chunks
        print(f"[Paraformer] Receiving audio: {len(audio_data)} samples, buffer: {len(processor.buffer)}, running: {processor.is_running}")
    
    return processor.add_audio(audio_data)


def get_current_video_markers() -> List[Tuple[float, float, str]]:
    """
    Get current video cutting markers from ongoing transcription.
    
    Returns:
        List of (start_time, end_time, text) tuples
    """
    processor = get_stream_processor()
    return processor.get_video_markers()


def finalize_stream() -> ASRResult:
    """
    Finalize the current stream and get complete results.
    
    Returns:
        Complete ASRResult with all segments
    """
    processor = get_stream_processor()
    return processor.stop()


def finalize_and_get_transcript(audio_data: bytes, start_timestamp: float) -> Optional[Dict]:
    """
    Process complete audio segment with offline Paraformer for best accuracy.
    
    Uses paraformer-zh with:
    - fsmn-vad: Voice Activity Detection to find speech boundaries
    - ct-punc: Punctuation restoration for proper sentences
    
    This is called when a sentence boundary is detected to get the
    final, high-accuracy transcript.
    
    Args:
        audio_data: Raw audio bytes (44100Hz, 16-bit, mono)
        start_timestamp: When this segment started
        
    Returns:
        Dict with 'full_text', 'segments', 'duration'
    """
    try:
        import scipy.signal as signal
        
        # Convert bytes to numpy array
        samples = np.frombuffer(audio_data, dtype=np.int16)
        
        # Resample from 44100Hz to 16000Hz
        num_samples = int(len(samples) * 16000 / 44100)
        resampled = signal.resample(samples, num_samples).astype(np.float32) / 32768.0
        
        duration = len(samples) / 44100.0
        
        print(f"[Paraformer] Finalizing {duration:.1f}s audio with VAD + Punctuation...")
        
        # Use preloaded offline model (paraformer-zh with VAD and punctuation)
        model = get_offline_model()
        
        result = model.generate(
            input=resampled,
            batch_size_s=300,
        )
        
        if not result or len(result) == 0:
            print(f"[Paraformer] No result from offline processing")
            return None
        
        # Parse result
        segments = []
        full_text = ""
        
        for res in result:
            text = res.get("text", "")
            if text:
                full_text += text + " "
                
                # Get timestamps if available
                timestamp_info = res.get("timestamp", [])
                if timestamp_info:
                    for ts_item in timestamp_info:
                        if isinstance(ts_item, (list, tuple)) and len(ts_item) >= 2:
                            start_ms = ts_item[0]
                            end_ms = ts_item[1]
                            word = ts_item[2] if len(ts_item) > 2 else ""
                            segments.append({
                                "text": str(word),
                                "start_time": start_ms / 1000.0,
                                "end_time": end_ms / 1000.0,
                                "is_final": True
                            })
                else:
                    # No detailed timestamps, create single segment with full text
                    segments.append({
                        "text": text,
                        "start_time": 0,
                        "end_time": duration,
                        "is_final": True
                    })
        
        full_text = full_text.strip()
        print(f"[Paraformer] Final result: \"{full_text[:100]}{'...' if len(full_text) > 100 else ''}\"")
        
        return {
            "full_text": full_text,
            "segments": segments,
            "duration": duration,
            "timestamp": start_timestamp
        }
        
    except Exception as e:
        print(f"[Paraformer] Error in offline processing: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Transcript Cache - Store ASR results per audio fragment
# ============================================================================

# Global transcript cache: {filename: {segments: [...], full_text: str, timestamp: float}}
_transcript_cache: Dict[str, Dict] = {}
_transcript_cache_lock = threading.Lock()
_transcript_cache_file = CACHE_DIR / "transcript_cache.json"


def _load_transcript_cache():
    """Load transcript cache from disk."""
    global _transcript_cache
    try:
        if _transcript_cache_file.exists():
            with open(_transcript_cache_file, 'r', encoding='utf-8') as f:
                _transcript_cache = json.load(f)
            print(f"[Paraformer] Loaded {len(_transcript_cache)} cached transcripts")
    except Exception as e:
        print(f"[Paraformer] Failed to load transcript cache: {e}")
        _transcript_cache = {}


def _save_transcript_cache():
    """Save transcript cache to disk."""
    try:
        with open(_transcript_cache_file, 'w', encoding='utf-8') as f:
            json.dump(_transcript_cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Paraformer] Failed to save transcript cache: {e}")


def save_fragment_transcript(
    filename: str,
    segments: List[Dict],
    start_timestamp: float,
    duration: float,
    full_text: str = None
):
    """
    Save ASR transcript for an audio fragment.
    
    Called when a fragment is finalized to store the ASR results.
    
    Args:
        filename: Fragment filename (e.g., "segment_20260131_152410_0000.wav")
        segments: List of segment dicts with text, start_time, end_time
        start_timestamp: Absolute timestamp when fragment started
        duration: Fragment duration in seconds
        full_text: Optional pre-computed full text (if not provided, computed from segments)
    """
    with _transcript_cache_lock:
        if full_text is None:
            full_text = " ".join(s.get("text", "") for s in segments if s.get("text"))
        
        _transcript_cache[filename] = {
            "segments": segments,
            "full_text": full_text,
            "start_timestamp": start_timestamp,
            "duration": duration,
            "cached_at": time.time()
        }
        _save_transcript_cache()
        
        # Debug output
        print(f"")
        print(f"╔══════════════════════════════════════════════════════════════════╗")
        print(f"║ [Paraformer] Transcript saved: {filename}")
        print(f"╠══════════════════════════════════════════════════════════════════╣")
        print(f"║ Duration: {duration:.1f}s | Segments: {len(segments)}")
        print(f"║ Text: {full_text[:60]}{'...' if len(full_text) > 60 else ''}")
        print(f"╚══════════════════════════════════════════════════════════════════╝")
        print(f"")


def get_fragment_transcript(filename: str) -> Optional[Dict]:
    """
    Get cached transcript for an audio fragment.
    
    Args:
        filename: Fragment filename
        
    Returns:
        Dict with segments, full_text, start_timestamp, duration or None if not cached
    """
    with _transcript_cache_lock:
        return _transcript_cache.get(filename)


def get_cached_segments_for_fragment(filename: str, fragment_start_ts: float) -> Optional[List[Dict]]:
    """
    Get segments with absolute timestamps for a fragment.
    
    Converts relative timestamps in cached segments to absolute timestamps.
    
    Args:
        filename: Fragment filename
        fragment_start_ts: Absolute timestamp when fragment started
        
    Returns:
        List of segment dicts with abs_start, abs_end, text, etc. or None
    """
    cached = get_fragment_transcript(filename)
    if not cached:
        print(f"[Paraformer] Cache MISS for {filename}")
        return None
    
    segments = cached.get("segments", [])
    enriched = []
    for seg in segments:
        enriched.append({
            "abs_start": fragment_start_ts + seg.get("start_time", 0),
            "abs_end": fragment_start_ts + seg.get("end_time", 0),
            "speaker": "stream",  # Streaming ASR doesn't do speaker diarization
            "text": seg.get("text", ""),
            "language": "auto",
            "emotion": None,
        })
    
    print(f"[Paraformer] Cache HIT for {filename}: {len(enriched)} segments, text: '{cached.get('full_text', '')[:50]}...'")
    return enriched


def has_cached_transcript(filename: str) -> bool:
    """Check if a fragment has a cached transcript."""
    with _transcript_cache_lock:
        return filename in _transcript_cache


def clear_transcript_cache():
    """Clear the transcript cache."""
    global _transcript_cache
    with _transcript_cache_lock:
        _transcript_cache = {}
        _save_transcript_cache()


# Load cache on module import
_load_transcript_cache()


# ============================================================================
# File Processing (for compatibility with existing workflow)
# ============================================================================

def process_audio_file(
    audio_path: str,
    return_timestamps: bool = True
) -> Optional[ASRResult]:
    """
    Process an audio file and return transcription with timestamps.
    
    This provides compatibility with the existing fragment-based workflow.
    
    Args:
        audio_path: Path to audio file (WAV)
        return_timestamps: Whether to include detailed timestamps
        
    Returns:
        ASRResult with segments and timestamps
    """
    try:
        import wave
        
        # Read audio file
        with wave.open(audio_path, 'rb') as wav:
            sample_rate = wav.getframerate()
            n_frames = wav.getnframes()
            audio_bytes = wav.readframes(n_frames)
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Create processor
        processor = AudioStreamProcessor(sample_rate=sample_rate)
        processor.start()
        
        # Process in chunks
        chunk_size = int(sample_rate * 0.6)  # 600ms chunks
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            processor.add_audio(chunk)
        
        # Finalize
        result = processor.stop()
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Failed to process audio file {audio_path}: {e}")
        return None


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """CLI interface for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FunASR Streaming ASR")
    parser.add_argument("--file", type=str, help="Process audio file")
    parser.add_argument("--test", action="store_true", help="Run test with synthetic audio")
    args = parser.parse_args()
    
    if args.file:
        print(f"Processing file: {args.file}")
        result = process_audio_file(args.file)
        if result:
            print(f"\nFull text: {result.full_text}")
            print(f"\nSegments ({len(result.segments)}):")
            for seg in result.segments:
                print(f"  [{seg.start_time:.2f}-{seg.end_time:.2f}] {seg.text}")
            
            # Get video markers
            processor = AudioStreamProcessor()
            processor.start()
            # Reprocess for markers
            import wave
            with wave.open(args.file, 'rb') as wav:
                sample_rate = wav.getframerate()
                audio_bytes = wav.readframes(wav.getnframes())
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            processor.add_audio(audio_data)
            result = processor.stop()
            markers = processor.get_video_markers()
            
            print(f"\nVideo cutting markers ({len(markers)}):")
            for start, end, text in markers:
                print(f"  [{start:.2f}-{end:.2f}] {text[:50]}...")
        else:
            print("Failed to process file")
    
    elif args.test:
        print("Running test with synthetic audio...")
        
        # Create synthetic audio (silence with some noise)
        sample_rate = 16000
        duration = 5  # seconds
        audio = (np.random.randn(sample_rate * duration) * 100).astype(np.int16)
        
        processor = AudioStreamProcessor(sample_rate=sample_rate, target_sample_rate=16000)
        processor.start()
        
        # Process in chunks
        chunk_size = int(sample_rate * 0.6)
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            segments = processor.add_audio(chunk)
            if segments:
                for seg in segments:
                    print(f"  [{seg.start_time:.2f}-{seg.end_time:.2f}] {seg.text}")
        
        result = processor.stop()
        print(f"\nTest complete. Duration: {result.duration:.2f}s")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
