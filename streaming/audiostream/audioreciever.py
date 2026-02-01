#!/usr/bin/env python3
"""
Audio Stream Receiver Server

Receives streaming audio from an Android app via HTTP POST requests
and saves the stream into 5-second fragments in the cache folder.

Also integrates with FunASR streaming for real-time transcription with timestamps.
"""

import os
import io
import time
import wave
import threading
import queue
from datetime import datetime
from flask import Flask, request, jsonify

import numpy as np

# Import streaming ASR module
try:
    from . import streaming_asr
    STREAMING_ASR_AVAILABLE = True
    print("[audioreciever] Streaming ASR imported via relative import")
except ImportError:
    try:
        import streaming_asr
        STREAMING_ASR_AVAILABLE = True
        print("[audioreciever] Streaming ASR imported via direct import")
    except ImportError as e:
        STREAMING_ASR_AVAILABLE = False
        print(f"[audioreciever] WARN: Streaming ASR not available: {e}")

# Configuration
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 5001
FRAGMENT_DURATION = 5  # seconds
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
DEBUG = True  # Enable debug logging
MAX_FRAGMENTS = 50  # Maximum number of fragment files to keep
METADATA_FILE = os.path.join(CACHE_DIR, 'fragments.json')  # JSON file tracking fragments
AUDIO_QUEUE_SIZE = 500  # Maximum audio chunks to buffer

# Default audio settings (can be overridden by client)
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2  # 16-bit audio

# Stats tracking
class Stats:
    def __init__(self):
        self.chunks_received = 0
        self.bytes_received = 0
        self.last_chunk_time = 0
        self.start_time = time.time()
        self.chunk_intervals = []  # Track time between chunks
    
    def log_chunk(self, size):
        current_time = time.time()
        
        # Track chunk interval
        if self.last_chunk_time > 0:
            interval = (current_time - self.last_chunk_time) * 1000  # in ms
            self.chunk_intervals.append(interval)
            # Keep only last 100 intervals
            if len(self.chunk_intervals) > 100:
                self.chunk_intervals.pop(0)
        
        self.chunks_received += 1
        self.bytes_received += size
        self.last_chunk_time = current_time
        
        if DEBUG and self.chunks_received % 50 == 0:  # Log every 50 chunks
            elapsed = current_time - self.start_time
            rate = self.bytes_received / elapsed if elapsed > 0 else 0
            
            # Calculate timing stats
            if self.chunk_intervals:
                avg_interval = sum(self.chunk_intervals) / len(self.chunk_intervals)
                min_interval = min(self.chunk_intervals)
                max_interval = max(self.chunk_intervals)
                jitter = max_interval - min_interval
                print(f"[STATS] Chunks: {self.chunks_received}, Rate: {rate/1024:.1f} KB/s, "
                      f"Interval: {avg_interval:.1f}ms (min:{min_interval:.1f}, max:{max_interval:.1f}, jitter:{jitter:.1f}ms)")
            else:
                print(f"[STATS] Chunks: {self.chunks_received}, Bytes: {self.bytes_received/1024:.1f}KB, Rate: {rate/1024:.1f} KB/s")
    
    def reset(self):
        self.chunks_received = 0
        self.bytes_received = 0
        self.last_chunk_time = 0
        self.start_time = time.time()
        self.chunk_intervals = []


stats = Stats()

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

app = Flask(__name__)


# Audio Buffer for real-time access
class AudioBuffer:
    """Thread-safe audio buffer for sharing audio data."""
    
    def __init__(self, max_seconds=10, sample_rate=DEFAULT_SAMPLE_RATE, channels=DEFAULT_CHANNELS):
        self.lock = threading.Lock()
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_samples = max_seconds * sample_rate * channels
        self.buffer = np.array([], dtype=np.int16)
        self.last_update = 0
    
    def add_audio(self, audio_data):
        """Add audio samples to the buffer."""
        with self.lock:
            # Convert bytes to numpy array if needed
            if isinstance(audio_data, bytes):
                samples = np.frombuffer(audio_data, dtype=np.int16)
            else:
                samples = audio_data
            
            # Append to buffer
            self.buffer = np.concatenate([self.buffer, samples])
            
            # Trim to max size
            if len(self.buffer) > self.max_samples:
                self.buffer = self.buffer[-self.max_samples:]
            
            self.last_update = time.time()
    
    def get_audio(self, seconds=None):
        """Get audio from the buffer."""
        with self.lock:
            if seconds is None:
                return self.buffer.copy()
            
            num_samples = int(seconds * self.sample_rate * self.channels)
            return self.buffer[-num_samples:].copy() if len(self.buffer) > 0 else np.array([], dtype=np.int16)
    
    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer = np.array([], dtype=np.int16)


audio_buffer = AudioBuffer()


# Audio Processing Queue
class AudioProcessingQueue:
    """
    Async audio processing queue to handle burst traffic.
    Audio chunks are quickly accepted and queued, then processed in background.
    """
    
    def __init__(self, max_size=AUDIO_QUEUE_SIZE):
        self.queue = queue.Queue(maxsize=max_size)
        self.running = False
        self.thread = None
        self.chunks_queued = 0
        self.chunks_processed = 0
        self.chunks_dropped = 0
    
    def start(self, stream_manager, audio_buffer_ref):
        """Start the background processing thread."""
        self.stream_manager = stream_manager
        self.audio_buffer = audio_buffer_ref
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        print("[INFO] Audio processing queue started")
    
    def stop(self):
        """Stop the processing thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def add_chunk(self, audio_data):
        """
        Add an audio chunk to the processing queue.
        Returns immediately, chunk is processed in background.
        """
        try:
            self.queue.put_nowait(audio_data)
            self.chunks_queued += 1
            return True
        except queue.Full:
            # Queue is full, drop the oldest chunk and add new one
            try:
                self.queue.get_nowait()
                self.chunks_dropped += 1
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(audio_data)
                self.chunks_queued += 1
                return True
            except queue.Full:
                self.chunks_dropped += 1
                return False
    
    def _process_loop(self):
        """Background thread that processes queued audio chunks."""
        print(f"[AudioProcessingQueue] _process_loop started, STREAMING_ASR_AVAILABLE={STREAMING_ASR_AVAILABLE}")
        
        while self.running:
            try:
                # Get chunk with timeout to allow clean shutdown
                audio_data = self.queue.get(timeout=0.5)
                
                # Update audio buffer
                self.audio_buffer.add_audio(audio_data)
                
                # Save audio to fragment file
                self.stream_manager.save_audio(audio_data)
                
                # Send to streaming ASR for real-time transcription
                if STREAMING_ASR_AVAILABLE:
                    try:
                        # Convert bytes to numpy array if needed
                        if isinstance(audio_data, bytes):
                            samples = np.frombuffer(audio_data, dtype=np.int16)
                        else:
                            samples = audio_data
                        
                        segments = streaming_asr.process_audio_realtime(samples)
                        if segments:
                            for seg in segments:
                                # Add segment to current fragment's transcript
                                self.stream_manager.add_asr_segment(seg)
                                if DEBUG:
                                    print(f"[ASR] [{seg.start_time:.2f}-{seg.end_time:.2f}] {seg.text}")
                    except Exception as e:
                        if DEBUG:
                            print(f"[WARN] ASR processing error: {e}")
                            import traceback
                            traceback.print_exc()
                
                self.chunks_processed += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                if DEBUG:
                    print(f"[ERROR] Audio processing error: {e}")
    
    def get_stats(self):
        """Get queue statistics."""
        return {
            'queued': self.chunks_queued,
            'processed': self.chunks_processed,
            'dropped': self.chunks_dropped,
            'pending': self.queue.qsize()
        }


audio_queue = AudioProcessingQueue()


# Audio Stream Manager
class AudioStreamManager:
    """Manages audio recording sessions and fragment files."""
    
    def __init__(self):
        self.session_id = None
        self.current_filename = None
        self.fragment_count = 0
        self.fragment_start_time = None
        self.audio_data = bytearray()
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.channels = DEFAULT_CHANNELS
        self.sample_width = DEFAULT_SAMPLE_WIDTH
        self.lock = threading.Lock()
        self.fragments_list = []
        self._load_fragments_metadata()
        
        # ASR segment tracking for current fragment
        self.current_fragment_segments = []
    
    def _load_fragments_metadata(self):
        """Load existing fragments metadata from JSON file."""
        import json
        try:
            if os.path.exists(METADATA_FILE):
                with open(METADATA_FILE, 'r') as f:
                    data = json.load(f)
                    self.fragments_list = data.get('fragments', [])
                    print(f"[INFO] Loaded {len(self.fragments_list)} existing fragments from metadata")
        except Exception as e:
            print(f"[WARN] Could not load fragments metadata: {e}")
            self.fragments_list = []
    
    def _save_fragments_metadata(self):
        """Save fragments metadata to JSON file."""
        import json
        try:
            latest = self.fragments_list[-1] if self.fragments_list else None
            data = {
                'updated': datetime.now().isoformat(),
                'total_fragments': len(self.fragments_list),
                'max_fragments': MAX_FRAGMENTS,
                'latest': latest,
                'fragments': self.fragments_list
            }
            with open(METADATA_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            if DEBUG:
                print(f"[DEBUG] Metadata saved: {len(self.fragments_list)} fragments, latest: {latest['filename'] if latest else 'none'}")
        except Exception as e:
            print(f"[ERROR] Could not save fragments metadata: {e}")
    
    def _cleanup_old_fragments(self):
        """Remove oldest fragments if we exceed the maximum."""
        while len(self.fragments_list) > MAX_FRAGMENTS:
            oldest = self.fragments_list.pop(0)
            old_file = os.path.join(CACHE_DIR, oldest['filename'])
            try:
                if os.path.exists(old_file):
                    os.remove(old_file)
                    print(f"[INFO] Removed old fragment: {oldest['filename']}")
            except Exception as e:
                print(f"[WARN] Could not remove old fragment {oldest['filename']}: {e}")
    
    def start_session(self, session_id=None, sample_rate=None, channels=None, sample_width=None):
        """Start a new recording session."""
        with self.lock:
            self.session_id = session_id or datetime.now().strftime('%Y%m%d_%H%M%S')
            self.sample_rate = sample_rate or DEFAULT_SAMPLE_RATE
            self.channels = channels or DEFAULT_CHANNELS
            self.sample_width = sample_width or DEFAULT_SAMPLE_WIDTH
            self.fragment_count = 0
            self.audio_data = bytearray()
            self.fragment_start_time = time.time()
            
            # Create first fragment
            self._start_new_fragment()
            
            if DEBUG:
                print(f"[DEBUG] Stream session started: {self.session_id}")
            
            return self.session_id
    
    def _start_new_fragment(self):
        """Start a new audio fragment file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.current_filename = f"fragment_{timestamp}_{self.fragment_count:04d}.wav"
        self.audio_data = bytearray()
        self.fragment_start_time = time.time()
        self.current_fragment_segments = []  # Reset ASR segments for new fragment
        
        if DEBUG:
            print(f"[DEBUG] Started new audio fragment: {self.current_filename} "
                  f"({self.sample_rate}Hz, {self.channels}ch, {self.sample_width*8}bit)")
    
    def add_asr_segment(self, segment):
        """Add an ASR segment to the current fragment's transcript."""
        with self.lock:
            if segment:
                self.current_fragment_segments.append({
                    'text': segment.text,
                    'start_time': segment.start_time,
                    'end_time': segment.end_time,
                    'confidence': segment.confidence,
                    'is_final': segment.is_final
                })
    
    def save_audio(self, audio_bytes):
        """Save audio data to the current fragment."""
        if not self.session_id:
            return
        
        with self.lock:
            # Add audio to buffer
            self.audio_data.extend(audio_bytes)
            
            # Check if we need to start a new fragment
            elapsed = time.time() - self.fragment_start_time
            if elapsed >= FRAGMENT_DURATION:
                self._finalize_fragment()
                self.fragment_count += 1
                self._start_new_fragment()
    
    def _finalize_fragment(self):
        """Finalize and save the current fragment."""
        if not self.audio_data:
            return
        
        filepath = os.path.join(CACHE_DIR, self.current_filename)
        
        try:
            # Write WAV file
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.sample_width)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(bytes(self.audio_data))
            
            # Calculate duration
            num_samples = len(self.audio_data) // (self.channels * self.sample_width)
            duration = num_samples / self.sample_rate
            
            # Get file size
            file_size = os.path.getsize(filepath)
            
            # Add to fragments list
            fragment_info = {
                'filename': self.current_filename,
                'session_id': self.session_id,
                'fragment_index': self.fragment_count,
                'start_time': datetime.fromtimestamp(self.fragment_start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': round(duration, 2),
                'sample_count': num_samples,
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'sample_width': self.sample_width,
                'file_size_bytes': file_size
            }
            self.fragments_list.append(fragment_info)
            
            # Cleanup old fragments
            self._cleanup_old_fragments()
            
            # Save metadata
            self._save_fragments_metadata()
            
            # Save ASR transcript for this fragment
            if STREAMING_ASR_AVAILABLE and self.current_fragment_segments:
                try:
                    streaming_asr.save_fragment_transcript(
                        filename=self.current_filename,
                        segments=self.current_fragment_segments,
                        start_timestamp=self.fragment_start_time,
                        duration=duration
                    )
                except Exception as e:
                    if DEBUG:
                        print(f"[WARN] Failed to save ASR transcript: {e}")
            
            if DEBUG:
                print(f"[DEBUG] Audio fragment saved: {filepath} ({num_samples} samples, {duration:.1f}s)")
            
        except Exception as e:
            print(f"[ERROR] Failed to save audio fragment: {e}")
    
    def end_session(self):
        """End the current recording session."""
        with self.lock:
            if self.session_id:
                # Finalize current fragment
                if self.audio_data:
                    self._finalize_fragment()
                
                session_id = self.session_id
                self.session_id = None
                self.current_filename = None
                
                if DEBUG:
                    print(f"[DEBUG] Audio session ended: {session_id}")
                
                return session_id
            return None


stream_manager = AudioStreamManager()


# Flask Routes

@app.route('/', methods=['GET'])
def index():
    """Root endpoint - returns basic server info."""
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


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'active_session': stream_manager.session_id,
        'chunks_received': stats.chunks_received,
        'queue_pending': audio_queue.queue.qsize() if audio_queue else 0
    })


@app.route('/stream/start', methods=['POST'])
def start_stream():
    """
    Start a new streaming session.
    
    Optional JSON body:
    {
        "session_id": "custom_session_id",
        "sample_rate": 44100,
        "channels": 1,
        "sample_width": 2
    }
    """
    if DEBUG:
        print(f"[DEBUG] /stream/start called from {request.remote_addr}")
    
    # Parse optional parameters
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get('session_id')
    sample_rate = data.get('sample_rate', DEFAULT_SAMPLE_RATE)
    channels = data.get('channels', DEFAULT_CHANNELS)
    sample_width = data.get('sample_width', DEFAULT_SAMPLE_WIDTH)
    
    # End any existing session
    if stream_manager.session_id:
        stream_manager.end_session()
    
    # Start new session
    session_id = stream_manager.start_session(
        session_id=session_id,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width
    )
    
    # Reset stats
    stats.reset()
    
    # Update audio buffer settings
    audio_buffer.sample_rate = sample_rate
    audio_buffer.channels = channels
    audio_buffer.clear()
    
    print(f"Started audio session: {session_id}")
    
    return jsonify({
        'status': 'started',
        'session_id': session_id,
        'sample_rate': sample_rate,
        'channels': channels,
        'sample_width': sample_width,
        'fragment_duration': FRAGMENT_DURATION,
        'message': 'Audio stream session started'
    })


@app.route('/stream/stop', methods=['POST'])
def stop_stream():
    """Stop the current streaming session."""
    session_id = stream_manager.end_session()
    
    if session_id:
        print(f"Ended audio session: {session_id}")
        return jsonify({
            'status': 'stopped',
            'session_id': session_id,
            'chunks_received': stats.chunks_received,
            'bytes_received': stats.bytes_received,
            'fragments_saved': stream_manager.fragment_count + 1
        })
    else:
        return jsonify({
            'status': 'no_session',
            'message': 'No active session to stop'
        })


@app.route('/audio/chunk', methods=['POST'])
def receive_audio_chunk():
    """
    Receive an audio chunk.
    
    Body: Raw PCM audio bytes (16-bit signed, little-endian)
    """
    # Check if there's an active session
    if not stream_manager.session_id:
        if DEBUG:
            print(f"[WARN] Audio chunk received without active session from {request.remote_addr}")
    
    if request.content_length == 0:
        if DEBUG:
            print(f"[DEBUG] /audio/chunk called with no data")
        return jsonify({'error': 'No data received'}), 400
    
    audio_data = request.get_data()
    
    if DEBUG and stats.chunks_received % 50 == 0:
        print(f"[DEBUG] /audio/chunk received {len(audio_data)} bytes from {request.remote_addr}")
    
    try:
        stats.log_chunk(len(audio_data))
        
        # Add to async processing queue (fast return)
        queued = audio_queue.add_chunk(audio_data)
        
        if DEBUG and stats.chunks_received % 50 == 0:
            queue_stats = audio_queue.get_stats()
            print(f"[DEBUG] Audio queued: {len(audio_data)} bytes, "
                  f"queue: {queue_stats['pending']}, dropped: {queue_stats['dropped']}")
        
        return jsonify({
            'status': 'received',
            'bytes': len(audio_data),
            'chunk_count': stats.chunks_received,
            'fragment': stream_manager.fragment_count,
            'queued': queued
        })
        
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] Audio chunk error: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/audio/wav', methods=['POST'])
def receive_wav_chunk():
    """
    Receive a WAV-formatted audio chunk.
    Extracts PCM data from WAV container.
    """
    if not stream_manager.session_id:
        if DEBUG:
            print(f"[WARN] WAV chunk received without active session from {request.remote_addr}")
    
    if request.content_length == 0:
        return jsonify({'error': 'No data received'}), 400
    
    wav_data = request.get_data()
    
    try:
        # Parse WAV header and extract PCM data
        wav_buffer = io.BytesIO(wav_data)
        with wave.open(wav_buffer, 'rb') as wav_file:
            # Update stream settings from WAV
            if stream_manager.session_id:
                stream_manager.sample_rate = wav_file.getframerate()
                stream_manager.channels = wav_file.getnchannels()
                stream_manager.sample_width = wav_file.getsampwidth()
            
            # Read PCM data
            pcm_data = wav_file.readframes(wav_file.getnframes())
        
        stats.log_chunk(len(pcm_data))
        
        # Add to processing queue
        queued = audio_queue.add_chunk(pcm_data)
        
        return jsonify({
            'status': 'received',
            'bytes': len(pcm_data),
            'chunk_count': stats.chunks_received,
            'fragment': stream_manager.fragment_count,
            'queued': queued
        })
        
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] WAV chunk error: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/fragments', methods=['GET'])
def list_fragments():
    """List all saved audio fragments."""
    fragments = [f for f in os.listdir(CACHE_DIR) if f.endswith('.wav')]
    fragments.sort()
    
    total_size = sum(
        os.path.getsize(os.path.join(CACHE_DIR, f)) 
        for f in fragments 
        if os.path.isfile(os.path.join(CACHE_DIR, f))
    )
    
    return jsonify({
        'count': len(fragments),
        'max_fragments': MAX_FRAGMENTS,
        'total_size_bytes': total_size,
        'fragment_duration': FRAGMENT_DURATION,
        'fragments': fragments
    })


@app.route('/fragments/latest', methods=['GET'])
def get_latest_fragment():
    """Get information about the latest completed fragment."""
    if stream_manager.fragments_list:
        latest = stream_manager.fragments_list[-1]
        return jsonify({
            'status': 'ok',
            'latest': latest,
            'total_fragments': len(stream_manager.fragments_list),
            'current_recording': stream_manager.current_filename
        })
    else:
        return jsonify({
            'status': 'no_fragments',
            'latest': None,
            'total_fragments': 0,
            'current_recording': stream_manager.current_filename
        })


@app.route('/fragments/metadata', methods=['GET'])
def get_fragments_metadata():
    """Get the full fragments metadata."""
    latest = stream_manager.fragments_list[-1] if stream_manager.fragments_list else None
    return jsonify({
        'updated': datetime.now().isoformat(),
        'total_fragments': len(stream_manager.fragments_list),
        'max_fragments': MAX_FRAGMENTS,
        'fragment_duration': FRAGMENT_DURATION,
        'latest': latest,
        'current_recording': stream_manager.current_filename,
        'fragments': stream_manager.fragments_list
    })


@app.route('/fragments/<filename>', methods=['DELETE'])
def delete_fragment(filename):
    """Delete a specific fragment."""
    filepath = os.path.join(CACHE_DIR, filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Fragment not found'}), 404
    
    try:
        os.remove(filepath)
        
        # Remove from fragments list
        stream_manager.fragments_list = [
            f for f in stream_manager.fragments_list 
            if f['filename'] != filename
        ]
        stream_manager._save_fragments_metadata()
        
        return jsonify({
            'status': 'deleted',
            'filename': filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/debug/stats', methods=['GET'])
def debug_stats():
    """Get current streaming statistics."""
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.wav')]
    cache_size = sum(
        os.path.getsize(os.path.join(CACHE_DIR, f)) 
        for f in cache_files 
        if os.path.isfile(os.path.join(CACHE_DIR, f))
    )
    
    queue_stats = audio_queue.get_stats()
    
    return jsonify({
        'chunks_received': stats.chunks_received,
        'bytes_received': stats.bytes_received,
        'uptime_seconds': time.time() - stats.start_time,
        'rate_kbps': (stats.bytes_received * 8 / 1024) / (time.time() - stats.start_time) if time.time() > stats.start_time else 0,
        'last_chunk_ago': time.time() - stats.last_chunk_time if stats.last_chunk_time > 0 else None,
        'cache_files': len(cache_files),
        'cache_size_bytes': cache_size,
        'max_fragments': MAX_FRAGMENTS,
        'fragment_duration': FRAGMENT_DURATION,
        'session_id': stream_manager.session_id,
        'fragment_count': stream_manager.fragment_count,
        'audio_settings': {
            'sample_rate': stream_manager.sample_rate,
            'channels': stream_manager.channels,
            'sample_width': stream_manager.sample_width
        },
        'queue': queue_stats
    })


@app.route('/audio/buffer', methods=['GET'])
def get_audio_buffer():
    """Get current audio buffer info."""
    return jsonify({
        'buffer_samples': len(audio_buffer.buffer),
        'buffer_seconds': len(audio_buffer.buffer) / (audio_buffer.sample_rate * audio_buffer.channels) if len(audio_buffer.buffer) > 0 else 0,
        'sample_rate': audio_buffer.sample_rate,
        'channels': audio_buffer.channels,
        'last_update': audio_buffer.last_update
    })


# For programmatic access to the audio buffer
class AudioBufferReader:
    """Helper class for reading audio data programmatically."""
    
    def __init__(self):
        self.buffer = audio_buffer
    
    def read(self, seconds=None):
        """Read audio from the buffer."""
        return self.buffer.get_audio(seconds)
    
    def get_sample_rate(self):
        return self.buffer.sample_rate
    
    def get_channels(self):
        return self.buffer.channels
    
    def is_available(self):
        return len(self.buffer.buffer) > 0


# ============================================================================
# ASR Endpoints (Streaming Speech Recognition)
# ============================================================================

@app.route('/asr/status', methods=['GET'])
def asr_status():
    """Get streaming ASR status."""
    if not STREAMING_ASR_AVAILABLE:
        return jsonify({
            'available': False,
            'error': 'Streaming ASR module not available'
        })
    
    try:
        processor = streaming_asr.get_stream_processor()
        return jsonify({
            'available': True,
            'is_running': processor.is_running,
            'stream_start_time': processor.stream_start_time,
            'segments_count': len(processor.asr.all_segments) if processor.asr else 0,
            'sample_rate': processor.sample_rate,
            'target_sample_rate': processor.target_sample_rate
        })
    except Exception as e:
        return jsonify({
            'available': True,
            'error': str(e)
        }), 500


@app.route('/asr/transcription', methods=['GET'])
def asr_transcription():
    """Get current ASR transcription with timestamps."""
    if not STREAMING_ASR_AVAILABLE:
        return jsonify({'error': 'Streaming ASR not available'}), 503
    
    try:
        processor = streaming_asr.get_stream_processor()
        segments = []
        
        if processor.asr and processor.asr.all_segments:
            for seg in processor.asr.all_segments:
                segments.append({
                    'text': seg.text,
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'confidence': seg.confidence,
                    'is_final': seg.is_final
                })
        
        # Build full text
        full_text = " ".join(s['text'] for s in segments)
        
        return jsonify({
            'segments': segments,
            'full_text': full_text,
            'segment_count': len(segments),
            'stream_start_time': processor.stream_start_time,
            'is_running': processor.is_running
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/asr/video_markers', methods=['GET'])
def asr_video_markers():
    """
    Get video cutting markers from ASR.
    
    These timestamps can be used to cut video clips at speech boundaries.
    """
    if not STREAMING_ASR_AVAILABLE:
        return jsonify({'error': 'Streaming ASR not available'}), 503
    
    try:
        markers = streaming_asr.get_current_video_markers()
        
        result = []
        for start, end, text in markers:
            result.append({
                'start_time': start,
                'end_time': end,
                'duration': end - start,
                'text': text
            })
        
        return jsonify({
            'markers': result,
            'count': len(result)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/asr/finalize', methods=['POST'])
def asr_finalize():
    """Finalize ASR stream and get complete results."""
    if not STREAMING_ASR_AVAILABLE:
        return jsonify({'error': 'Streaming ASR not available'}), 503
    
    try:
        result = streaming_asr.finalize_stream()
        
        return jsonify({
            'status': 'finalized',
            'full_text': result.full_text,
            'duration': result.duration,
            'timestamp': result.timestamp,
            'segments': [s.to_dict() for s in result.segments]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/asr/reset', methods=['POST'])
def asr_reset():
    """Reset ASR stream for new session."""
    if not STREAMING_ASR_AVAILABLE:
        return jsonify({'error': 'Streaming ASR not available'}), 503
    
    try:
        processor = streaming_asr.get_stream_processor()
        processor.start()
        
        return jsonify({
            'status': 'reset',
            'stream_start_time': processor.stream_start_time
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    asr_status_msg = "✓ Streaming ASR enabled" if STREAMING_ASR_AVAILABLE else "✗ Streaming ASR not available"
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║              Audio Stream Receiver Server                            ║
╠══════════════════════════════════════════════════════════════════════╣
║  Listening on: http://{HOST}:{PORT}                                      ║
║  Cache directory: {CACHE_DIR}
║  Fragment duration: {FRAGMENT_DURATION} seconds                                         ║
║  Max fragments: {MAX_FRAGMENTS}                                                  ║
║  {asr_status_msg}
╠══════════════════════════════════════════════════════════════════════╣
║  Stream Endpoints:                                                   ║
║    GET  /health           - Health check                             ║
║    POST /stream/start     - Start streaming session                  ║
║    POST /stream/stop      - Stop streaming session                   ║
║    POST /audio/chunk      - Send raw PCM audio chunk                 ║
║    POST /audio/wav        - Send WAV-formatted audio chunk           ║
║    GET  /fragments        - List saved fragments                     ║
║    GET  /fragments/latest - Get latest fragment info                 ║
║    DELETE /fragments/<n>  - Delete a fragment                        ║
║    GET  /debug/stats      - Get streaming statistics                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  ASR Endpoints (Speech Recognition):                                 ║
║    GET  /asr/status        - ASR status                              ║
║    GET  /asr/transcription - Current transcription with timestamps   ║
║    GET  /asr/video_markers - Video cutting markers                   ║
║    POST /asr/finalize      - Finalize and get complete results       ║
║    POST /asr/reset         - Reset ASR for new session               ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Start the audio processing queue
    audio_queue.start(stream_manager, audio_buffer)
    print(f"✓ Audio processing queue started (buffer: {AUDIO_QUEUE_SIZE} chunks)")
    
    print("\nTo use programmatically:")
    print("  from audioreciever import AudioBufferReader")
    print("  reader = AudioBufferReader()")
    print("  audio = reader.read(seconds=5)")
    print()
    
    app.run(host=HOST, port=PORT, threaded=True)
