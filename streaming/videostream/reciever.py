#!/usr/bin/env python3
"""
Video Stream Receiver Server

Receives streaming video from an Android app via HTTP POST requests
and saves the stream into 10-second fragments in the cache folder.
Also exposes the stream as a virtual camera for OpenCV.
"""

import os
import io
import time
import threading
import subprocess
import queue
from datetime import datetime
from flask import Flask, request, jsonify

import cv2
import numpy as np

# Try to import pyfakewebcam for virtual camera support
try:
    import pyfakewebcam
    VIRTUAL_CAM_AVAILABLE = True
except ImportError:
    VIRTUAL_CAM_AVAILABLE = False
    print("Warning: pyfakewebcam not available. Virtual camera disabled.")

# Configuration
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 5000
FRAGMENT_DURATION = 10  # seconds
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
DEBUG = True  # Enable debug logging
MAX_FRAGMENTS = 50  # Maximum number of fragment files to keep
METADATA_FILE = os.path.join(CACHE_DIR, 'fragments.json')  # JSON file tracking fragments
FRAME_QUEUE_SIZE = 300  # Maximum frames to buffer (about 10 seconds at 30fps)

# Stats tracking
class Stats:
    def __init__(self):
        self.frames_received = 0
        self.bytes_received = 0
        self.last_frame_time = 0
        self.start_time = time.time()
        self.frame_intervals = []  # Track time between frames
        self.last_interval_report = 0
    
    def log_frame(self, size):
        current_time = time.time()
        
        # Track frame interval
        if self.last_frame_time > 0:
            interval = (current_time - self.last_frame_time) * 1000  # in ms
            self.frame_intervals.append(interval)
            # Keep only last 100 intervals
            if len(self.frame_intervals) > 100:
                self.frame_intervals.pop(0)
        
        self.frames_received += 1
        self.bytes_received += size
        self.last_frame_time = current_time
        
        if DEBUG and self.frames_received % 30 == 0:  # Log every 30 frames
            elapsed = current_time - self.start_time
            fps = self.frames_received / elapsed if elapsed > 0 else 0
            
            # Calculate frame timing stats
            if self.frame_intervals:
                avg_interval = sum(self.frame_intervals) / len(self.frame_intervals)
                min_interval = min(self.frame_intervals)
                max_interval = max(self.frame_intervals)
                jitter = max_interval - min_interval
                print(f"[STATS] Frames: {self.frames_received}, FPS: {fps:.1f}, "
                      f"Interval: {avg_interval:.1f}ms (min:{min_interval:.1f}, max:{max_interval:.1f}, jitter:{jitter:.1f}ms)")
            else:
                print(f"[STATS] Frames: {self.frames_received}, Bytes: {self.bytes_received/1024:.1f}KB, FPS: {fps:.1f}")

stats = Stats()

# Virtual camera configuration
VIRTUAL_CAM_DEVICE = '/dev/video20'  # v4l2loopback device
VIRTUAL_CAM_WIDTH = 640
VIRTUAL_CAM_HEIGHT = 480
VIRTUAL_CAM_FPS = 30

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

app = Flask(__name__)

# Shared frame buffer for virtual camera
class FrameBuffer:
    """Thread-safe frame buffer for sharing frames between receiver and virtual camera."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.frame_available = threading.Event()
        self.last_update = 0
    
    def update_frame(self, frame):
        """Update the current frame."""
        with self.lock:
            self.frame = frame.copy()
            self.last_update = time.time()
            self.frame_available.set()
    
    def get_frame(self):
        """Get the current frame."""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
    
    def wait_for_frame(self, timeout=1.0):
        """Wait for a new frame to be available."""
        return self.frame_available.wait(timeout)
    
    def clear_event(self):
        """Clear the frame available event."""
        self.frame_available.clear()


frame_buffer = FrameBuffer()


# Frame Processing Queue - handles burst traffic by buffering frames
class FrameProcessingQueue:
    """
    Async frame processing queue to handle burst traffic.
    Frames are quickly accepted and queued, then processed in background.
    """
    
    def __init__(self, max_size=FRAME_QUEUE_SIZE):
        self.queue = queue.Queue(maxsize=max_size)
        self.running = False
        self.thread = None
        self.frames_queued = 0
        self.frames_processed = 0
        self.frames_dropped = 0
    
    def start(self, stream_manager, frame_buffer_ref):
        """Start the background processing thread."""
        self.stream_manager = stream_manager
        self.frame_buffer = frame_buffer_ref
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        print("[INFO] Frame processing queue started")
    
    def stop(self):
        """Stop the processing thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def add_frame(self, frame):
        """
        Add a frame to the processing queue.
        Returns immediately, frame is processed in background.
        """
        try:
            self.queue.put_nowait(frame)
            self.frames_queued += 1
            return True
        except queue.Full:
            # Queue is full, drop the oldest frame and add new one
            try:
                self.queue.get_nowait()
                self.frames_dropped += 1
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(frame)
                self.frames_queued += 1
                return True
            except queue.Full:
                self.frames_dropped += 1
                return False
    
    def _process_loop(self):
        """Background thread that processes queued frames."""
        while self.running:
            try:
                # Get frame with timeout to allow clean shutdown
                frame = self.queue.get(timeout=0.5)
                
                # Update frame buffer for virtual camera
                self.frame_buffer.update_frame(frame)
                
                # Save frame to video file
                self.stream_manager.save_frame(frame)
                
                self.frames_processed += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                if DEBUG:
                    print(f"[ERROR] Frame processing error: {e}")
    
    def get_stats(self):
        """Get queue statistics."""
        return {
            'queued': self.frames_queued,
            'processed': self.frames_processed,
            'dropped': self.frames_dropped,
            'pending': self.queue.qsize()
        }


frame_queue = FrameProcessingQueue()


# Virtual Camera Manager
class VirtualCameraManager:
    """Manages the virtual camera device using v4l2loopback."""
    
    def __init__(self, device=VIRTUAL_CAM_DEVICE, width=VIRTUAL_CAM_WIDTH, 
                 height=VIRTUAL_CAM_HEIGHT, fps=VIRTUAL_CAM_FPS):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = None
        self.running = False
        self.thread = None
    
    def is_device_available(self):
        """Check if the v4l2loopback device exists."""
        return os.path.exists(self.device)
    
    def setup_v4l2loopback(self):
        """Instructions for setting up v4l2loopback."""
        return """
        To set up v4l2loopback virtual camera:
        
        1. Install v4l2loopback:
           sudo apt-get install v4l2loopback-dkms v4l2loopback-utils
        
        2. Load the kernel module:
           sudo modprobe v4l2loopback devices=1 video_nr=20 card_label="AndroidStream" exclusive_caps=1
        
        3. To make it persistent, add to /etc/modules-load.d/v4l2loopback.conf:
           v4l2loopback
        
        4. And add options to /etc/modprobe.d/v4l2loopback.conf:
           options v4l2loopback devices=1 video_nr=20 card_label="AndroidStream" exclusive_caps=1
        """
    
    def start(self):
        """Start the virtual camera."""
        if not VIRTUAL_CAM_AVAILABLE:
            print("Virtual camera not available: pyfakewebcam not installed")
            return False
        
        if not self.is_device_available():
            print(f"Virtual camera device {self.device} not found.")
            print(self.setup_v4l2loopback())
            return False
        
        try:
            self.camera = pyfakewebcam.FakeWebcam(self.device, self.width, self.height)
            self.running = True
            self.thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.thread.start()
            print(f"Virtual camera started on {self.device}")
            return True
        except Exception as e:
            print(f"Failed to start virtual camera: {e}")
            return False
    
    def _stream_loop(self):
        """Main loop for streaming frames to virtual camera."""
        frame_interval = 1.0 / self.fps
        
        # Create a blank frame as placeholder
        blank_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "Waiting for stream...", (50, self.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        while self.running:
            start_time = time.time()
            
            frame = frame_buffer.get_frame()
            if frame is not None:
                # Resize frame to match virtual camera dimensions
                if frame.shape[:2] != (self.height, self.width):
                    frame = cv2.resize(frame, (self.width, self.height))
                # Convert BGR to RGB for pyfakewebcam
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = cv2.cvtColor(blank_frame, cv2.COLOR_BGR2RGB)
            
            try:
                self.camera.schedule_frame(frame_rgb)
            except Exception as e:
                print(f"Error sending frame to virtual camera: {e}")
            
            # Maintain frame rate
            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    
    def stop(self):
        """Stop the virtual camera."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.camera = None
        print("Virtual camera stopped")


virtual_camera = VirtualCameraManager()


# Frame decoder for extracting frames from video stream
class FrameDecoder:
    """Decodes video frames from incoming stream data."""
    
    def __init__(self):
        self.buffer = bytearray()
        self.lock = threading.Lock()
    
    def feed_data(self, data):
        """Feed video data and try to extract frames."""
        with self.lock:
            self.buffer.extend(data)
            self._try_decode_frames()
    
    def _try_decode_frames(self):
        """Try to decode frames from the buffer."""
        if len(self.buffer) < 1000:
            return
        
        try:
            # Try to decode as JPEG frames (common for Android camera streams)
            # Look for JPEG markers
            jpeg_start = self.buffer.find(b'\xff\xd8')  # JPEG SOI marker
            if jpeg_start >= 0:
                jpeg_end = self.buffer.find(b'\xff\xd9', jpeg_start)  # JPEG EOI marker
                if jpeg_end >= 0:
                    jpeg_data = bytes(self.buffer[jpeg_start:jpeg_end + 2])
                    self.buffer = self.buffer[jpeg_end + 2:]
                    
                    # Decode JPEG to frame
                    nparr = np.frombuffer(jpeg_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        frame_buffer.update_frame(frame)
                        return
            
            # Limit buffer size to prevent memory issues
            if len(self.buffer) > 10 * 1024 * 1024:  # 10MB limit
                self.buffer = self.buffer[-1024 * 1024:]  # Keep last 1MB
                
        except Exception as e:
            print(f"Frame decode error: {e}")


# Stream state management
class StreamManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.video_writer = None
        self.current_filepath = None
        self.current_filename = None
        self.fragment_start_time = None
        self.fragment_count = 0
        self.session_id = None
        self.frame_decoder = FrameDecoder()
        self.frames_in_fragment = 0
        self.frame_width = None
        self.frame_height = None
        self.target_fps = 15  # Target FPS for video files
        self.fragments_list = []  # List of fragment info dicts
        self._load_metadata()
    
    def _load_metadata(self):
        """Load existing metadata from JSON file."""
        try:
            if os.path.exists(METADATA_FILE):
                import json
                with open(METADATA_FILE, 'r') as f:
                    data = json.load(f)
                    self.fragments_list = data.get('fragments', [])
                    # Verify files still exist
                    self.fragments_list = [
                        f for f in self.fragments_list 
                        if os.path.exists(os.path.join(CACHE_DIR, f['filename']))
                    ]
                    if DEBUG:
                        print(f"[DEBUG] Loaded {len(self.fragments_list)} existing fragments from metadata")
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] Could not load metadata: {e}")
            self.fragments_list = []
    
    def _save_metadata(self):
        """Save current metadata to JSON file."""
        import json
        try:
            # Determine latest fragment
            latest = self.fragments_list[-1] if self.fragments_list else None
            
            metadata = {
                'updated': datetime.now().isoformat(),
                'total_fragments': len(self.fragments_list),
                'max_fragments': MAX_FRAGMENTS,
                'latest': latest,
                'fragments': self.fragments_list
            }
            
            with open(METADATA_FILE, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            if DEBUG:
                print(f"[DEBUG] Metadata saved: {len(self.fragments_list)} fragments, latest: {latest['filename'] if latest else 'none'}")
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] Could not save metadata: {e}")
    
    def _cleanup_old_fragments(self):
        """Remove oldest fragments if we exceed MAX_FRAGMENTS."""
        while len(self.fragments_list) >= MAX_FRAGMENTS:
            oldest = self.fragments_list.pop(0)
            oldest_path = os.path.join(CACHE_DIR, oldest['filename'])
            try:
                if os.path.exists(oldest_path):
                    os.remove(oldest_path)
                    if DEBUG:
                        print(f"[DEBUG] Deleted old fragment: {oldest['filename']}")
            except Exception as e:
                if DEBUG:
                    print(f"[DEBUG] Could not delete {oldest['filename']}: {e}")
    
    def _add_fragment_to_list(self, filename, frames, duration):
        """Add a completed fragment to the tracking list."""
        filepath = os.path.join(CACHE_DIR, filename)
        fragment_info = {
            'filename': filename,
            'filepath': filepath,
            'frames': frames,
            'duration': round(duration, 2),
            'size': os.path.getsize(filepath) if os.path.exists(filepath) else 0,
            'created': datetime.now().isoformat(),
            'timestamp': time.time()
        }
        self.fragments_list.append(fragment_info)
        self._save_metadata()
    
    def get_fragment_filename(self):
        """Generate a filename for the current fragment."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"fragment_{timestamp}_{self.fragment_count:04d}.avi"
    
    def start_new_fragment(self, width=None, height=None):
        """Start a new video fragment file."""
        # Close current video writer if exists
        if self.video_writer is not None:
            self.video_writer.release()
            duration = time.time() - self.fragment_start_time
            if DEBUG:
                print(f"[DEBUG] Video fragment saved: {self.current_filepath} ({self.frames_in_fragment} frames, {duration:.1f}s)")
            # Add to tracking list
            self._add_fragment_to_list(self.current_filename, self.frames_in_fragment, duration)
        
        # Cleanup old fragments before creating new one
        self._cleanup_old_fragments()
        
        # Use provided dimensions or defaults
        if width:
            self.frame_width = width
        if height:
            self.frame_height = height
        
        if not self.frame_width or not self.frame_height:
            self.frame_width = 640
            self.frame_height = 480
        
        # Create new fragment
        self.current_filename = self.get_fragment_filename()
        self.current_filepath = os.path.join(CACHE_DIR, self.current_filename)
        
        # Use XVID codec for AVI files (widely compatible)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(
            self.current_filepath, 
            fourcc, 
            self.target_fps, 
            (self.frame_width, self.frame_height)
        )
        
        self.fragment_start_time = time.time()
        self.fragment_count += 1
        self.frames_in_fragment = 0
        
        if DEBUG:
            print(f"[DEBUG] Started new video fragment: {self.current_filename} ({self.frame_width}x{self.frame_height} @ {self.target_fps}fps)")
    
    def write_frame(self, frame):
        """Write a decoded frame to the current video fragment."""
        with self.lock:
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            # Start new fragment if needed
            if self.video_writer is None:
                self.start_new_fragment(width, height)
            
            # Check if dimensions changed
            if width != self.frame_width or height != self.frame_height:
                if DEBUG:
                    print(f"[DEBUG] Frame size changed from {self.frame_width}x{self.frame_height} to {width}x{height}")
                # Resize frame to match current video writer
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            
            # Check if we need to rotate to a new fragment (10 seconds)
            elapsed = time.time() - self.fragment_start_time
            if elapsed >= FRAGMENT_DURATION:
                self.start_new_fragment(width, height)
            
            # Write frame to video
            self.video_writer.write(frame)
            self.frames_in_fragment += 1
            
            if DEBUG and self.frames_in_fragment % 30 == 0:
                print(f"[DEBUG] Fragment progress: {self.frames_in_fragment} frames, {elapsed:.1f}s elapsed")
    
    def write_data(self, data):
        """Write raw data - tries to decode as JPEG and write as video frame."""
        # Try to decode as JPEG frame
        try:
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                self.write_frame(frame)
                return True
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] Could not decode data as frame: {e}")
        
        # Feed to frame decoder for MJPEG streams
        self.frame_decoder.feed_data(data)
        return False
    
    def save_frame(self, frame):
        """Save a decoded OpenCV frame to the video fragment."""
        self.write_frame(frame)
    
    def close(self):
        """Close the current video fragment."""
        with self.lock:
            if self.video_writer is not None:
                self.video_writer.release()
                duration = time.time() - self.fragment_start_time if self.fragment_start_time else 0
                print(f"[DEBUG] Final video fragment saved: {self.current_filepath} ({self.frames_in_fragment} frames)")
                # Add to tracking list
                if self.current_filename:
                    self._add_fragment_to_list(self.current_filename, self.frames_in_fragment, duration)
                self.video_writer = None
                self.current_filepath = None
                self.current_filename = None

    def start_session(self, session_id):
        """Start a new streaming session."""
        with self.lock:
            self.session_id = session_id
            self.fragment_count = 0
            print(f"Started session: {session_id}")
    
    def end_session(self):
        """End the current streaming session."""
        self.close()
        with self.lock:
            print(f"Ended session: {self.session_id}")
            self.session_id = None


stream_manager = StreamManager()


@app.route('/', methods=['GET'])
def index():
    """Root endpoint - returns basic server info."""
    return jsonify({
        'service': 'Video Stream Receiver',
        'status': 'running',
        'version': '1.0',
        'endpoints': {
            'health': '/health',
            'stream_start': '/stream/start',
            'stream_stop': '/stream/stop',
            'camera_frame': '/camera/frame',
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
        'frames_received': stats.frames_received,
        'queue_pending': frame_queue.queue.qsize() if frame_queue else 0
    })


@app.route('/stream/start', methods=['POST'])
def start_stream():
    """
    Start a new streaming session.
    
    Optional JSON body:
    {
        "session_id": "unique_session_identifier"
    }
    """
    if DEBUG:
        print(f"[DEBUG] /stream/start called from {request.remote_addr}")
    
    # Handle both JSON and non-JSON requests
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        data = request.get_json(force=True, silent=True)
        if data and 'session_id' in data:
            session_id = data['session_id']
    except:
        pass
    
    stream_manager.start_session(session_id)
    
    if DEBUG:
        print(f"[DEBUG] Stream session started: {session_id}")
    
    return jsonify({
        'status': 'started',
        'session_id': session_id,
        'fragment_duration': FRAGMENT_DURATION
    })


@app.route('/stream/data', methods=['POST'])
def receive_stream_data():
    """
    Receive video stream data chunk.
    
    The video data should be sent as raw bytes in the request body.
    Content-Type should be 'application/octet-stream' or 'video/mp4'.
    """
    if request.content_length == 0:
        return jsonify({'error': 'No data received'}), 400
    
    # Read raw video data from request body
    video_data = request.get_data()
    
    if video_data:
        stream_manager.write_data(video_data)
        return jsonify({
            'status': 'received',
            'bytes': len(video_data)
        })
    else:
        return jsonify({'error': 'No data in request'}), 400


@app.route('/stream/chunk', methods=['POST'])
def receive_stream_chunk():
    """
    Alternative endpoint for chunked streaming.
    Handles multipart form data with video/image chunks.
    Accepts 'video', 'chunk', or 'frame' as the form field name.
    """
    if DEBUG:
        print(f"[DEBUG] /stream/chunk called, content_length: {request.content_length}, files: {list(request.files.keys())}")
    
    # Check for various possible field names
    file_data = None
    field_found = None
    for field_name in ['video', 'chunk', 'frame', 'image']:
        if field_name in request.files:
            file_data = request.files[field_name].read()
            field_found = field_name
            break
    
    if file_data:
        if DEBUG and stats.frames_received % 30 == 0:
            print(f"[DEBUG] Received chunk via '{field_found}': {len(file_data)} bytes")
        
        # Try to decode as JPEG frame and save to video
        try:
            nparr = np.frombuffer(file_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                # Update frame buffer for virtual camera
                frame_buffer.update_frame(frame)
                # Save frame to video file
                stream_manager.save_frame(frame)
                stats.log_frame(len(file_data))
                
                if DEBUG and stats.frames_received % 30 == 0:
                    print(f"[DEBUG] Frame saved to video: {frame.shape[1]}x{frame.shape[0]}")
            else:
                if DEBUG:
                    print(f"[DEBUG] Could not decode chunk as image")
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] Frame decode error: {e}")
        
        return jsonify({
            'status': 'received',
            'bytes': len(file_data),
            'fragment': stream_manager.fragment_count
        })
    elif request.data:
        if DEBUG:
            print(f"[DEBUG] Received raw data: {len(request.data)} bytes")
        # Try to decode raw data as frame
        try:
            nparr = np.frombuffer(request.data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                frame_buffer.update_frame(frame)
                stream_manager.save_frame(frame)
                stats.log_frame(len(request.data))
        except:
            pass
        return jsonify({
            'status': 'received',
            'bytes': len(request.data)
        })
    else:
        if DEBUG:
            print(f"[DEBUG] No data found in request")
        return jsonify({'error': 'No video data found'}), 400


@app.route('/stream/stop', methods=['POST'])
def stop_stream():
    """Stop the current streaming session and finalize all fragments."""
    stream_manager.end_session()
    
    return jsonify({
        'status': 'stopped',
        'total_fragments': stream_manager.fragment_count
    })


@app.route('/fragments', methods=['GET'])
def list_fragments():
    """List all saved video fragments."""
    fragments = []
    for filename in sorted(os.listdir(CACHE_DIR)):
        if filename.endswith(('.mp4', '.webm', '.mkv', '.avi')):
            filepath = os.path.join(CACHE_DIR, filename)
            fragments.append({
                'filename': filename,
                'size': os.path.getsize(filepath),
                'created': datetime.fromtimestamp(
                    os.path.getctime(filepath)
                ).isoformat()
            })
    
    return jsonify({
        'fragments': fragments,
        'total': len(fragments)
    })


@app.route('/fragments/<filename>', methods=['DELETE'])
def delete_fragment(filename):
    """Delete a specific video fragment."""
    filepath = os.path.join(CACHE_DIR, filename)
    
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify({'status': 'deleted', 'filename': filename})
    else:
        return jsonify({'error': 'Fragment not found'}), 404


# Virtual Camera endpoints
@app.route('/camera/start', methods=['POST'])
def start_virtual_camera():
    """Start the virtual camera."""
    success = virtual_camera.start()
    if success:
        return jsonify({
            'status': 'started',
            'device': virtual_camera.device,
            'resolution': f"{virtual_camera.width}x{virtual_camera.height}",
            'fps': virtual_camera.fps
        })
    else:
        return jsonify({
            'status': 'failed',
            'error': 'Could not start virtual camera',
            'setup_instructions': virtual_camera.setup_v4l2loopback()
        }), 500


@app.route('/camera/stop', methods=['POST'])
def stop_virtual_camera():
    """Stop the virtual camera."""
    virtual_camera.stop()
    return jsonify({'status': 'stopped'})


@app.route('/camera/status', methods=['GET'])
def virtual_camera_status():
    """Get virtual camera status."""
    return jsonify({
        'running': virtual_camera.running,
        'device': virtual_camera.device,
        'device_available': virtual_camera.is_device_available(),
        'resolution': f"{virtual_camera.width}x{virtual_camera.height}",
        'fps': virtual_camera.fps,
        'pyfakewebcam_available': VIRTUAL_CAM_AVAILABLE
    })


@app.route('/camera/frame', methods=['POST'])
def receive_frame():
    """
    Receive a single JPEG frame directly.
    This is an alternative to streaming video data.
    The Android app can send individual JPEG frames.
    Uses async queue for fast response under burst traffic.
    """
    # Check if there's an active session
    if not stream_manager.session_id:
        # No active session - accept frame anyway but warn
        if DEBUG:
            print(f"[WARN] Frame received without active session from {request.remote_addr}")
        # Still process it to avoid data loss
    
    if request.content_length == 0:
        if DEBUG:
            print(f"[DEBUG] /camera/frame called with no data")
        return jsonify({'error': 'No data received'}), 400
    
    frame_data = request.get_data()
    
    if DEBUG and stats.frames_received % 30 == 0:
        print(f"[DEBUG] /camera/frame received {len(frame_data)} bytes from {request.remote_addr}")
    
    try:
        # Decode JPEG frame
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            stats.log_frame(len(frame_data))
            
            # Add to async processing queue (fast return)
            queued = frame_queue.add_frame(frame)
            
            if DEBUG and stats.frames_received % 30 == 0:
                queue_stats = frame_queue.get_stats()
                print(f"[DEBUG] Frame queued: {frame.shape[1]}x{frame.shape[0]}, "
                      f"queue: {queue_stats['pending']}, dropped: {queue_stats['dropped']}")
            
            return jsonify({
                'status': 'received',
                'width': frame.shape[1],
                'height': frame.shape[0],
                'frame_count': stats.frames_received,
                'fragment': stream_manager.fragment_count,
                'queued': queued
            })
        else:
            if DEBUG:
                print(f"[DEBUG] Failed to decode JPEG frame")
            return jsonify({'error': 'Failed to decode frame'}), 400
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] Frame error: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/camera/setup', methods=['GET'])
def virtual_camera_setup():
    """Get setup instructions for virtual camera."""
    return jsonify({
        'instructions': virtual_camera.setup_v4l2loopback(),
        'device': virtual_camera.device
    })


@app.route('/debug/stats', methods=['GET'])
def debug_stats():
    """Get current streaming statistics."""
    import os
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.avi')]
    cache_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in cache_files if os.path.isfile(os.path.join(CACHE_DIR, f)))
    
    queue_stats = frame_queue.get_stats()
    
    return jsonify({
        'frames_received': stats.frames_received,
        'bytes_received': stats.bytes_received,
        'uptime_seconds': time.time() - stats.start_time,
        'fps': stats.frames_received / (time.time() - stats.start_time) if time.time() > stats.start_time else 0,
        'last_frame_ago': time.time() - stats.last_frame_time if stats.last_frame_time > 0 else None,
        'cache_files': len(cache_files),
        'cache_size_bytes': cache_size,
        'max_fragments': MAX_FRAGMENTS,
        'session_id': stream_manager.session_id,
        'fragment_count': stream_manager.fragment_count,
        'queue': queue_stats
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
    """Get the full fragments metadata (same as fragments.json)."""
    latest = stream_manager.fragments_list[-1] if stream_manager.fragments_list else None
    return jsonify({
        'updated': datetime.now().isoformat(),
        'total_fragments': len(stream_manager.fragments_list),
        'max_fragments': MAX_FRAGMENTS,
        'latest': latest,
        'current_recording': stream_manager.current_filename,
        'fragments': stream_manager.fragments_list
    })


# OpenCV helper function for external scripts
def get_opencv_capture():
    """
    Returns an OpenCV-compatible capture object.
    
    Usage from external script:
        from reciever import get_opencv_capture
        cap = get_opencv_capture()
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Frame', frame)
    """
    if virtual_camera.is_device_available():
        return cv2.VideoCapture(VIRTUAL_CAM_DEVICE)
    else:
        print(f"Virtual camera device {VIRTUAL_CAM_DEVICE} not available")
        print(virtual_camera.setup_v4l2loopback())
        return None


class FrameBufferCapture:
    """
    OpenCV-compatible capture class that reads from the frame buffer.
    Use this when v4l2loopback is not available.
    
    Usage:
        cap = FrameBufferCapture()
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Frame', frame)
    """
    
    def __init__(self):
        self.opened = True
    
    def isOpened(self):
        return self.opened
    
    def read(self):
        frame = frame_buffer.get_frame()
        if frame is not None:
            return True, frame
        return False, None
    
    def release(self):
        self.opened = False
    
    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return VIRTUAL_CAM_WIDTH
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return VIRTUAL_CAM_HEIGHT
        elif prop == cv2.CAP_PROP_FPS:
            return VIRTUAL_CAM_FPS
        return 0


if __name__ == '__main__':
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║              Video Stream Receiver Server                            ║
╠══════════════════════════════════════════════════════════════════════╣
║  Listening on: http://{HOST}:{PORT}                                      ║
║  Cache directory: {CACHE_DIR}
║  Fragment duration: {FRAGMENT_DURATION} seconds                                        ║
║  Virtual camera: {VIRTUAL_CAM_DEVICE}                                    ║
╠══════════════════════════════════════════════════════════════════════╣
║  Stream Endpoints:                                                   ║
║    GET  /health           - Health check                             ║
║    POST /stream/start     - Start streaming session                  ║
║    POST /stream/data      - Send video data (raw bytes)              ║
║    POST /stream/chunk     - Send video chunk (multipart)             ║
║    POST /stream/stop      - Stop streaming session                   ║
║    GET  /fragments        - List saved fragments                     ║
║    DELETE /fragments/<n>  - Delete a fragment                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  Virtual Camera Endpoints:                                           ║
║    POST /camera/start     - Start virtual camera                     ║
║    POST /camera/stop      - Stop virtual camera                      ║
║    GET  /camera/status    - Get virtual camera status                ║
║    POST /camera/frame     - Send a single JPEG frame                 ║
║    GET  /camera/setup     - Get v4l2loopback setup instructions      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check virtual camera availability
    if VIRTUAL_CAM_AVAILABLE:
        if virtual_camera.is_device_available():
            print(f"✓ Virtual camera device {VIRTUAL_CAM_DEVICE} is available")
        else:
            print(f"✗ Virtual camera device {VIRTUAL_CAM_DEVICE} not found")
            print("  Run 'GET /camera/setup' for setup instructions")
    else:
        print("✗ pyfakewebcam not installed. Virtual camera disabled.")
    
    # Start the frame processing queue
    frame_queue.start(stream_manager, frame_buffer)
    print(f"✓ Frame processing queue started (buffer: {FRAME_QUEUE_SIZE} frames)")
    
    print("\nTo use with OpenCV:")
    print(f"  cap = cv2.VideoCapture('{VIRTUAL_CAM_DEVICE}')")
    print("  OR")
    print("  from reciever import FrameBufferCapture")
    print("  cap = FrameBufferCapture()")
    print()
    
    app.run(host=HOST, port=PORT, threaded=True)
