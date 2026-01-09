# Video Stream System Documentation

A complete video streaming and analysis system that receives video from an Android app, saves fragments, performs object detection with YOLO, and facial recognition with CompreFace.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Components](#components)
3. [Receiver Server (reciever.py)](#receiver-server-recieverpy)
4. [Object Detection (opencv.py)](#object-detection-opencvpy)
5. [Facial Recognition (facial_recognition.py)](#facial-recognition-facial_recognitionpy)
6. [JSON File Formats](#json-file-formats)
7. [Configuration](#configuration)
8. [Usage Examples](#usage-examples)

---

## System Overview

```
┌─────────────────┐     HTTP POST      ┌──────────────────┐
│  Android App    │ ─────────────────► │  Receiver Server │
│  (Camera)       │   /camera/frame    │  (reciever.py)   │
└─────────────────┘                    └────────┬─────────┘
                                                │
                                                ▼
                                       ┌────────────────┐
                                       │  cache/ folder │
                                       │  (AVI files)   │
                                       └───────┬────────┘
                                               │
                        ┌──────────────────────┼──────────────────────┐
                        ▼                      ▼                      ▼
               ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
               │  opencv.py     │     │ facial_recog.  │     │ Virtual Camera │
               │  (YOLO)        │     │ (CompreFace)   │     │ (/dev/video20) │
               └────────────────┘     └────────────────┘     └────────────────┘
```

---

## Components

| File | Purpose |
|------|---------|
| `reciever.py` | Flask server receiving video stream from Android |
| `opencv.py` | YOLO-based object detection on video fragments |
| `facial_recognition.py` | CompreFace-based facial recognition |
| `cache/` | Directory storing video fragments (AVI files) |
| `opencv_cache/` | Directory storing processed videos with detection overlays |
| `face_cache/` | Directory storing facial recognition results |

---

## Receiver Server (reciever.py)

### Configuration

```python
HOST = '0.0.0.0'          # Listen on all interfaces
PORT = 5000               # Server port
FRAGMENT_DURATION = 10    # Seconds per video fragment
MAX_FRAGMENTS = 50        # Maximum fragments to keep
FRAME_QUEUE_SIZE = 300    # Frame buffer size for burst traffic
VIRTUAL_CAM_DEVICE = '/dev/video20'  # Virtual camera device
```

### HTTP Endpoints

#### Root & Health

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server info and available endpoints |
| `/health` | GET | Health check with session status |

**Response `/health`:**
```json
{
  "status": "ok",
  "timestamp": "2026-01-08T20:00:00.000000",
  "active_session": "20260108_200000",
  "frames_received": 1500,
  "queue_pending": 5
}
```

#### Streaming

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stream/start` | POST | Start a new streaming session |
| `/stream/stop` | POST | Stop current session and finalize fragment |
| `/stream/data` | POST | Send raw video bytes (alternative) |
| `/stream/chunk` | POST | Send multipart video chunk (alternative) |

**Request `/stream/start`:**
```json
{
  "session_id": "optional_custom_id",
  "fps": 15,
  "width": 1920,
  "height": 1080
}
```

**Response `/stream/start`:**
```json
{
  "status": "started",
  "session_id": "20260108_200000",
  "message": "Stream session started"
}
```

#### Frame Upload

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/camera/frame` | POST | Send a single JPEG frame (primary method) |
| `/camera/start` | POST | Start virtual camera |
| `/camera/stop` | POST | Stop virtual camera |
| `/camera/status` | GET | Get virtual camera status |
| `/camera/setup` | GET | Get v4l2loopback setup instructions |

**Request `/camera/frame`:**
- Body: Raw JPEG bytes
- Content-Type: `image/jpeg` or `application/octet-stream`

**Response `/camera/frame`:**
```json
{
  "status": "received",
  "width": 1920,
  "height": 1080,
  "frame_count": 150,
  "fragment": 3,
  "queued": true
}
```

#### Fragment Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/fragments` | GET | List all saved fragments |
| `/fragments/latest` | GET | Get latest fragment info |
| `/fragments/metadata` | GET | Get full fragments.json data |
| `/fragments/<filename>` | DELETE | Delete a specific fragment |

**Response `/fragments`:**
```json
{
  "count": 10,
  "max_fragments": 50,
  "total_size_bytes": 52428800,
  "fragments": ["fragment_20260108_200000_0000.avi", "..."]
}
```

#### Debug

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/debug/stats` | GET | Get detailed streaming statistics |

**Response `/debug/stats`:**
```json
{
  "frames_received": 1500,
  "bytes_received": 150000000,
  "uptime_seconds": 3600,
  "fps": 15.5,
  "last_frame_ago": 0.5,
  "cache_files": 10,
  "cache_size_bytes": 52428800,
  "session_id": "20260108_200000",
  "fragment_count": 3,
  "queue": {
    "queued": 1500,
    "processed": 1495,
    "dropped": 0,
    "pending": 5
  }
}
```

### Classes

#### FrameBuffer
Thread-safe buffer for sharing frames between receiver and virtual camera.

```python
frame_buffer.update_frame(frame)  # Update current frame
frame_buffer.get_frame()          # Get current frame
frame_buffer.wait_for_frame(1.0)  # Wait for new frame
```

#### FrameProcessingQueue
Async queue for handling burst traffic.

```python
frame_queue.add_frame(frame)      # Add frame to queue
frame_queue.get_stats()           # Get queue statistics
```

#### StreamManager
Manages video recording sessions and fragment files.

```python
stream_manager.start_session(session_id, fps, width, height)
stream_manager.save_frame(frame)
stream_manager.end_session()
```

#### VirtualCameraManager
Manages v4l2loopback virtual camera for OpenCV access.

```python
virtual_camera.start()
virtual_camera.stop()
virtual_camera.is_device_available()
```

---

## Object Detection (opencv.py)

### Configuration

```python
CONFIDENCE_THRESHOLD = 0.5    # Minimum detection confidence
NMS_THRESHOLD = 0.4           # Non-maximum suppression threshold
NUM_THREADS = 32              # CPU threads for inference
```

### Supported Models

| Model | Flag | Description |
|-------|------|-------------|
| YOLO11n | `--model yolo11n` | Nano - fastest, least accurate |
| YOLO11s | `--model yolo11s` | Small |
| YOLO11m | `--model yolo11m` | Medium (default) |
| YOLO11l | `--model yolo11l` | Large |
| YOLO11x | `--model yolo11x` | Extra-large - most accurate |
| YOLOv4-tiny | `--model yolov4-tiny` | Legacy fast model |
| YOLOv4 | `--model yolov4` | Legacy accurate model |

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--file <path>` | Process specific video file | - |
| `--latest` | Process latest fragment | - |
| `--watch` | Continuously watch for new fragments | - |
| `--interval <sec>` | Watch interval | 15 |
| `--model <name>` | YOLO model to use | yolo11m |
| `--confidence <val>` | Detection confidence threshold | 0.5 |
| `--save-video` | Save processed video with overlays | False |

### Classes

#### ObjectDetector
YOLO-based object detection using Ultralytics.

```python
detector = ObjectDetector(model_name='yolo11m')
detections = detector.detect(frame)
# Returns: [{'class': 'person', 'confidence': 0.95, 'box': [x1,y1,x2,y2]}]
```

#### FragmentProcessor
Processes video fragments and saves results.

```python
processor = FragmentProcessor(detector)
results = processor.process_video(video_path)
```

---

## Facial Recognition (facial_recognition.py)

### Configuration

```python
COMPREFACE_DOMAIN = 'http://localhost'
COMPREFACE_PORT = '8000'
COMPREFACE_API_KEY = '185efb81-9b55-4f72-b38c-d8e2bd0d4a3b'
FRAME_SAMPLE_RATE = 5         # Process every Nth frame
FACE_DETECTION_THRESHOLD = 0.8
RECOGNITION_THRESHOLD = 0.85   # Similarity for recognizing known faces
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--file <path>` | Process specific video file | - |
| `--latest` | Process latest fragment | - |
| `--watch` | Continuously watch for new fragments | - |
| `--interval <sec>` | Watch interval | 15 |
| `--domain <url>` | CompreFace server domain | http://localhost |
| `--port <port>` | CompreFace server port | 8000 |
| `--api-key <key>` | CompreFace API key | - |
| `--sample-rate <n>` | Process every Nth frame | 5 |
| `--auto-enroll` | Auto-add new faces to CompreFace | False |
| `--clear` | Clear all subjects and JSON files | - |
| `--rename <id> <name>` | Rename a face (syncs to CompreFace) | - |
| `--list-faces` | List all known faces | - |
| `--no-quality-update` | Disable auto quality updates | False |

### Classes

#### FaceDatabase
Manages local face database in JSON files.

```python
db = FaceDatabase()
face_id = db.add_or_update_face(face_data)
db.rename_face(face_id, "John Doe")
db.finalize(current_faces)
```

#### ImageQualityAnalyzer
Calculates image quality scores (0-100).

```python
score = ImageQualityAnalyzer.calculate_quality_score(image)
is_better, new_score, old_score = ImageQualityAnalyzer.is_better_quality(new_img, old_path)
```

Quality factors:
- **Sharpness** (30 pts) - Laplacian variance
- **Brightness** (20 pts) - Optimal around 130
- **Contrast** (20 pts) - Standard deviation
- **Size** (30 pts) - Larger faces score higher

#### FaceManager
Manages face operations with CompreFace sync.

```python
manager = FaceManager(compreface_processor, face_database)
manager.rename_face(old_id, new_name)  # Syncs to CompreFace
manager.update_face_if_better(face_id, new_image)  # Quality-based update
faces = manager.list_faces()
```

#### CompreFaceProcessor
Interface to CompreFace API.

```python
processor = CompreFaceProcessor(domain, port, api_key)
faces = processor.detect_faces(image_path)
processor.add_subject(subject_name)
processor.add_face_to_subject(subject_name, image_path)
processor.rename_subject(old_name, new_name)
processor.delete_subject(subject_name)
processor.delete_all_subjects()
subjects = processor.get_all_subjects()
```

#### VideoFaceProcessor
Processes video fragments for facial recognition.

```python
processor = VideoFaceProcessor(
    compreface_processor=cf_processor,
    sample_rate=5,
    auto_enroll=True,
    update_quality=True
)
results = processor.process_video(video_path)
```

---

## JSON File Formats

### cache/fragments.json

Tracks all video fragments in the cache.

```json
{
  "updated": "2026-01-08T20:00:00.000000",
  "total_fragments": 10,
  "max_fragments": 50,
  "latest": {
    "filename": "fragment_20260108_200000_0000.avi",
    "session_id": "20260108_200000",
    "fragment_index": 0,
    "start_time": "2026-01-08T20:00:00.000000",
    "end_time": "2026-01-08T20:00:10.000000",
    "duration_seconds": 10.0,
    "frame_count": 150,
    "fps": 15,
    "width": 1920,
    "height": 1080,
    "file_size_bytes": 5242880
  },
  "fragments": [
    { "...same structure as latest..." }
  ]
}
```

### face_cache/all_faces.json

Cumulative database of all detected faces.

```json
{
  "created": "2026-01-08T20:00:00.000000",
  "updated": "2026-01-08T20:30:00.000000",
  "total_faces": 5,
  "faces": {
    "face_abc12345": {
      "face_id": "face_abc12345",
      "name": "John Doe",
      "description": "",
      "original_id": "face_abc12345",
      "first_seen": "2026-01-08T20:00:00.000000",
      "last_seen": "2026-01-08T20:30:00.000000",
      "detection_count": 15,
      "best_quality_score": 75.5,
      "age": {
        "probability": 0.95,
        "low": 25,
        "high": 32
      },
      "gender": {
        "probability": 0.98,
        "value": "male"
      },
      "detections": [
        {
          "timestamp": "2026-01-08T20:30:00.000000",
          "box": {
            "probability": 0.99,
            "x_min": 100,
            "y_min": 50,
            "x_max": 300,
            "y_max": 350
          },
          "landmarks": [[150, 120], [250, 120], [200, 180], [160, 280], [240, 280]],
          "source": "fragment_20260108_200000_0000.avi"
        }
      ]
    }
  }
}
```

### face_cache/current_faces.json

Results from the latest video processing.

```json
{
  "source_file": "fragment_20260108_200000_0000.avi",
  "processed_at": "2026-01-08T20:00:30.000000",
  "video_info": {
    "width": 1920,
    "height": 1080,
    "fps": 15.0,
    "total_frames": 150,
    "duration_seconds": 10.0
  },
  "processing_time_seconds": 25.5,
  "total_faces_detected": 10,
  "unique_detections": 2,
  "faces": [
    {
      "frame": 15,
      "timestamp": 1.0,
      "source": "fragment_20260108_200000_0000.avi",
      "face_id": "face_abc12345",
      "box": {
        "probability": 0.99,
        "x_min": 100,
        "y_min": 50,
        "x_max": 300,
        "y_max": 350
      },
      "landmarks": [[150, 120], [250, 120], [200, 180], [160, 280], [240, 280]],
      "age": {"probability": 0.95, "low": 25, "high": 32},
      "gender": {"probability": 0.98, "value": "male"},
      "subjects": [
        {"subject": "John Doe", "similarity": 0.95}
      ]
    }
  ],
  "face_summary": [
    {
      "face_id": "face_abc12345",
      "detection_count": 5,
      "first_frame": 15,
      "last_frame": 135,
      "age": {"probability": 0.95, "low": 25, "high": 32},
      "gender": {"probability": 0.98, "value": "male"},
      "subjects": [{"subject": "John Doe", "similarity": 0.95}]
    }
  ]
}
```

### opencv_cache/<video>_detections.json

Object detection results for a processed video.

```json
{
  "source_file": "fragment_20260108_200000_0000.avi",
  "processed_at": "2026-01-08T20:01:00.000000",
  "model": "yolo11m",
  "confidence_threshold": 0.5,
  "video_info": {
    "width": 1920,
    "height": 1080,
    "fps": 15.0,
    "total_frames": 150,
    "duration_seconds": 10.0
  },
  "processing_time_seconds": 15.2,
  "total_detections": 250,
  "detections_by_class": {
    "person": 200,
    "car": 30,
    "dog": 20
  },
  "frames": [
    {
      "frame": 1,
      "timestamp": 0.067,
      "detections": [
        {
          "class": "person",
          "confidence": 0.95,
          "box": {
            "x1": 100,
            "y1": 50,
            "x2": 300,
            "y2": 400
          },
          "center": [200, 225],
          "size": [200, 350]
        }
      ]
    }
  ]
}
```

---

## Configuration

### Environment Setup

```bash
# Install dependencies
pip3 install flask opencv-python numpy ultralytics compreface-sdk pyfakewebcam

# Install system dependencies (for virtual camera)
sudo apt-get install v4l2loopback-dkms v4l2loopback-utils libgl1-mesa-glx

# Load virtual camera kernel module
sudo modprobe v4l2loopback devices=1 video_nr=20 card_label="AndroidStream" exclusive_caps=1
```

### CompreFace Setup

CompreFace runs as a Docker container:

```bash
# Start CompreFace (docker-compose)
docker-compose up -d

# Access UI at http://localhost:8000
# Create a Recognition Service and get the API key
```

---

## Usage Examples

### Start the Receiver Server

```bash
cd /home/ling/main/videostream
python3 reciever.py
```

### Process Latest Fragment with YOLO

```bash
# Basic processing
python3 opencv.py --latest

# With specific model and save video
python3 opencv.py --latest --model yolo11x --save-video

# Watch mode
python3 opencv.py --watch --interval 10
```

### Facial Recognition

```bash
# Process specific file with auto-enroll
python3 facial_recognition.py --file fragment_20260108_200000_0000.avi --auto-enroll

# List all known faces
python3 facial_recognition.py --list-faces

# Rename a face
python3 facial_recognition.py --rename face_abc12345 "John Doe"

# Clear all faces
python3 facial_recognition.py --clear

# Watch mode
python3 facial_recognition.py --watch --auto-enroll
```

### Android App Configuration

Configure your Android app to send frames to:

```
POST http://<server-ip>:5000/camera/frame
Content-Type: image/jpeg
Body: <JPEG bytes>
```

Start/stop session:
```
POST http://<server-ip>:5000/stream/start
POST http://<server-ip>:5000/stream/stop
```

---

## Directory Structure

```
/home/ling/main/videostream/
├── reciever.py              # Video stream receiver server
├── opencv.py                # YOLO object detection
├── facial_recognition.py    # CompreFace facial recognition
├── videostream.md           # This documentation
├── cache/                   # Video fragments
│   ├── fragments.json       # Fragment metadata
│   └── fragment_*.avi       # Video files
├── opencv_cache/            # Object detection results
│   ├── *_detections.json    # Detection results
│   └── *_processed.avi      # Videos with overlays
└── face_cache/              # Facial recognition results
    ├── all_faces.json       # Cumulative face database
    ├── current_faces.json   # Latest processing results
    └── face_images/         # Cropped face images
        └── face_*.jpg       # Individual face images
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `libGL.so.1 not found` | `sudo apt-get install libgl1-mesa-glx` |
| Virtual camera not found | `sudo modprobe v4l2loopback devices=1 video_nr=20` |
| CompreFace connection refused | Ensure Docker container is running |
| High jitter in frame timing | Check Android app for buffering issues |
| Duplicate face enrollments | Use `--clear` then re-process with updated script |

### Debug Commands

```bash
# Check receiver health
curl http://localhost:5000/health

# Get streaming stats
curl http://localhost:5000/debug/stats

# List fragments
curl http://localhost:5000/fragments
```
