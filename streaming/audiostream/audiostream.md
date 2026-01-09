# Audio Stream System Documentation

A complete audio streaming system that receives audio from an Android app and saves it into 5-second WAV fragments.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Configuration](#configuration)
3. [HTTP Endpoints](#http-endpoints)
4. [Classes](#classes)
5. [JSON File Formats](#json-file-formats)
6. [Usage Examples](#usage-examples)

---

## System Overview

```
┌─────────────────┐     HTTP POST      ┌──────────────────┐
│  Android App    │ ─────────────────► │  Audio Receiver  │
│  (Microphone)   │   /audio/chunk     │ (audioreciever)  │
└─────────────────┘                    └────────┬─────────┘
                                                │
                                                ▼
                                       ┌────────────────┐
                                       │  cache/ folder │
                                       │  (WAV files)   │
                                       └────────────────┘
```

---

## Configuration

```python
HOST = '0.0.0.0'              # Listen on all interfaces
PORT = 5001                   # Server port
FRAGMENT_DURATION = 5         # Seconds per audio fragment
MAX_FRAGMENTS = 50            # Maximum fragments to keep
AUDIO_QUEUE_SIZE = 500        # Audio chunk buffer size

# Default audio settings
DEFAULT_SAMPLE_RATE = 44100   # Hz
DEFAULT_CHANNELS = 1          # Mono
DEFAULT_SAMPLE_WIDTH = 2      # 16-bit audio (2 bytes per sample)
```

---

## HTTP Endpoints

### Root & Health

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server info and available endpoints |
| `/health` | GET | Health check with session status |

**Response `/`:**
```json
{
  "service": "Audio Stream Receiver",
  "status": "running",
  "version": "1.0",
  "endpoints": {
    "health": "/health",
    "stream_start": "/stream/start",
    "stream_stop": "/stream/stop",
    "audio_chunk": "/audio/chunk",
    "fragments": "/fragments"
  }
}
```

**Response `/health`:**
```json
{
  "status": "ok",
  "timestamp": "2026-01-08T20:00:00.000000",
  "active_session": "20260108_200000",
  "chunks_received": 500,
  "queue_pending": 3
}
```

### Streaming Control

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stream/start` | POST | Start a new streaming session |
| `/stream/stop` | POST | Stop current session and finalize fragment |

**Request `/stream/start`:**
```json
{
  "session_id": "optional_custom_id",
  "sample_rate": 44100,
  "channels": 1,
  "sample_width": 2
}
```

**Response `/stream/start`:**
```json
{
  "status": "started",
  "session_id": "20260108_200000",
  "sample_rate": 44100,
  "channels": 1,
  "sample_width": 2,
  "fragment_duration": 5,
  "message": "Audio stream session started"
}
```

**Response `/stream/stop`:**
```json
{
  "status": "stopped",
  "session_id": "20260108_200000",
  "chunks_received": 500,
  "bytes_received": 1024000,
  "fragments_saved": 5
}
```

### Audio Upload

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/audio/chunk` | POST | Send raw PCM audio chunk (primary method) |
| `/audio/wav` | POST | Send WAV-formatted audio chunk |

**Request `/audio/chunk`:**
- Body: Raw PCM bytes (16-bit signed, little-endian)
- Content-Type: `application/octet-stream`

**Response `/audio/chunk`:**
```json
{
  "status": "received",
  "bytes": 4096,
  "chunk_count": 150,
  "fragment": 3,
  "queued": true
}
```

**Request `/audio/wav`:**
- Body: WAV file data
- Content-Type: `audio/wav` or `application/octet-stream`

### Fragment Management

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
  "total_size_bytes": 4410000,
  "fragment_duration": 5,
  "fragments": ["fragment_20260108_200000_0000.wav", "..."]
}
```

**Response `/fragments/latest`:**
```json
{
  "status": "ok",
  "latest": {
    "filename": "fragment_20260108_200010_0002.wav",
    "session_id": "20260108_200000",
    "fragment_index": 2,
    "start_time": "2026-01-08T20:00:10.000000",
    "end_time": "2026-01-08T20:00:15.000000",
    "duration_seconds": 5.0,
    "sample_count": 220500,
    "sample_rate": 44100,
    "channels": 1,
    "sample_width": 2,
    "file_size_bytes": 441044
  },
  "total_fragments": 3,
  "current_recording": "fragment_20260108_200015_0003.wav"
}
```

### Debug & Buffer

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/debug/stats` | GET | Get detailed streaming statistics |
| `/audio/buffer` | GET | Get current audio buffer info |

**Response `/debug/stats`:**
```json
{
  "chunks_received": 500,
  "bytes_received": 1024000,
  "uptime_seconds": 60.5,
  "rate_kbps": 135.2,
  "last_chunk_ago": 0.05,
  "cache_files": 10,
  "cache_size_bytes": 4410000,
  "max_fragments": 50,
  "fragment_duration": 5,
  "session_id": "20260108_200000",
  "fragment_count": 3,
  "audio_settings": {
    "sample_rate": 44100,
    "channels": 1,
    "sample_width": 2
  },
  "queue": {
    "queued": 500,
    "processed": 497,
    "dropped": 0,
    "pending": 3
  }
}
```

**Response `/audio/buffer`:**
```json
{
  "buffer_samples": 441000,
  "buffer_seconds": 10.0,
  "sample_rate": 44100,
  "channels": 1,
  "last_update": 1704744000.5
}
```

---

## Classes

### Stats
Tracks streaming statistics.

```python
stats.log_chunk(size)    # Log a received chunk
stats.reset()            # Reset all stats
```

### AudioBuffer
Thread-safe buffer for sharing audio data.

```python
audio_buffer = AudioBuffer(max_seconds=10, sample_rate=44100, channels=1)
audio_buffer.add_audio(audio_bytes)     # Add audio samples
audio_buffer.get_audio(seconds=5)       # Get last N seconds
audio_buffer.clear()                    # Clear buffer
```

### AudioProcessingQueue
Async queue for handling burst traffic.

```python
audio_queue.start(stream_manager, audio_buffer)
audio_queue.add_chunk(audio_data)       # Add chunk to queue
audio_queue.get_stats()                 # Get queue statistics
audio_queue.stop()                      # Stop processing
```

### AudioStreamManager
Manages audio recording sessions and fragment files.

```python
stream_manager.start_session(session_id, sample_rate, channels, sample_width)
stream_manager.save_audio(audio_bytes)  # Save audio to current fragment
stream_manager.end_session()            # Finalize and end session
```

### AudioBufferReader
Helper class for programmatic access to audio buffer.

```python
from audioreciever import AudioBufferReader

reader = AudioBufferReader()
audio = reader.read(seconds=5)          # Get last 5 seconds
sample_rate = reader.get_sample_rate()
channels = reader.get_channels()
is_available = reader.is_available()
```

---

## JSON File Formats

### cache/fragments.json

Tracks all audio fragments in the cache.

```json
{
  "updated": "2026-01-08T20:00:15.000000",
  "total_fragments": 3,
  "max_fragments": 50,
  "latest": {
    "filename": "fragment_20260108_200010_0002.wav",
    "session_id": "20260108_200000",
    "fragment_index": 2,
    "start_time": "2026-01-08T20:00:10.000000",
    "end_time": "2026-01-08T20:00:15.000000",
    "duration_seconds": 5.0,
    "sample_count": 220500,
    "sample_rate": 44100,
    "channels": 1,
    "sample_width": 2,
    "file_size_bytes": 441044
  },
  "fragments": [
    {
      "filename": "fragment_20260108_200000_0000.wav",
      "session_id": "20260108_200000",
      "fragment_index": 0,
      "start_time": "2026-01-08T20:00:00.000000",
      "end_time": "2026-01-08T20:00:05.000000",
      "duration_seconds": 5.0,
      "sample_count": 220500,
      "sample_rate": 44100,
      "channels": 1,
      "sample_width": 2,
      "file_size_bytes": 441044
    }
  ]
}
```

---

## Usage Examples

### Start the Server

```bash
cd /home/ling/main/audiostream
python3 audioreciever.py
```

Output:
```
╔══════════════════════════════════════════════════════════════════════╗
║              Audio Stream Receiver Server                            ║
╠══════════════════════════════════════════════════════════════════════╣
║  Listening on: http://0.0.0.0:5001                                   ║
║  Cache directory: /home/ling/main/audiostream/cache                  ║
║  Fragment duration: 5 seconds                                        ║
...
```

### Android App Integration

Configure your Android app to send audio to:

**Start session:**
```
POST http://<server-ip>:5001/stream/start
Content-Type: application/json

{
  "sample_rate": 44100,
  "channels": 1,
  "sample_width": 2
}
```

**Send audio chunks:**
```
POST http://<server-ip>:5001/audio/chunk
Content-Type: application/octet-stream
Body: <raw PCM bytes>
```

**Stop session:**
```
POST http://<server-ip>:5001/stream/stop
```

### Check Server Status

```bash
# Health check
curl http://localhost:5001/health

# Get statistics
curl http://localhost:5001/debug/stats

# List fragments
curl http://localhost:5001/fragments

# Get latest fragment
curl http://localhost:5001/fragments/latest
```

### Programmatic Access

```python
from audioreciever import AudioBufferReader
import numpy as np

# Create reader
reader = AudioBufferReader()

# Wait for audio
while not reader.is_available():
    time.sleep(0.1)

# Read last 5 seconds of audio
audio = reader.read(seconds=5)

# Get audio settings
sample_rate = reader.get_sample_rate()
channels = reader.get_channels()

# Process audio (example: calculate RMS)
rms = np.sqrt(np.mean(audio.astype(np.float32)**2))
print(f"RMS: {rms}")
```

---

## Directory Structure

```
/home/ling/main/audiostream/
├── audioreciever.py         # Audio stream receiver server
├── audiostream.md           # This documentation
└── cache/                   # Audio fragments
    ├── fragments.json       # Fragment metadata
    └── fragment_*.wav       # Audio files (WAV format)
```

---

## Audio Format Details

| Property | Default | Description |
|----------|---------|-------------|
| Sample Rate | 44100 Hz | Samples per second |
| Channels | 1 (Mono) | Number of audio channels |
| Sample Width | 2 bytes | Bits per sample (16-bit) |
| Format | PCM | Raw pulse-code modulation |
| Container | WAV | RIFF WAVE format |

### Calculating Data Sizes

```
Bytes per second = sample_rate × channels × sample_width
                 = 44100 × 1 × 2
                 = 88,200 bytes/sec

5-second fragment = 88,200 × 5 = 441,000 bytes (+ WAV header)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No audio received | Check Android app is sending to correct IP:5001 |
| Choppy audio | Increase chunk size on Android side |
| High latency | Check network connection, reduce queue size |
| Fragments not saving | Check write permissions on cache folder |

### Debug Commands

```bash
# Check health
curl http://localhost:5001/health

# Get detailed stats
curl http://localhost:5001/debug/stats

# Check audio buffer
curl http://localhost:5001/audio/buffer
```
