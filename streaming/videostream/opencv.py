#!/usr/bin/env python3
"""
OpenCV Video Fragment Analyzer

Analyzes video fragments from the stream receiver using CPU-based object detection.
Uses YOLOv4-tiny or MobileNet SSD for efficient CPU inference.
Outputs processed videos with bounding boxes and JSON results.
# Process the latest fragment
python3 opencv.py

# Process all unprocessed fragments
python3 opencv.py --all

# Watch mode - continuously process new fragments
python3 opencv.py --watch --interval 10

# Process a specific file
python3 opencv.py --file fragment_20260108_123456_0001.avi
"""

import os
import sys
import json
import time
import cv2
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Get number of CPUs and configure OpenCV threading
NUM_CPUS = multiprocessing.cpu_count()
print(f"[INFO] Detected {NUM_CPUS} CPUs")

# Set OpenCV to use all available threads
cv2.setNumThreads(NUM_CPUS)

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, 'cache')
OPENCV_CACHE_DIR = os.path.join(SCRIPT_DIR, 'opencv_cache')
FRAGMENTS_METADATA = os.path.join(CACHE_DIR, 'fragments.json')
PROCESSED_METADATA = os.path.join(OPENCV_CACHE_DIR, 'processed_fragments.json')
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

# Detection settings
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4  # Non-maximum suppression threshold
INPUT_WIDTH = 416
INPUT_HEIGHT = 416

# Ensure directories exist
os.makedirs(OPENCV_CACHE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# COCO class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Generate random colors for each class
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)

# Available models configuration
# Darknet-based models (OpenCV DNN)
DARKNET_MODELS = {
    'yolov4-tiny': {
        'cfg_url': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg',
        'weights_url': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights',
        'cfg_file': 'yolov4-tiny.cfg',
        'weights_file': 'yolov4-tiny.weights',
        'input_size': 416,
        'description': 'Fast, lightweight (~23MB) - Good for real-time',
        'accuracy': 'Low-Medium'
    },
    'yolov4': {
        'cfg_url': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg',
        'weights_url': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4.weights',
        'cfg_file': 'yolov4.cfg',
        'weights_file': 'yolov4.weights',
        'input_size': 608,
        'description': 'Accurate, larger model (~256MB)',
        'accuracy': 'High'
    },
    'yolov4-csp': {
        'cfg_url': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-csp.cfg',
        'weights_url': 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-csp.weights',
        'cfg_file': 'yolov4-csp.cfg',
        'weights_file': 'yolov4-csp.weights',
        'input_size': 512,
        'description': 'Balanced accuracy/speed (~202MB)',
        'accuracy': 'High'
    },
    'yolov3': {
        'cfg_url': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
        'weights_url': 'https://pjreddie.com/media/files/yolov3.weights',
        'cfg_file': 'yolov3.cfg',
        'weights_file': 'yolov3.weights',
        'input_size': 608,
        'description': 'Classic YOLOv3 (~248MB)',
        'accuracy': 'Medium-High'
    },
    'yolov3-spp': {
        'cfg_url': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-spp.cfg',
        'weights_url': 'https://pjreddie.com/media/files/yolov3-spp.weights',
        'cfg_file': 'yolov3-spp.cfg',
        'weights_file': 'yolov3-spp.weights',
        'input_size': 608,
        'description': 'YOLOv3 with SPP (~248MB) - Better for varied object sizes',
        'accuracy': 'Medium-High'
    }
}

# Ultralytics YOLO11 models (latest generation)
YOLO11_MODELS = {
    'yolo11n': {
        'model_name': 'yolo11n.pt',
        'description': 'YOLO11 Nano - Fastest (~6MB)',
        'accuracy': 'Medium',
        'params': '2.6M'
    },
    'yolo11s': {
        'model_name': 'yolo11s.pt',
        'description': 'YOLO11 Small - Fast (~22MB)',
        'accuracy': 'Medium-High',
        'params': '9.4M'
    },
    'yolo11m': {
        'model_name': 'yolo11m.pt',
        'description': 'YOLO11 Medium - Balanced (~39MB)',
        'accuracy': 'High',
        'params': '20.1M'
    },
    'yolo11l': {
        'model_name': 'yolo11l.pt',
        'description': 'YOLO11 Large - Accurate (~49MB)',
        'accuracy': 'Very High',
        'params': '25.3M'
    },
    'yolo11x': {
        'model_name': 'yolo11x.pt',
        'description': 'YOLO11 XLarge - Most Accurate (~97MB)',
        'accuracy': 'Highest',
        'params': '56.9M'
    }
}

# Combined models dictionary
MODELS = {**DARKNET_MODELS, **YOLO11_MODELS}

# Default model - use YOLO11 nano for best balance of speed and accuracy
DEFAULT_MODEL = 'yolo11n'


class YOLO11Detector:
    """Object detector using Ultralytics YOLO11 models."""
    
    def __init__(self, model_name='yolo11n'):
        self.model = None
        self.model_name = model_name
        self.model_config = YOLO11_MODELS.get(model_name)
        
        if not self.model_config:
            print(f"[ERROR] Unknown YOLO11 model: {model_name}")
            self.model_name = 'yolo11n'
            self.model_config = YOLO11_MODELS['yolo11n']
        
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO11 model using Ultralytics."""
        try:
            from ultralytics import YOLO
            
            print(f"[INFO] Loading {self.model_name} model...")
            print(f"[INFO] Model: {self.model_config['description']}")
            print(f"[INFO] Accuracy: {self.model_config['accuracy']}")
            print(f"[INFO] Parameters: {self.model_config['params']}")
            
            # Load model - will auto-download if not present
            model_path = os.path.join(MODELS_DIR, self.model_config['model_name'])
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                print(f"[INFO] Downloading {self.model_name}...")
                self.model = YOLO(self.model_config['model_name'])
                # Save to models directory
                # Model is auto-cached by Ultralytics
            
            print(f"[INFO] {self.model_name} model loaded successfully")
            
        except ImportError:
            print("[ERROR] Ultralytics not installed. Run: pip3 install ultralytics")
            self.model = None
        except Exception as e:
            print(f"[ERROR] Failed to load YOLO11 model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def detect(self, frame):
        """
        Detect objects in a frame using YOLO11.
        
        Returns:
            List of detections in the same format as ObjectDetector.
        """
        if self.model is None:
            return []
        
        # Run inference
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    
                    x1, y1, x2, y2 = map(int, box)
                    w = x2 - x1
                    h = y2 - y1
                    
                    # Get class name from model
                    class_name = result.names[cls_id] if cls_id in result.names else f"class_{cls_id}"
                    
                    detections.append({
                        'class': class_name,
                        'class_id': cls_id,
                        'confidence': round(conf, 3),
                        'box': {
                            'x': max(0, x1),
                            'y': max(0, y1),
                            'width': w,
                            'height': h
                        },
                        'center': {
                            'x': x1 + w // 2,
                            'y': y1 + h // 2
                        }
                    })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on the frame."""
        for det in detections:
            box = det['box']
            x, y, w, h = box['x'], box['y'], box['width'], box['height']
            class_id = det['class_id'] % len(COLORS)
            label = f"{det['class']}: {det['confidence']:.2f}"
            
            # Get color for this class
            color = [int(c) for c in COLORS[class_id]]
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(frame, (x, y - text_height - 10), 
                         (x + text_width, y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame


class ObjectDetector:
    """CPU-based object detector using OpenCV DNN module (for Darknet models)."""
    
    def __init__(self, model_name=None):
        self.net = None
        self.model_name = model_name or 'yolov4-tiny'
        self.model_config = DARKNET_MODELS.get(self.model_name)
        self.output_layers = None
        self.input_size = INPUT_WIDTH
        
        if not self.model_config:
            print(f"[ERROR] Unknown Darknet model: {self.model_name}")
            print(f"[INFO] Available Darknet models: {', '.join(DARKNET_MODELS.keys())}")
            self.model_name = 'yolov4-tiny'
            self.model_config = DARKNET_MODELS['yolov4-tiny']
        
        self._load_model()
    
    def _download_model_files(self):
        """Download model files if not present."""
        import urllib.request
        
        if not self.model_config:
            return None, None
            
        cfg_file = os.path.join(MODELS_DIR, self.model_config['cfg_file'])
        weights_file = os.path.join(MODELS_DIR, self.model_config['weights_file'])
        
        if not os.path.exists(cfg_file):
            print(f"[INFO] Downloading {self.model_name} config...")
            urllib.request.urlretrieve(self.model_config['cfg_url'], cfg_file)
            print(f"[INFO] Downloaded: {cfg_file}")
        
        if not os.path.exists(weights_file):
            weights_size = self._get_weights_size()
            print(f"[INFO] Downloading {self.model_name} weights ({weights_size})...")
            print(f"[INFO] This may take a while for larger models...")
            urllib.request.urlretrieve(self.model_config['weights_url'], weights_file)
            print(f"[INFO] Downloaded: {weights_file}")
        
        return cfg_file, weights_file
    
    def _get_weights_size(self):
        """Get approximate weights file size for display."""
        sizes = {
            'yolov4-tiny': '~23MB',
            'yolov4': '~256MB',
            'yolov4-csp': '~202MB',
            'yolov3': '~248MB',
            'yolov3-spp': '~248MB'
        }
        return sizes.get(self.model_name, 'unknown size')
    
    def _load_model(self):
        """Load the object detection model."""
        try:
            cfg_file, weights_file = self._download_model_files()
            
            print(f"[INFO] Loading {self.model_name} model with {NUM_CPUS} threads...")
            print(f"[INFO] Model: {self.model_config['description']}")
            print(f"[INFO] Accuracy: {self.model_config['accuracy']}")
            
            self.net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)
            
            # Use OpenCV backend with CPU target - this enables multi-threading
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Set input size from model config
            self.input_size = self.model_config['input_size']
            
            print(f"[INFO] Input size: {self.input_size}x{self.input_size}")
            print(f"[INFO] OpenCV using {cv2.getNumThreads()} threads for inference")
            
            # Get output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            print(f"[INFO] {self.model_name} model loaded successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self.net = None
    
    def detect(self, frame):
        """
        Detect objects in a frame.
        
        Returns:
            List of detections, each containing:
            {
                'class': class_name,
                'class_id': class_index,
                'confidence': float,
                'box': {'x': int, 'y': int, 'width': int, 'height': int},
                'center': {'x': int, 'y': int}
            }
        """
        if self.net is None:
            return []
        
        height, width = frame.shape[:2]
        
        # Create blob from image using model's input size
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (self.input_size, self.input_size),
                                      swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Run forward pass
        outputs = self.net.forward(self.output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > CONFIDENCE_THRESHOLD:
                    # Scale bounding box to frame size
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Calculate top-left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                detections.append({
                    'class': COCO_CLASSES[class_ids[i]],
                    'class_id': int(class_ids[i]),
                    'confidence': round(confidences[i], 3),
                    'box': {
                        'x': max(0, x),
                        'y': max(0, y),
                        'width': w,
                        'height': h
                    },
                    'center': {
                        'x': x + w // 2,
                        'y': y + h // 2
                    }
                })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on the frame."""
        for det in detections:
            box = det['box']
            x, y, w, h = box['x'], box['y'], box['width'], box['height']
            class_id = det['class_id']
            label = f"{det['class']}: {det['confidence']:.2f}"
            
            # Get color for this class
            color = [int(c) for c in COLORS[class_id]]
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame


def create_detector(model_name):
    """Factory function to create the appropriate detector based on model name."""
    if model_name in YOLO11_MODELS:
        return YOLO11Detector(model_name=model_name)
    elif model_name in DARKNET_MODELS:
        return ObjectDetector(model_name=model_name)
    else:
        print(f"[WARN] Unknown model '{model_name}', defaulting to {DEFAULT_MODEL}")
        return YOLO11Detector(model_name=DEFAULT_MODEL)


class FragmentProcessor:
    """Processes video fragments and generates analysis results."""
    
    def __init__(self, model_name=None):
        self.model_name = model_name or DEFAULT_MODEL
        self.detector = create_detector(self.model_name)
        self.processed_files = set()
        self._load_processed_metadata()
    
    def _load_processed_metadata(self):
        """Load list of already processed files."""
        try:
            if os.path.exists(PROCESSED_METADATA):
                with open(PROCESSED_METADATA, 'r') as f:
                    data = json.load(f)
                    self.processed_files = set(f['source_file'] for f in data.get('fragments', []))
        except Exception as e:
            print(f"[WARN] Could not load processed metadata: {e}")
            self.processed_files = set()
    
    def _save_processed_metadata(self, fragments_data):
        """Save processed fragments metadata."""
        try:
            # Load existing data
            existing = {'fragments': []}
            if os.path.exists(PROCESSED_METADATA):
                with open(PROCESSED_METADATA, 'r') as f:
                    existing = json.load(f)
            
            # Update with new data
            existing['updated'] = datetime.now().isoformat()
            existing['total_processed'] = len(existing.get('fragments', [])) + len(fragments_data)
            
            for frag in fragments_data:
                existing['fragments'].append(frag)
            
            # Keep only last 50 entries
            if len(existing['fragments']) > 50:
                existing['fragments'] = existing['fragments'][-50:]
            
            # Determine latest
            if existing['fragments']:
                existing['latest'] = existing['fragments'][-1]
            
            with open(PROCESSED_METADATA, 'w') as f:
                json.dump(existing, f, indent=2)
            
            print(f"[INFO] Saved processed metadata to {PROCESSED_METADATA}")
            
        except Exception as e:
            print(f"[ERROR] Could not save processed metadata: {e}")
    
    def get_latest_fragment(self):
        """Get the latest unprocessed fragment from the receiver's metadata."""
        try:
            if not os.path.exists(FRAGMENTS_METADATA):
                print(f"[WARN] Fragments metadata not found: {FRAGMENTS_METADATA}")
                return None
            
            with open(FRAGMENTS_METADATA, 'r') as f:
                data = json.load(f)
            
            fragments = data.get('fragments', [])
            
            # Find unprocessed fragments (from newest to oldest)
            for frag in reversed(fragments):
                source_file = frag.get('filename')
                if source_file and source_file not in self.processed_files:
                    source_path = os.path.join(CACHE_DIR, source_file)
                    if os.path.exists(source_path):
                        return frag
            
            # If all are processed, return the latest anyway for re-processing
            if fragments:
                latest = data.get('latest') or fragments[-1]
                source_path = os.path.join(CACHE_DIR, latest.get('filename', ''))
                if os.path.exists(source_path):
                    return latest
            
            return None
            
        except Exception as e:
            print(f"[ERROR] Could not read fragments metadata: {e}")
            return None
    
    def process_fragment(self, fragment_info):
        """
        Process a single video fragment.
        
        Returns:
            Dictionary with processing results
        """
        source_file = fragment_info.get('filename')
        source_path = os.path.join(CACHE_DIR, source_file)
        
        if not os.path.exists(source_path):
            print(f"[ERROR] Source file not found: {source_path}")
            return None
        
        print(f"\n[INFO] Processing: {source_file}")
        print(f"[INFO] Using {NUM_CPUS} CPU threads for inference")
        
        # Generate output filename
        base_name = os.path.splitext(source_file)[0]
        output_file = f"{base_name}_processed.avi"
        output_path = os.path.join(OPENCV_CACHE_DIR, output_file)
        
        # Open source video
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {source_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 15
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[INFO] Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Read all frames first for batch processing
        print("[INFO] Reading all frames...")
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        print(f"[INFO] Read {len(frames)} frames, starting parallel detection...")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing stats
        frame_results = []
        start_time = time.time()
        
        # Object tracking summary
        object_summary = {}
        
        # Process frames - OpenCV DNN uses internal threading
        # We process sequentially but each inference uses all threads
        for frame_count, frame in enumerate(frames, 1):
            # Run detection (uses all CPU threads internally)
            detections = self.detector.detect(frame)
            
            # Draw detections on frame
            annotated_frame = self.detector.draw_detections(frame.copy(), detections)
            
            # Add frame info overlay
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{len(frames)}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Objects: {len(detections)}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Write to output video
            out.write(annotated_frame)
            
            # Store frame results
            if detections:
                frame_results.append({
                    'frame': frame_count,
                    'timestamp': round(frame_count / fps, 3),
                    'objects': detections
                })
                
                # Update object summary
                for det in detections:
                    class_name = det['class']
                    if class_name not in object_summary:
                        object_summary[class_name] = {
                            'count': 0,
                            'total_confidence': 0,
                            'frames_detected': 0,
                            'positions': []
                        }
                    object_summary[class_name]['count'] += 1
                    object_summary[class_name]['total_confidence'] += det['confidence']
                    object_summary[class_name]['positions'].append({
                        'frame': frame_count,
                        'center': det['center'],
                        'size': {'width': det['box']['width'], 'height': det['box']['height']}
                    })
            
            # Progress update
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed if elapsed > 0 else 0
                print(f"[INFO] Progress: {frame_count}/{len(frames)} frames ({fps_processing:.1f} fps)")
        
        # Release resources
        out.release()
        
        processing_time = time.time() - start_time
        
        # Calculate summary statistics
        for class_name in object_summary:
            obj = object_summary[class_name]
            obj['avg_confidence'] = round(obj['total_confidence'] / obj['count'], 3) if obj['count'] > 0 else 0
            obj['frames_detected'] = len(set(p['frame'] for p in obj['positions']))
            # Keep only last 10 positions to limit JSON size
            obj['positions'] = obj['positions'][-10:]
            del obj['total_confidence']
        
        # Build result
        result = {
            'source_file': source_file,
            'processed_file': output_file,
            'processed_path': output_path,
            'processed_at': datetime.now().isoformat(),
            'processing_time_seconds': round(processing_time, 2),
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'duration_seconds': round(total_frames / fps, 2)
            },
            'detection_summary': {
                'total_detections': sum(obj['count'] for obj in object_summary.values()),
                'unique_classes': len(object_summary),
                'frames_with_detections': len(frame_results),
                'objects': object_summary
            },
            'frame_details': frame_results[-100:]  # Keep last 100 frame details
        }
        
        print(f"\n[INFO] Processing complete!")
        print(f"[INFO] Output: {output_path}")
        print(f"[INFO] Time: {processing_time:.2f}s")
        print(f"[INFO] Detected {result['detection_summary']['total_detections']} objects "
              f"across {result['detection_summary']['frames_with_detections']} frames")
        
        if object_summary:
            print("[INFO] Objects found:")
            for class_name, stats in object_summary.items():
                print(f"       - {class_name}: {stats['count']} detections "
                      f"(avg confidence: {stats['avg_confidence']:.2f})")
        
        return result
    
    def process_latest(self):
        """Process the latest unprocessed fragment."""
        fragment = self.get_latest_fragment()
        
        if not fragment:
            print("[WARN] No fragments to process")
            return None
        
        result = self.process_fragment(fragment)
        
        if result:
            self.processed_files.add(result['source_file'])
            self._save_processed_metadata([result])
        
        return result
    
    def process_all_new(self):
        """Process all new unprocessed fragments."""
        try:
            if not os.path.exists(FRAGMENTS_METADATA):
                print(f"[WARN] Fragments metadata not found")
                return []
            
            with open(FRAGMENTS_METADATA, 'r') as f:
                data = json.load(f)
            
            fragments = data.get('fragments', [])
            results = []
            
            for frag in fragments:
                source_file = frag.get('filename')
                if source_file and source_file not in self.processed_files:
                    source_path = os.path.join(CACHE_DIR, source_file)
                    if os.path.exists(source_path):
                        result = self.process_fragment(frag)
                        if result:
                            results.append(result)
                            self.processed_files.add(source_file)
            
            if results:
                self._save_processed_metadata(results)
            
            return results
            
        except Exception as e:
            print(f"[ERROR] Could not process fragments: {e}")
            return []


def list_models():
    """Print available models."""
    print("\n" + "=" * 70)
    print("YOLO11 Models (Ultralytics - Recommended)")
    print("=" * 70)
    for name, config in YOLO11_MODELS.items():
        default = " ★ DEFAULT" if name == DEFAULT_MODEL else ""
        print(f"  {name}{default}")
        print(f"    Description: {config['description']}")
        print(f"    Accuracy: {config['accuracy']}")
        print(f"    Parameters: {config['params']}")
        print()
    
    print("=" * 70)
    print("Legacy Darknet Models (OpenCV DNN)")
    print("=" * 70)
    for name, config in DARKNET_MODELS.items():
        print(f"  {name}")
        print(f"    Description: {config['description']}")
        print(f"    Accuracy: {config['accuracy']}")
        print(f"    Input size: {config['input_size']}x{config['input_size']}")
        print()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='OpenCV Video Fragment Analyzer with YOLO Object Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
YOLO11 Models (Recommended - Ultralytics):
  yolo11n      - Nano, fastest (~6MB)
  yolo11s      - Small, fast (~22MB)
  yolo11m      - Medium, balanced (~39MB)
  yolo11l      - Large, accurate (~49MB)
  yolo11x      - XLarge, most accurate (~97MB)

Legacy Darknet Models:
  yolov4-tiny  - Fast, lightweight (~23MB)
  yolov4       - Accurate (~256MB)
  yolov4-csp   - Balanced (~202MB)
  yolov3       - Classic (~248MB)
  yolov3-spp   - With SPP (~248MB)

Examples:
  python3 opencv.py --file fragment.avi              # Use default YOLO11n
  python3 opencv.py --model yolo11x --file frag.avi  # Use YOLO11 XLarge
  python3 opencv.py --model yolov4 --watch           # Use legacy YOLOv4
  python3 opencv.py --list-models                    # Show all models
        """
    )
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        choices=list(MODELS.keys()),
                        help=f'YOLO model to use (default: {DEFAULT_MODEL})')
    parser.add_argument('--all', action='store_true', help='Process all unprocessed fragments')
    parser.add_argument('--watch', action='store_true', help='Continuously watch for new fragments')
    parser.add_argument('--interval', type=int, default=5, help='Watch interval in seconds (default: 5)')
    parser.add_argument('--file', type=str, help='Process a specific file')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    args = parser.parse_args()
    
    # Update global confidence threshold
    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.confidence
    
    if args.list_models:
        list_models()
        return
    
    print(f"\n[INFO] Using model: {args.model}")
    print(f"[INFO] Confidence threshold: {args.confidence}")
    
    processor = FragmentProcessor(model_name=args.model)
    
    if args.file:
        # Process specific file
        fragment_info = {'filename': os.path.basename(args.file)}
        result = processor.process_fragment(fragment_info)
        if result:
            print(f"\n[RESULT] Saved to: {PROCESSED_METADATA}")
    
    elif args.all:
        # Process all new fragments
        results = processor.process_all_new()
        print(f"\n[INFO] Processed {len(results)} fragments")
    
    elif args.watch:
        # Watch mode - continuously process new fragments
        print(f"[INFO] Watching for new fragments (interval: {args.interval}s)")
        print("[INFO] Press Ctrl+C to stop")
        
        try:
            while True:
                result = processor.process_latest()
                if result:
                    print(f"[INFO] Processed: {result['source_file']}")
                else:
                    print(f"[INFO] No new fragments. Waiting...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[INFO] Stopped watching")
    
    else:
        # Process latest fragment
        result = processor.process_latest()
        if result:
            print(f"\n[RESULT] Results saved to: {PROCESSED_METADATA}")
        else:
            print("[INFO] No fragments to process")


if __name__ == '__main__':
    main()
