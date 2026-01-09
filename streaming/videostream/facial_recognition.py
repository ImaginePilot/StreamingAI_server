#!/usr/bin/env python3
"""
Facial Recognition Video Processor using CompreFace

Processes video fragments and performs facial recognition using CompreFace API.
Stores results in all_faces.json (cumulative) and current_faces.json (latest).
"""

import os
import sys
import json
import time
import cv2
import numpy as np
from datetime import datetime
import tempfile
import argparse

# CompreFace SDK imports
from compreface import CompreFace
from compreface.service import RecognitionService
from compreface.collections import FaceCollection

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, 'cache')
FACE_CACHE_DIR = os.path.join(SCRIPT_DIR, 'face_cache')
FRAGMENTS_METADATA = os.path.join(CACHE_DIR, 'fragments.json')
ALL_FACES_FILE = os.path.join(FACE_CACHE_DIR, 'all_faces.json')
CURRENT_FACES_FILE = os.path.join(FACE_CACHE_DIR, 'current_faces.json')

# CompreFace configuration
COMPREFACE_DOMAIN = 'http://localhost'
COMPREFACE_PORT = '8000'
COMPREFACE_API_KEY = '185efb81-9b55-4f72-b38c-d8e2bd0d4a3b'

# Detection settings
FRAME_SAMPLE_RATE = 5  # Process every Nth frame to reduce API calls
FACE_DETECTION_THRESHOLD = 0.8  # Minimum confidence for face detection
SIMILARITY_THRESHOLD = 0.9  # Threshold for considering faces as the same person
RECOGNITION_THRESHOLD = 0.85  # Threshold for recognizing existing subjects (skip enrollment)

# Directory for saved face images
FACE_IMAGES_DIR = os.path.join(FACE_CACHE_DIR, 'face_images')

# Ensure directories exist
os.makedirs(FACE_CACHE_DIR, exist_ok=True)
os.makedirs(FACE_IMAGES_DIR, exist_ok=True)


class FaceDatabase:
    """Manages the face database stored in JSON files."""
    
    def __init__(self):
        self.all_faces = self._load_all_faces()
    
    def _load_all_faces(self):
        """Load existing faces from all_faces.json."""
        try:
            if os.path.exists(ALL_FACES_FILE):
                with open(ALL_FACES_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[WARN] Could not load all_faces.json: {e}")
        
        return {
            'created': datetime.now().isoformat(),
            'updated': datetime.now().isoformat(),
            'total_faces': 0,
            'faces': {}
        }
    
    def _save_all_faces(self):
        """Save faces to all_faces.json."""
        self.all_faces['updated'] = datetime.now().isoformat()
        self.all_faces['total_faces'] = len(self.all_faces['faces'])
        
        try:
            with open(ALL_FACES_FILE, 'w') as f:
                json.dump(self.all_faces, f, indent=2)
            print(f"[INFO] Saved {len(self.all_faces['faces'])} faces to all_faces.json")
        except Exception as e:
            print(f"[ERROR] Could not save all_faces.json: {e}")
    
    def _save_current_faces(self, faces_data):
        """Save current detection results to current_faces.json."""
        try:
            with open(CURRENT_FACES_FILE, 'w') as f:
                json.dump(faces_data, f, indent=2)
            print(f"[INFO] Saved current faces to current_faces.json")
        except Exception as e:
            print(f"[ERROR] Could not save current_faces.json: {e}")
    
    def _generate_face_id(self):
        """Generate a unique face ID."""
        import uuid
        return f"face_{uuid.uuid4().hex[:8]}"
    
    def add_or_update_face(self, face_data, embedding=None):
        """
        Add a new face or update existing one based on similarity.
        
        Returns the face_id (existing or new).
        """
        # For now, create a new entry for each unique detection
        # In a real system, you would compare embeddings to find matches
        
        face_id = self._generate_face_id()
        
        face_entry = {
            'face_id': face_id,
            'name': '',  # To be filled by user later
            'description': '',  # To be filled by user later
            'first_seen': datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat(),
            'detection_count': 1,
            'age': face_data.get('age'),
            'gender': face_data.get('gender'),
            'detections': [{
                'timestamp': datetime.now().isoformat(),
                'box': face_data.get('box'),
                'landmarks': face_data.get('landmarks'),
                'source': face_data.get('source')
            }]
        }
        
        self.all_faces['faces'][face_id] = face_entry
        return face_id
    
    def update_face_detection(self, face_id, face_data):
        """Update an existing face with new detection data."""
        if face_id in self.all_faces['faces']:
            face = self.all_faces['faces'][face_id]
            face['last_seen'] = datetime.now().isoformat()
            face['detection_count'] += 1
            
            # Keep last 10 detections
            face['detections'].append({
                'timestamp': datetime.now().isoformat(),
                'box': face_data.get('box'),
                'landmarks': face_data.get('landmarks'),
                'source': face_data.get('source')
            })
            if len(face['detections']) > 10:
                face['detections'] = face['detections'][-10:]
    
    def finalize(self, current_faces):
        """Save all data to files."""
        self._save_all_faces()
        self._save_current_faces(current_faces)
    
    def rename_face(self, face_id, new_name):
        """
        Rename a face in the database.
        
        Args:
            face_id: The face ID to rename
            new_name: The new name for the face
            
        Returns:
            True if successful, False otherwise
        """
        if face_id not in self.all_faces['faces']:
            print(f"[ERROR] Face ID '{face_id}' not found in database")
            return False
        
        self.all_faces['faces'][face_id]['name'] = new_name
        self.all_faces['updated'] = datetime.now().isoformat()
        self._save_all_faces()
        print(f"[INFO] Updated face '{face_id}' name to '{new_name}' in database")
        return True
    
    def get_face(self, face_id):
        """Get face data by ID."""
        return self.all_faces['faces'].get(face_id)
    
    def update_face_image_quality(self, face_id, quality_score):
        """
        Update the best image quality score for a face.
        
        Args:
            face_id: The face ID
            quality_score: The new quality score
        """
        if face_id in self.all_faces['faces']:
            self.all_faces['faces'][face_id]['best_quality_score'] = quality_score
            self._save_all_faces()


class ImageQualityAnalyzer:
    """Analyzes image quality for face images."""
    
    @staticmethod
    def calculate_quality_score(image):
        """
        Calculate a quality score for a face image.
        Higher score = better quality.
        
        Factors considered:
        - Sharpness (Laplacian variance)
        - Brightness
        - Contrast
        - Face size
        
        Args:
            image: OpenCV image (BGR format) or image path
            
        Returns:
            Quality score (0-100)
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                return 0
        
        if image is None or image.size == 0:
            return 0
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Sharpness - Laplacian variance (higher = sharper)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 500 * 30, 30)  # Max 30 points
        
        # 2. Brightness - should be in a good range (not too dark/bright)
        mean_brightness = np.mean(gray)
        # Ideal brightness around 120-140
        brightness_deviation = abs(mean_brightness - 130)
        brightness_score = max(0, 20 - (brightness_deviation / 130 * 20))  # Max 20 points
        
        # 3. Contrast - standard deviation of pixel values
        contrast = np.std(gray)
        contrast_score = min(contrast / 60 * 20, 20)  # Max 20 points
        
        # 4. Size - larger face images are generally better
        height, width = image.shape[:2]
        size = height * width
        size_score = min(size / 40000 * 30, 30)  # Max 30 points (good at 200x200)
        
        total_score = sharpness_score + brightness_score + contrast_score + size_score
        
        return round(total_score, 2)
    
    @staticmethod
    def is_better_quality(new_image, old_image_path):
        """
        Check if a new image is better quality than an existing one.
        
        Args:
            new_image: New image (OpenCV format or path)
            old_image_path: Path to existing image
            
        Returns:
            Tuple of (is_better, new_score, old_score)
        """
        new_score = ImageQualityAnalyzer.calculate_quality_score(new_image)
        old_score = ImageQualityAnalyzer.calculate_quality_score(old_image_path)
        
        return (new_score > old_score, new_score, old_score)


class FaceManager:
    """
    Manages face operations including renaming and image quality updates.
    Syncs changes between local database and CompreFace.
    """
    
    def __init__(self, compreface_processor=None, face_database=None):
        self.processor = compreface_processor or CompreFaceProcessor()
        self.face_db = face_database or FaceDatabase()
        self.quality_analyzer = ImageQualityAnalyzer()
    
    def rename_face(self, face_id, new_name):
        """
        Rename a face in both local database and CompreFace.
        
        Args:
            face_id: The face ID (subject name in CompreFace)
            new_name: The new name
            
        Returns:
            True if successful, False otherwise
        """
        # First, rename in CompreFace
        if self.processor.rename_subject(face_id, new_name):
            # Then update local database
            if face_id in self.face_db.all_faces['faces']:
                # Move the entry to new key and update name
                face_data = self.face_db.all_faces['faces'].pop(face_id)
                face_data['face_id'] = new_name
                face_data['name'] = new_name
                face_data['original_id'] = face_id  # Keep track of original ID
                self.face_db.all_faces['faces'][new_name] = face_data
                self.face_db.all_faces['updated'] = datetime.now().isoformat()
                self.face_db._save_all_faces()
                
                print(f"[SUCCESS] Renamed face '{face_id}' to '{new_name}' in both CompreFace and local database")
                return True
            else:
                # Face not in local DB but renamed in CompreFace
                print(f"[WARN] Face '{face_id}' renamed in CompreFace but not found in local database")
                return True
        
        return False
    
    def update_face_if_better(self, face_id, new_image, new_image_path=None):
        """
        Update a face's image in CompreFace if the new image is better quality.
        
        Args:
            face_id: The face ID (subject name in CompreFace)
            new_image: New image (OpenCV format)
            new_image_path: Optional path if image is already saved
            
        Returns:
            Tuple of (was_updated, new_score, old_score)
        """
        # Get existing image path
        existing_image_path = os.path.join(FACE_IMAGES_DIR, f"{face_id}.jpg")
        
        if not os.path.exists(existing_image_path):
            # No existing image, save this one
            if new_image_path is None:
                new_image_path = existing_image_path
                cv2.imwrite(new_image_path, new_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Add to CompreFace
            self.processor.add_face_to_subject(face_id, new_image_path)
            new_score = self.quality_analyzer.calculate_quality_score(new_image)
            self.face_db.update_face_image_quality(face_id, new_score)
            print(f"[INFO] Added first image for '{face_id}' (quality: {new_score})")
            return (True, new_score, 0)
        
        # Compare quality
        is_better, new_score, old_score = self.quality_analyzer.is_better_quality(
            new_image, existing_image_path
        )
        
        if is_better:
            print(f"[INFO] Found better image for '{face_id}' (new: {new_score} vs old: {old_score})")
            
            # Save new image
            temp_path = os.path.join(FACE_IMAGES_DIR, f"{face_id}_new.jpg")
            cv2.imwrite(temp_path, new_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Add new image to CompreFace subject
            image_id = self.processor.add_face_to_subject(face_id, temp_path)
            
            if image_id:
                # Replace old image with new one
                os.replace(temp_path, existing_image_path)
                self.face_db.update_face_image_quality(face_id, new_score)
                print(f"[SUCCESS] Updated image for '{face_id}'")
                return (True, new_score, old_score)
            else:
                # Failed to add, clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return (False, new_score, old_score)
        else:
            print(f"[INFO] Existing image for '{face_id}' is better (existing: {old_score} vs new: {new_score})")
            return (False, new_score, old_score)
    
    def list_faces(self):
        """List all faces in the database with their info."""
        faces = []
        for face_id, face_data in self.face_db.all_faces['faces'].items():
            faces.append({
                'face_id': face_id,
                'name': face_data.get('name', ''),
                'detection_count': face_data.get('detection_count', 0),
                'first_seen': face_data.get('first_seen'),
                'last_seen': face_data.get('last_seen'),
                'age': face_data.get('age'),
                'gender': face_data.get('gender'),
                'quality_score': face_data.get('best_quality_score', 0)
            })
        return faces


class CompreFaceProcessor:
    """Processes video frames using CompreFace API."""
    
    def __init__(self, domain=COMPREFACE_DOMAIN, port=COMPREFACE_PORT, api_key=COMPREFACE_API_KEY):
        self.domain = domain
        self.port = port
        self.api_key = api_key
        self.compre_face = None
        self.recognition = None
        self.face_collection = None
        self.subjects = None
        self._init_compreface()
    
    def _init_compreface(self):
        """Initialize CompreFace connection."""
        try:
            print(f"[INFO] Connecting to CompreFace at {self.domain}:{self.port}")
            
            self.compre_face = CompreFace(
                domain=self.domain,
                port=self.port,
                options={
                    "limit": 0,
                    "det_prob_threshold": FACE_DETECTION_THRESHOLD,
                    "prediction_count": 1,
                    "face_plugins": "age,gender,landmarks",
                    "status": True
                }
            )
            
            self.recognition = self.compre_face.init_face_recognition(self.api_key)
            self.face_collection = self.recognition.get_face_collection()
            self.subjects = self.recognition.get_subjects()
            
            print("[INFO] CompreFace initialized successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize CompreFace: {e}")
            raise
    
    def detect_faces(self, image_path):
        """
        Detect and recognize faces in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detected faces with their attributes
        """
        try:
            result = self.recognition.recognize(image_path=image_path)
            return result.get('result', [])
        except Exception as e:
            print(f"[ERROR] Face detection failed: {e}")
            return []
    
    def get_all_subjects(self):
        """
        Get all subjects in CompreFace.
        
        Returns:
            List of subject names
        """
        try:
            result = self.subjects.list()
            return result.get('subjects', [])
        except Exception as e:
            print(f"[ERROR] Failed to get subjects: {e}")
            return []
    
    def add_subject(self, subject_name):
        """
        Add a new subject to CompreFace.
        
        Args:
            subject_name: Name for the new subject
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.subjects.add(subject_name)
            print(f"[INFO] Added new subject: {subject_name}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to add subject {subject_name}: {e}")
            return False
    
    def add_face_to_subject(self, subject_name, image_path):
        """
        Add a face image to a subject for recognition training.
        
        Args:
            subject_name: Name of the subject
            image_path: Path to the face image
            
        Returns:
            The image_id if successful, None otherwise
        """
        try:
            result = self.face_collection.add(image_path=image_path, subject=subject_name)
            image_id = result.get('image_id')
            print(f"[INFO] Added face to subject '{subject_name}' (image_id: {image_id})")
            return image_id
        except Exception as e:
            print(f"[ERROR] Failed to add face to subject {subject_name}: {e}")
            return None
    
    def delete_subject(self, subject_name):
        """
        Delete a subject from CompreFace.
        
        Args:
            subject_name: Name of the subject to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.subjects.delete(subject_name)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to delete subject {subject_name}: {e}")
            return False
    
    def delete_all_subjects(self):
        """
        Delete all subjects from CompreFace.
        
        Returns:
            Number of subjects deleted
        """
        subjects = self.get_all_subjects()
        deleted = 0
        
        print(f"[INFO] Deleting {len(subjects)} subjects from CompreFace...")
        
        for subject_name in subjects:
            if self.delete_subject(subject_name):
                deleted += 1
                print(f"[INFO] Deleted subject: {subject_name}")
        
        print(f"[INFO] Deleted {deleted}/{len(subjects)} subjects")
        return deleted
    
    def rename_subject(self, old_name, new_name):
        """
        Rename a subject in CompreFace.
        
        Args:
            old_name: Current subject name
            new_name: New subject name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.subjects.rename(old_name, new_name)
            print(f"[INFO] Renamed subject '{old_name}' to '{new_name}'")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to rename subject {old_name}: {e}")
            return False
    
    def get_subject_faces(self, subject_name):
        """
        Get all face images for a subject.
        
        Args:
            subject_name: Name of the subject
            
        Returns:
            List of face image IDs
        """
        try:
            result = self.face_collection.list(subject=subject_name)
            return result.get('faces', [])
        except Exception as e:
            print(f"[ERROR] Failed to get faces for subject {subject_name}: {e}")
            return []
    
    def delete_face_from_subject(self, image_id):
        """
        Delete a specific face image from a subject.
        
        Args:
            image_id: The image ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.face_collection.delete(image_id=image_id)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to delete face image {image_id}: {e}")
            return False
    
    def detect_faces_from_bytes(self, image_bytes):
        """
        Detect faces from image bytes.
        
        Args:
            image_bytes: Image as bytes
            
        Returns:
            List of detected faces
        """
        try:
            # Save to temporary file since SDK might not support bytes directly
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name
            
            try:
                result = self.recognition.recognize(image_path=tmp_path)
                return result.get('result', [])
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            print(f"[ERROR] Face detection from bytes failed: {e}")
            return []


class VideoFaceProcessor:
    """Processes video fragments for facial recognition."""
    
    def __init__(self, compreface_processor=None, sample_rate=5, auto_enroll=False, update_quality=True):
        self.face_db = FaceDatabase()
        self.processor = compreface_processor or CompreFaceProcessor()
        self.processed_files = set()
        self.sample_rate = sample_rate
        self.auto_enroll = auto_enroll
        self.update_quality = update_quality  # Whether to update with better quality images
        self.enrolled_faces = set()  # Track faces we've already enrolled
        self.face_manager = FaceManager(self.processor, self.face_db)
        self.quality_analyzer = ImageQualityAnalyzer()
        self._load_enrolled_faces()
    
    def _load_enrolled_faces(self):
        """Load list of already enrolled faces from CompreFace."""
        try:
            subjects = self.processor.get_all_subjects()
            self.enrolled_faces = set(subjects)
            print(f"[INFO] Loaded {len(self.enrolled_faces)} existing subjects from CompreFace")
        except Exception as e:
            print(f"[WARN] Could not load existing subjects: {e}")
    
    def _save_face_image(self, frame, box, face_id):
        """
        Crop and save a face image from a video frame.
        
        Args:
            frame: The video frame (numpy array)
            box: Bounding box dict with x_min, y_min, x_max, y_max
            face_id: Unique ID for the face
            
        Returns:
            Path to saved image or None
        """
        try:
            face_img = self._crop_face(frame, box)
            if face_img is None:
                return None
            
            # Save image
            image_path = os.path.join(FACE_IMAGES_DIR, f"{face_id}.jpg")
            cv2.imwrite(image_path, face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            return image_path
            
        except Exception as e:
            print(f"[ERROR] Failed to save face image: {e}")
            return None
    
    def _crop_face(self, frame, box):
        """
        Crop a face from a video frame.
        
        Args:
            frame: The video frame (numpy array)
            box: Bounding box dict with x_min, y_min, x_max, y_max
            
        Returns:
            Cropped face image or None
        """
        try:
            x_min = max(0, box['x_min'])
            y_min = max(0, box['y_min'])
            x_max = min(frame.shape[1], box['x_max'])
            y_max = min(frame.shape[0], box['y_max'])
            
            # Add some padding (10%)
            pad_x = int((x_max - x_min) * 0.1)
            pad_y = int((y_max - y_min) * 0.1)
            
            x_min = max(0, x_min - pad_x)
            y_min = max(0, y_min - pad_y)
            x_max = min(frame.shape[1], x_max + pad_x)
            y_max = min(frame.shape[0], y_max + pad_y)
            
            # Crop face
            face_img = frame[y_min:y_max, x_min:x_max]
            
            if face_img.size == 0:
                return None
            
            return face_img
            
        except Exception as e:
            print(f"[ERROR] Failed to crop face: {e}")
            return None
    
    def _enroll_face(self, face_id, image_path, face_data):
        """
        Enroll a new face in CompreFace.
        
        Args:
            face_id: Unique ID for the face (will be used as subject name)
            image_path: Path to the face image
            face_data: Additional face metadata
            
        Returns:
            True if successful
        """
        if face_id in self.enrolled_faces:
            return False
        
        try:
            # Create subject name with metadata
            age_info = face_data.get('age', {})
            gender_info = face_data.get('gender', {})
            
            age_str = f"{age_info.get('low', '?')}-{age_info.get('high', '?')}" if age_info else "unknown"
            gender_str = gender_info.get('value', 'unknown') if gender_info else "unknown"
            
            # Use face_id as subject name (can be renamed in UI later)
            subject_name = face_id
            
            # Add subject
            if self.processor.add_subject(subject_name):
                # Add face image to subject
                image_id = self.processor.add_face_to_subject(subject_name, image_path)
                if image_id:
                    self.enrolled_faces.add(face_id)
                    print(f"[INFO] Enrolled new face: {face_id} ({gender_str}, age {age_str})")
                    return True
            
            return False
            
        except Exception as e:
            print(f"[ERROR] Failed to enroll face {face_id}: {e}")
            return False
    
    def get_latest_fragment(self):
        """Get the latest fragment from the receiver's metadata."""
        try:
            if not os.path.exists(FRAGMENTS_METADATA):
                print(f"[WARN] Fragments metadata not found: {FRAGMENTS_METADATA}")
                return None
            
            with open(FRAGMENTS_METADATA, 'r') as f:
                data = json.load(f)
            
            latest = data.get('latest')
            if latest:
                source_path = os.path.join(CACHE_DIR, latest.get('filename', ''))
                if os.path.exists(source_path):
                    return latest
            
            return None
            
        except Exception as e:
            print(f"[ERROR] Could not read fragments metadata: {e}")
            return None
    
    def process_video(self, video_path, source_name=None):
        """
        Process a video file for facial recognition.
        
        Args:
            video_path: Path to the video file
            source_name: Optional name for the source (used in logging)
            
        Returns:
            Dictionary with processing results
        """
        if not os.path.exists(video_path):
            print(f"[ERROR] Video file not found: {video_path}")
            return None
        
        source_name = source_name or os.path.basename(video_path)
        print(f"\n[INFO] Processing video: {source_name}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 15
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[INFO] Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        print(f"[INFO] Sampling every {self.sample_rate} frames")
        
        # Process frames
        frame_count = 0
        faces_detected = []
        current_faces = {
            'source_file': source_name,
            'processed_at': datetime.now().isoformat(),
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'duration_seconds': round(total_frames / fps, 2)
            },
            'faces': []
        }
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Sample frames
            if frame_count % self.sample_rate != 0:
                continue
            
            # Convert frame to JPEG bytes
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            image_bytes = buffer.tobytes()
            
            # Detect faces
            detected = self.processor.detect_faces_from_bytes(image_bytes)
            
            if detected:
                print(f"[INFO] Frame {frame_count}: Detected {len(detected)} face(s)")
                
                for face in detected:
                    # Extract face data
                    face_data = {
                        'frame': frame_count,
                        'timestamp': round(frame_count / fps, 3),
                        'source': source_name,
                        'box': face.get('box'),
                        'landmarks': face.get('landmarks'),
                        'age': face.get('age'),
                        'gender': face.get('gender'),
                        'subjects': face.get('subjects', [])
                    }
                    
                    # Add to database
                    face_id = self.face_db.add_or_update_face(face_data)
                    face_data['face_id'] = face_id
                    
                    # Auto-enroll new faces (only if not already recognized by CompreFace)
                    if self.auto_enroll:
                        subjects = face.get('subjects', [])
                        is_recognized = False
                        recognized_subject = None
                        
                        # Check if CompreFace recognized this face as an existing subject
                        for subject in subjects:
                            similarity = subject.get('similarity', 0)
                            subject_name = subject.get('subject', '')
                            if similarity >= RECOGNITION_THRESHOLD:
                                is_recognized = True
                                recognized_subject = subject_name
                                print(f"[INFO] Frame {frame_count}: Face recognized as '{subject_name}' ({similarity*100:.1f}% match)")
                                break
                        
                        box = face.get('box')
                        
                        if is_recognized and self.update_quality and box:
                            # Face is recognized - check if this is a better quality image
                            face_img = self._crop_face(frame, box)
                            if face_img is not None:
                                was_updated, new_score, old_score = self.face_manager.update_face_if_better(
                                    recognized_subject, face_img
                                )
                                if was_updated:
                                    print(f"[QUALITY] Updated '{recognized_subject}' with better image (score: {old_score} -> {new_score})")
                        
                        elif not is_recognized and box:
                            # New face - enroll it
                            image_path = self._save_face_image(frame, box, face_id)
                            if image_path:
                                # Calculate and store quality score
                                quality_score = self.quality_analyzer.calculate_quality_score(image_path)
                                self.face_db.update_face_image_quality(face_id, quality_score)
                                # Enroll in CompreFace
                                self._enroll_face(face_id, image_path, face_data)
                    
                    faces_detected.append(face_data)
                    current_faces['faces'].append(face_data)
            
            # Progress update
            if frame_count % (self.sample_rate * 10) == 0:
                elapsed = time.time() - start_time
                progress = (frame_count / total_frames) * 100
                print(f"[INFO] Progress: {frame_count}/{total_frames} ({progress:.1f}%)")
        
        cap.release()
        
        processing_time = time.time() - start_time
        
        # Summary
        current_faces['processing_time_seconds'] = round(processing_time, 2)
        current_faces['total_faces_detected'] = len(faces_detected)
        current_faces['unique_detections'] = len(set(f['face_id'] for f in faces_detected))
        
        # Aggregate face info
        face_summary = {}
        for face in faces_detected:
            face_id = face['face_id']
            if face_id not in face_summary:
                face_summary[face_id] = {
                    'face_id': face_id,
                    'detection_count': 0,
                    'first_frame': face['frame'],
                    'last_frame': face['frame'],
                    'age': face.get('age'),
                    'gender': face.get('gender'),
                    'subjects': face.get('subjects', [])
                }
            face_summary[face_id]['detection_count'] += 1
            face_summary[face_id]['last_frame'] = face['frame']
        
        current_faces['face_summary'] = list(face_summary.values())
        
        # Save results
        self.face_db.finalize(current_faces)
        
        print(f"\n[INFO] Processing complete!")
        print(f"[INFO] Time: {processing_time:.2f}s")
        print(f"[INFO] Faces detected: {len(faces_detected)}")
        print(f"[INFO] Unique faces: {len(face_summary)}")
        
        return current_faces
    
    def process_latest(self):
        """Process the latest video fragment."""
        fragment = self.get_latest_fragment()
        
        if not fragment:
            print("[WARN] No fragments to process")
            return None
        
        video_path = os.path.join(CACHE_DIR, fragment.get('filename', ''))
        return self.process_video(video_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Facial Recognition Video Processor using CompreFace',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 facial_recognition.py --file fragment.avi
  python3 facial_recognition.py --latest
  python3 facial_recognition.py --watch --interval 15
  python3 facial_recognition.py --domain http://192.168.1.100 --port 8000
  python3 facial_recognition.py --rename face_abc123 "John Doe"
  python3 facial_recognition.py --list-faces
  python3 facial_recognition.py --clear

Output files:
  face_cache/all_faces.json     - Cumulative database of all detected faces
  face_cache/current_faces.json - Results from the latest processing
        """
    )
    
    parser.add_argument('--file', type=str, help='Process a specific video file')
    parser.add_argument('--latest', action='store_true', help='Process the latest fragment')
    parser.add_argument('--watch', action='store_true', help='Continuously watch for new fragments')
    parser.add_argument('--interval', type=int, default=15, help='Watch interval in seconds (default: 15)')
    parser.add_argument('--domain', type=str, default=COMPREFACE_DOMAIN, help=f'CompreFace domain (default: {COMPREFACE_DOMAIN})')
    parser.add_argument('--port', type=str, default=COMPREFACE_PORT, help=f'CompreFace port (default: {COMPREFACE_PORT})')
    parser.add_argument('--api-key', type=str, default=COMPREFACE_API_KEY, help='CompreFace API key')
    parser.add_argument('--sample-rate', type=int, default=5, help='Process every Nth frame (default: 5)')
    parser.add_argument('--auto-enroll', action='store_true', help='Automatically enroll new faces as subjects in CompreFace')
    parser.add_argument('--clear', action='store_true', help='Clear all subjects from CompreFace and reset local JSON files')
    parser.add_argument('--rename', nargs=2, metavar=('FACE_ID', 'NEW_NAME'), help='Rename a face (syncs to CompreFace)')
    parser.add_argument('--list-faces', action='store_true', help='List all known faces')
    parser.add_argument('--no-quality-update', action='store_true', help='Disable automatic quality-based image updates')
    
    args = parser.parse_args()
    
    # Get sample rate from args
    sample_rate = args.sample_rate
    
    print(f"\n[INFO] CompreFace Facial Recognition Processor")
    print(f"[INFO] Server: {args.domain}:{args.port}")
    print(f"[INFO] Sample rate: every {sample_rate} frames")
    if args.auto_enroll:
        print(f"[INFO] Auto-enroll: ENABLED - new faces will be added to CompreFace")
    
    try:
        # Initialize processor
        cf_processor = CompreFaceProcessor(
            domain=args.domain,
            port=args.port,
            api_key=args.api_key
        )
        
        # Handle --clear flag
        if args.clear:
            print(f"\n[CLEAR] Clearing all data...")
            
            # Delete all subjects from CompreFace
            cf_processor.delete_all_subjects()
            
            # Clear local JSON files
            if os.path.exists(ALL_FACES_FILE):
                os.remove(ALL_FACES_FILE)
                print(f"[CLEAR] Deleted {ALL_FACES_FILE}")
            
            if os.path.exists(CURRENT_FACES_FILE):
                os.remove(CURRENT_FACES_FILE)
                print(f"[CLEAR] Deleted {CURRENT_FACES_FILE}")
            
            # Clear face images folder
            if os.path.exists(FACE_IMAGES_DIR):
                import shutil
                shutil.rmtree(FACE_IMAGES_DIR)
                os.makedirs(FACE_IMAGES_DIR, exist_ok=True)
                print(f"[CLEAR] Cleared {FACE_IMAGES_DIR}")
            
            print(f"[CLEAR] Done! All face data has been cleared.")
            return
        
        # Handle --list-faces
        if args.list_faces:
            face_manager = FaceManager(cf_processor, FaceDatabase())
            faces = face_manager.list_faces()
            
            if not faces:
                print("[INFO] No faces in database")
                return
            
            print(f"\n[INFO] Found {len(faces)} face(s) in database:\n")
            print(f"{'Face ID':<20} {'Name':<20} {'Gender':<10} {'Age':<10} {'Detections':<12} {'Quality':<10}")
            print("-" * 82)
            
            for face in faces:
                face_id = face['face_id'][:18] if len(face['face_id']) > 18 else face['face_id']
                name = (face['name'][:18] if face['name'] else '-')
                gender = face.get('gender', {})
                gender_str = gender.get('value', '-') if gender else '-'
                age = face.get('age', {})
                age_str = f"{age.get('low', '?')}-{age.get('high', '?')}" if age else '-'
                detections = face.get('detection_count', 0)
                quality = face.get('quality_score', 0)
                
                print(f"{face_id:<20} {name:<20} {gender_str:<10} {age_str:<10} {detections:<12} {quality:<10.1f}")
            
            print()
            return
        
        # Handle --rename
        if args.rename:
            old_name, new_name = args.rename
            face_manager = FaceManager(cf_processor, FaceDatabase())
            
            if face_manager.rename_face(old_name, new_name):
                print(f"\n[SUCCESS] Renamed '{old_name}' to '{new_name}'")
            else:
                print(f"\n[ERROR] Failed to rename '{old_name}'")
            return
        
        # Determine if quality updates are enabled
        update_quality = not args.no_quality_update
        
        processor = VideoFaceProcessor(
            compreface_processor=cf_processor, 
            sample_rate=sample_rate, 
            auto_enroll=args.auto_enroll,
            update_quality=update_quality
        )
        
        if update_quality and args.auto_enroll:
            print(f"[INFO] Quality updates: ENABLED - better images will replace existing ones")
        
        if args.file:
            # Process specific file - check cache folder if not found directly
            video_path = args.file
            if not os.path.exists(video_path):
                # Try in cache folder
                cache_path = os.path.join(CACHE_DIR, os.path.basename(args.file))
                if os.path.exists(cache_path):
                    video_path = cache_path
                    print(f"[INFO] Found file in cache folder: {cache_path}")
            
            result = processor.process_video(video_path)
            if result:
                print(f"\n[RESULT] Results saved to:")
                print(f"  - {ALL_FACES_FILE}")
                print(f"  - {CURRENT_FACES_FILE}")
        
        elif args.watch:
            # Watch mode
            print(f"[INFO] Watching for new fragments (interval: {args.interval}s)")
            print("[INFO] Press Ctrl+C to stop")
            
            processed_files = set()
            
            try:
                while True:
                    fragment = processor.get_latest_fragment()
                    if fragment:
                        filename = fragment.get('filename', '')
                        if filename and filename not in processed_files:
                            video_path = os.path.join(CACHE_DIR, filename)
                            result = processor.process_video(video_path)
                            if result:
                                processed_files.add(filename)
                        else:
                            print(f"[INFO] No new fragments. Waiting...")
                    else:
                        print(f"[INFO] No fragments available. Waiting...")
                    
                    time.sleep(args.interval)
                    
            except KeyboardInterrupt:
                print("\n[INFO] Stopped watching")
        
        else:
            # Process latest fragment by default
            result = processor.process_latest()
            if result:
                print(f"\n[RESULT] Results saved to:")
                print(f"  - {ALL_FACES_FILE}")
                print(f"  - {CURRENT_FACES_FILE}")
            else:
                print("[INFO] No fragments to process")
    
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
