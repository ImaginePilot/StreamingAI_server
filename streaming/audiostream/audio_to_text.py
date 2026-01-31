#!/usr/bin/env python3
"""
Audio Transcription with Dual-Model Pipeline for Maximum Accuracy
==================================================================

Uses two complementary tools:
1. Pyannote.audio - Speaker diarization with WeSpeaker embeddings 
   (WHO speaks WHEN + voice fingerprints for cross-session identification)
2. OpenAI Whisper - Speech-to-text transcription (WHAT was said)

This combination provides:
- Precise speaker boundaries from Pyannote's neural diarization
- Persistent speaker identification via Pyannote's built-in WeSpeaker embeddings
- High-quality transcription from Whisper

Note: Pyannote community-1 model includes WeSpeaker embeddings, eliminating
the need for a separate SpeechBrain model while maintaining speaker tracking.

Requirements:
    pip install pyannote.audio openai-whisper torch torchaudio numpy scipy
    
    Set HF_TOKEN environment variable with your HuggingFace token:
    export HF_TOKEN="hf_your_token_here"

Usage:
    # Process single file
    python audio_to_text.py --file cache/fragment_001.wav
    
    # Process all files in directory
    python audio_to_text.py --dir cache/
    
    # Watch directory for new files
    python audio_to_text.py --watch cache/
    
    # List known speakers
    python audio_to_text.py --list-speakers
    
    # Label a speaker
    python audio_to_text.py --label-speaker SPK_0001 "John Doe"
"""

import os
import sys
import json
import argparse
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration loaded from audio_env.json
_config = None

def get_config():
    """Load configuration from audio_env.json."""
    global _config
    if _config is None:
        config_file = Path(__file__).parent / "audio_env.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                _config = json.load(f)
            
            # Apply environment variables
            env_vars = _config.get("environment", {})
            for key, value in env_vars.items():
                current = os.environ.get(key, "")
                if value not in current:
                    os.environ[key] = f"{value}:{current}" if current else value
            
            print(f"[INFO] Loaded config from {config_file}")
        else:
            # Default config
            _config = {
                "huggingface_token": None,
                "models": {
                    "pyannote_diarization": "pyannote/speaker-diarization-community-1",
                    "pyannote_embedding": "pyannote/wespeaker-voxceleb-resnet34-LM",
                    "whisper_model": "base"
                },
                "paths": {
                    "voices_database": "transcripts/voices.json",
                    "conversation_history": "transcripts/conversation_history.json"
                },
                "settings": {
                    "speaker_similarity_threshold": 0.75
                }
            }
    return _config

# Lazy imports for heavy libraries
_whisper = None
_funasr_model = None
_pyannote_pipeline = None
_pyannote_embedding_model = None
_torch = None


def get_torch():
    """Lazy load torch."""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def get_funasr_model():
    """Lazy load FunASR model (supports Fun-ASR-Nano-2512 and SenseVoice models)."""
    global _funasr_model
    if _funasr_model is None:
        from funasr import AutoModel
        
        config = get_config()
        models_config = config.get("models", {})
        model_name = models_config.get("funasr_model", "iic/SenseVoiceSmall")
        hub = models_config.get("funasr_hub", "ms")  # hf for HuggingFace, ms for ModelScope
        
        print(f"[INFO] Loading FunASR model: {model_name} (hub: {hub})")
        
        # Check if it's a Fun-ASR model (requires special loading from HuggingFace)
        is_fun_asr = "Fun-ASR" in model_name or "FunAudioLLM" in model_name
        
        device = "cuda" if get_torch().cuda.is_available() else "cpu"
        
        if is_fun_asr:
            # Fun-ASR-Nano-2512 requires direct loading from HuggingFace with remote code
            # It uses a custom model.py from the repo
            try:
                from huggingface_hub import snapshot_download
                import importlib.util
                import sys
                
                # Download model files from HuggingFace
                print(f"[INFO] Downloading Fun-ASR model from HuggingFace...")
                model_path = snapshot_download(repo_id=model_name)
                
                # Load the custom model.py from the downloaded repo
                model_py_path = Path(model_path) / "model.py"
                if not model_py_path.exists():
                    raise FileNotFoundError(f"model.py not found in {model_path}")
                
                spec = importlib.util.spec_from_file_location("fun_asr_model", model_py_path)
                fun_asr_module = importlib.util.module_from_spec(spec)
                sys.modules["fun_asr_model"] = fun_asr_module
                spec.loader.exec_module(fun_asr_module)
                
                # Load model using FunASRNano class from model.py
                FunASRNano = fun_asr_module.FunASRNano
                model_obj, kwargs = FunASRNano.from_pretrained(model=model_name, device=device, hub=hub)
                model_obj.eval()
                
                # Wrap in a simple object that has generate() method
                class FunASRWrapper:
                    def __init__(self, model, kwargs, language):
                        self.model = model
                        self.kwargs = kwargs
                        self.language = language
                        
                    def generate(self, input, **gen_kwargs):
                        # Use the inference method from FunASRNano
                        lang = gen_kwargs.get("language", self.language)
                        results = self.model.inference(
                            data_in=[input] if isinstance(input, str) else input,
                            language=lang,
                            **self.kwargs
                        )
                        # Format results to match FunASR output format
                        return [{"text": r[0].get("text", "") if r else ""} for r in results]
                
                language = models_config.get("funasr_language", "中文")
                _funasr_model = FunASRWrapper(model_obj, kwargs, language)
                _funasr_model._is_fun_asr = True
                print("[INFO] Fun-ASR model loaded successfully")
                
            except Exception as e:
                print(f"[WARN] Failed to load Fun-ASR model: {e}")
                print("[INFO] Falling back to SenseVoiceSmall...")
                _funasr_model = AutoModel(
                    model="iic/SenseVoiceSmall",
                    vad_model="fsmn-vad",
                    punc_model="ct-punc",
                    device=device,
                    disable_update=True,
                )
                _funasr_model._is_fun_asr = False
        else:
            # Standard SenseVoice/Paraformer models
            _funasr_model = AutoModel(
                model=model_name,
                vad_model="fsmn-vad",
                punc_model="ct-punc",
                device=device,
                disable_update=True,
            )
            _funasr_model._is_fun_asr = False
        print("[INFO] FunASR model loaded")
    return _funasr_model


def get_whisper_model(model_size: str = None):
    """Lazy load Whisper model."""
    global _whisper
    if _whisper is None:
        import whisper
        
        # Use provided size or get from config
        if model_size is None:
            config = get_config()
            model_size = config.get("models", {}).get("whisper_model", "base")
        
        print(f"[INFO] Loading Whisper model: {model_size}")
        _whisper = whisper.load_model(model_size)
        print("[INFO] Whisper model loaded")
    return _whisper


def get_hf_token():
    """Get HuggingFace token from config file or environment variable."""
    # First try from config
    config = get_config()
    token = config.get("huggingface_token")
    if token:
        return token
    
    # Try legacy hf_token.txt file
    token_file = Path(__file__).parent / "hf_token.txt"
    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            return token
    
    # Fall back to environment variable
    return os.environ.get("HF_TOKEN")


def get_pyannote_pipeline():
    """Lazy load Pyannote diarization pipeline."""
    global _pyannote_pipeline
    if _pyannote_pipeline is None:
        from pyannote.audio import Pipeline
        
        config = get_config()
        hf_token = get_hf_token()
        if not hf_token:
            print("[ERROR] HuggingFace token not found!")
            print("       Option 1: Add 'huggingface_token' to audio_env.json")
            print("       Option 2: Create hf_token.txt with your token")
            print("       Option 3: export HF_TOKEN='hf_your_token_here'")
            print("       Get token from: https://huggingface.co/settings/tokens")
            sys.exit(1)
        
        model_name = config.get("models", {}).get(
            "pyannote_diarization", 
            "pyannote/speaker-diarization-community-1"
        )
        
        print(f"[INFO] Loading Pyannote diarization pipeline: {model_name}")
        _pyannote_pipeline = Pipeline.from_pretrained(
            model_name,
            token=hf_token
        )
        
        # Apply hyperparameters from config
        # Note: community-1 uses powerset mode + VBx clustering
        # Available params: segmentation.min_duration_off, clustering.{threshold, Fa, Fb}
        pyannote_config = config.get("pyannote", {})
        hyperparams = {}
        
        # Segmentation parameters (only min_duration_off for powerset models)
        seg_params = {}
        if pyannote_config.get("segmentation_min_duration_off") is not None:
            seg_params["min_duration_off"] = pyannote_config["segmentation_min_duration_off"]
        if seg_params:
            hyperparams["segmentation"] = seg_params
        
        # Clustering parameters (VBx: threshold, Fa, Fb)
        clust_params = {}
        if pyannote_config.get("clustering_threshold") is not None:
            clust_params["threshold"] = pyannote_config["clustering_threshold"]
        if pyannote_config.get("clustering_Fa") is not None:
            clust_params["Fa"] = pyannote_config["clustering_Fa"]
        if pyannote_config.get("clustering_Fb") is not None:
            clust_params["Fb"] = pyannote_config["clustering_Fb"]
        if clust_params:
            hyperparams["clustering"] = clust_params
        
        if hyperparams:
            print(f"[INFO] Applying pyannote hyperparameters: {hyperparams}")
            _pyannote_pipeline.instantiate(hyperparams)
        
        # Move to GPU if available
        torch = get_torch()
        if torch.cuda.is_available():
            _pyannote_pipeline.to(torch.device("cuda"))
            print("[INFO] Pyannote pipeline moved to GPU")
        else:
            print("[INFO] Pyannote pipeline running on CPU")
        
        print("[INFO] Pyannote pipeline loaded")
    return _pyannote_pipeline


def get_pyannote_embedding_model():
    """Lazy load Pyannote embedding model (WeSpeaker-based)."""
    global _pyannote_embedding_model
    if _pyannote_embedding_model is None:
        from pyannote.audio import Model
        
        config = get_config()
        hf_token = get_hf_token()
        
        model_name = config.get("models", {}).get(
            "pyannote_embedding",
            "pyannote/wespeaker-voxceleb-resnet34-LM"
        )
        
        print(f"[INFO] Loading Pyannote embedding model: {model_name}")
        _pyannote_embedding_model = Model.from_pretrained(
            model_name,
            token=hf_token
        )
        
        # Move to GPU if available
        torch = get_torch()
        if torch.cuda.is_available():
            _pyannote_embedding_model.to(torch.device("cuda"))
            print("[INFO] Embedding model moved to GPU")
        else:
            print("[INFO] Embedding model running on CPU")
        
        print("[INFO] Pyannote embedding model loaded")
    return _pyannote_embedding_model


@dataclass
class SpeakerInfo:
    """Information about a known speaker."""
    speaker_id: str
    label: Optional[str]
    embedding: List[float]
    first_seen: str
    last_seen: str
    sample_count: int
    

@dataclass
class TranscriptionSegment:
    """A transcribed segment with speaker info."""
    start_time: float
    end_time: float
    speaker_id: str
    speaker_label: Optional[str]
    text: str
    confidence: float
    language: Optional[str] = None
    emotion: Optional[str] = None
    audio_type: Optional[str] = None
    

@dataclass
class TranscriptionResult:
    """Full transcription result for an audio file."""
    file_path: str
    file_hash: str
    timestamp: str
    duration: float
    segments: List[TranscriptionSegment]
    speakers_detected: List[str]
    detected_language: Optional[str] = None
    asr_backend: Optional[str] = None


class VoicesDatabase:
    """
    Persistent database of known speaker voice embeddings.
    Uses WeSpeaker embeddings for cross-session speaker identification.
    """
    
    def __init__(self, db_path: str = None):
        config = get_config()
        if db_path is None:
            db_path = config.get("paths", {}).get(
                "voices_database", 
                "transcripts/voices.json"
            )
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.speakers: Dict[str, SpeakerInfo] = {}
        self.next_speaker_num = 1
        self.similarity_threshold = config.get("settings", {}).get(
            "speaker_similarity_threshold", 0.75
        )
        self._load()
    
    def _load(self):
        """Load speakers from disk."""
        if self.db_path.exists():
            try:
                raw = self.db_path.read_text().strip()
                if not raw:
                    raise ValueError("empty file")
                data = json.loads(raw)
                
                for sid, info in data.get("speakers", {}).items():
                    self.speakers[sid] = SpeakerInfo(**info)
                
                self.next_speaker_num = data.get("next_speaker_num", 1)
                self.similarity_threshold = data.get("similarity_threshold", 0.75)
                
                print(f"[INFO] Loaded {len(self.speakers)} known speakers from database")
            except Exception as e:
                print(f"[WARN] Failed to load voices database: {e}. Resetting database file.")
                self.speakers = {}
                self.next_speaker_num = 1
                self.similarity_threshold = get_config().get("settings", {}).get(
                    "speaker_similarity_threshold", 0.75
                )
                self._save()
    
    def _save(self):
        """Save speakers to disk."""
        data = {
            "speakers": {sid: asdict(info) for sid, info in self.speakers.items()},
            "next_speaker_num": self.next_speaker_num,
            "similarity_threshold": self.similarity_threshold,
            "updated": datetime.now().isoformat()
        }
        
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        emb1 = np.array(emb1).flatten()
        emb2 = np.array(emb2).flatten()
        dot = np.dot(emb1, emb2)
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        if norm == 0:
            return 0.0
        return float(dot / norm)
    
    def find_or_create_speaker(self, embedding: np.ndarray) -> Tuple[str, Optional[str], bool]:
        """
        Find matching speaker or create new one.
        
        Returns:
            Tuple of (speaker_id, speaker_label, is_new)
        """
        embedding_list = embedding.flatten().tolist()
        now = datetime.now().isoformat()
        
        # Find best matching speaker
        best_match = None
        best_similarity = 0.0
        
        for sid, info in self.speakers.items():
            similarity = self._cosine_similarity(embedding, np.array(info.embedding))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = sid
        
        # Check if similarity exceeds threshold
        if best_match and best_similarity >= self.similarity_threshold:
            # Update existing speaker
            speaker = self.speakers[best_match]
            
            # Update embedding with weighted average (favor newer samples slightly)
            old_emb = np.array(speaker.embedding)
            weight = min(0.2, 1.0 / (speaker.sample_count + 1))
            new_emb = (1 - weight) * old_emb + weight * embedding
            speaker.embedding = new_emb.flatten().tolist()
            
            speaker.last_seen = now
            speaker.sample_count += 1
            self._save()
            
            print(f"[MATCH] Speaker {best_match} (similarity: {best_similarity:.3f})")
            return best_match, speaker.label, False
        
        # Create new speaker
        speaker_id = f"SPK_{self.next_speaker_num:04d}"
        self.next_speaker_num += 1
        
        self.speakers[speaker_id] = SpeakerInfo(
            speaker_id=speaker_id,
            label=None,
            embedding=embedding_list,
            first_seen=now,
            last_seen=now,
            sample_count=1
        )
        self._save()
        
        print(f"[NEW] Created speaker {speaker_id}")
        return speaker_id, None, True
    
    def label_speaker(self, speaker_id: str, label: str) -> bool:
        """Assign a human-readable label to a speaker."""
        if speaker_id in self.speakers:
            self.speakers[speaker_id].label = label
            self._save()
            print(f"[INFO] Labeled {speaker_id} as '{label}'")
            return True
        return False
    
    def list_speakers(self) -> List[Dict[str, Any]]:
        """List all known speakers."""
        return [
            {
                "id": info.speaker_id,
                "label": info.label or "(unlabeled)",
                "first_seen": info.first_seen,
                "last_seen": info.last_seen,
                "samples": info.sample_count
            }
            for info in self.speakers.values()
        ]


class ConversationHistory:
    """
    Persistent storage for transcription history.
    Maintains a searchable log of all transcriptions.
    """
    
    def __init__(self, history_path: str = None):
        config = get_config()
        if history_path is None:
            history_path = config.get("paths", {}).get(
                "conversation_history",
                "transcripts/conversation_history.json"
            )
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.transcriptions: List[Dict] = []
        self.processed_hashes: set = set()
        self._load()
    
    def _load(self):
        """Load history from disk."""
        if self.history_path.exists():
            try:
                raw = self.history_path.read_text().strip()
                if not raw:
                    raise ValueError("empty file")
                data = json.loads(raw)
                
                self.transcriptions = data.get("transcriptions", [])
                self.processed_hashes = set(data.get("processed_hashes", []))
                
                print(f"[INFO] Loaded {len(self.transcriptions)} transcription records")
            except Exception as e:
                print(f"[WARN] Failed to load history: {e}. Resetting history file.")
                self.transcriptions = []
                self.processed_hashes = set()
                self._save()
    
    def _save(self):
        """Save history to disk."""
        data = {
            "transcriptions": self.transcriptions,
            "processed_hashes": list(self.processed_hashes),
            "updated": datetime.now().isoformat()
        }
        
        with open(self.history_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def is_processed(self, file_hash: str) -> bool:
        """Check if file was already processed."""
        return file_hash in self.processed_hashes
    
    def add_transcription(self, result: TranscriptionResult):
        """Add a transcription result to history."""
        record = {
            "file": result.file_path,
            "hash": result.file_hash,
            "timestamp": result.timestamp,
            "duration": result.duration,
            "speakers": result.speakers_detected,
            "segments": [asdict(seg) for seg in result.segments]
        }
        
        self.transcriptions.append(record)
        self.processed_hashes.add(result.file_hash)
        self._save()
    
    def get_recent(self, count: int = 10) -> List[Dict]:
        """Get most recent transcriptions."""
        return self.transcriptions[-count:]


class AudioTranscriber:
    """
    Dual-model audio transcriber combining:
    - Pyannote for speaker diarization + WeSpeaker embeddings
    - VoiceMemory for persistent cross-session speaker identification
    - Whisper/FunASR for transcription
    """
    
    def __init__(
        self,
        whisper_model: str = "base",
        voices_db_path: str = "transcripts/voices.json",
        history_path: str = "transcripts/conversation_history.json",
        use_voice_memory: bool = True
    ):
        self.whisper_model = whisper_model
        self.use_voice_memory = use_voice_memory
        
        # Use new VoiceMemory module if available, fallback to VoicesDatabase
        if use_voice_memory:
            try:
                from audiostream.voice_memory import VoiceMemory
                self.voice_memory = VoiceMemory(db_path=voices_db_path)
                self.voices_db = None  # Not used when VoiceMemory is active
                print("[INFO] Using VoiceMemory for cross-session speaker identification")
            except ImportError:
                print("[WARN] VoiceMemory not available, using legacy VoicesDatabase")
                self.voice_memory = None
                self.voices_db = VoicesDatabase(voices_db_path)
        else:
            self.voice_memory = None
            self.voices_db = VoicesDatabase(voices_db_path)
        
        self.history = ConversationHistory(history_path)
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute MD5 hash of file for deduplication."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _extract_audio_segment(
        self, 
        audio_path: Path, 
        start_time: float, 
        end_time: float
    ) -> np.ndarray:
        """Extract audio segment for embedding computation."""
        import torchaudio
        
        waveform, sample_rate = torchaudio.load(str(audio_path))
        
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Ensure valid bounds
        start_sample = max(0, start_sample)
        end_sample = min(waveform.shape[1], end_sample)
        
        segment = waveform[:, start_sample:end_sample]
        
        # Convert to mono if stereo
        if segment.shape[0] > 1:
            segment = segment.mean(dim=0, keepdim=True)
        
        return segment
    
    def _get_speaker_embedding(self, audio_segment, sample_rate: int = 16000) -> np.ndarray:
        """Compute Pyannote/WeSpeaker embedding for audio segment."""
        from pyannote.audio import Inference
        
        embedding_model = get_pyannote_embedding_model()
        
        # Create inference object
        inference = Inference(embedding_model, window="whole")
        
        # Ensure correct shape and convert to numpy for Inference
        if hasattr(audio_segment, 'numpy'):
            audio_np = audio_segment.numpy()
        else:
            audio_np = np.array(audio_segment)
        
        # Ensure shape is (channels, samples)
        if audio_np.ndim == 1:
            audio_np = audio_np.reshape(1, -1)
        elif audio_np.ndim == 3:
            audio_np = audio_np.squeeze(0)
        
        # Create a dict that Inference expects: {"waveform": tensor, "sample_rate": int}
        torch = get_torch()
        audio_dict = {
            "waveform": torch.from_numpy(audio_np).float(),
            "sample_rate": sample_rate
        }
        
        # Compute embedding
        embedding = inference(audio_dict)
        return embedding.flatten()
    
    def process_file(self, audio_path: str, force: bool = False) -> Optional[TranscriptionResult]:
        """
        Process an audio file with the triple-model pipeline.
        
        Args:
            audio_path: Path to audio file (WAV format recommended)
            force: Force reprocessing even if already processed
            
        Returns:
            TranscriptionResult or None if already processed
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            print(f"[ERROR] File not found: {audio_path}")
            return None
        
        # Check if already processed
        file_hash = self._compute_file_hash(audio_path)
        if not force and self.history.is_processed(file_hash):
            print(f"[SKIP] Already processed: {audio_path.name}")
            return None
        
        print(f"\n{'='*60}")
        print(f"Processing: {audio_path.name}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Run Pyannote diarization
            print("\n[1/3] Running Pyannote speaker diarization...")
            pipeline = get_pyannote_pipeline()
            
            # Get pyannote parameters from config
            config = get_config()
            pyannote_config = config.get("pyannote", {})
            
            # Build pipeline kwargs
            pipeline_kwargs = {}
            if pyannote_config.get("num_speakers") is not None:
                pipeline_kwargs["num_speakers"] = pyannote_config["num_speakers"]
            if pyannote_config.get("min_speakers") is not None:
                pipeline_kwargs["min_speakers"] = pyannote_config["min_speakers"]
            if pyannote_config.get("max_speakers") is not None:
                pipeline_kwargs["max_speakers"] = pyannote_config["max_speakers"]
            
            if pipeline_kwargs:
                print(f"    Pyannote params: {pipeline_kwargs}")
            
            output = pipeline(str(audio_path), **pipeline_kwargs)
            
            # Extract speaker segments from diarization
            # DiarizeOutput has .speaker_diarization (Annotation) which has .itertracks()
            diarization_segments = []
            annotation = output.speaker_diarization
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                diarization_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "pyannote_speaker": speaker  # Temporary ID from Pyannote
                })
            
            print(f"    Found {len(diarization_segments)} speech segments")
            
            if not diarization_segments:
                print("[WARN] No speech detected in audio")
                return TranscriptionResult(
                    file_path=str(audio_path),
                    file_hash=file_hash,
                    timestamp=datetime.now().isoformat(),
                    duration=0.0,
                    segments=[],
                    speakers_detected=[]
                )
            
            # Step 2: Compute WeSpeaker embeddings for each unique Pyannote speaker
            print("\n[2/3] Computing speaker embeddings for cross-session identification...")
            
            # Group segments by Pyannote speaker to get one embedding per speaker
            pyannote_to_segments = {}
            for seg in diarization_segments:
                ps = seg["pyannote_speaker"]
                if ps not in pyannote_to_segments:
                    pyannote_to_segments[ps] = []
                pyannote_to_segments[ps].append(seg)
            
            print(f"    Found {len(pyannote_to_segments)} unique speakers in this segment")
            
            # Map Pyannote speakers to persistent speaker IDs
            pyannote_to_persistent = {}
            for pyannote_speaker, segments in pyannote_to_segments.items():
                # Use the longest segment for embedding (more reliable)
                longest_seg = max(segments, key=lambda s: s["end"] - s["start"])
                seg_duration = longest_seg["end"] - longest_seg["start"]
                
                # Extract audio and compute embedding
                audio_segment = self._extract_audio_segment(
                    audio_path, 
                    longest_seg["start"], 
                    longest_seg["end"]
                )
                
                # Need minimum audio length for reliable embedding
                if audio_segment.shape[-1] < 16000 * 0.5:  # Less than 0.5 seconds
                    print(f"    [WARN] Segment too short for {pyannote_speaker} ({seg_duration:.2f}s), skipping embedding")
                    # Create temporary speaker ID
                    temp_id = f"TEMP_{pyannote_speaker}"
                    pyannote_to_persistent[pyannote_speaker] = (temp_id, None, 0.0)
                    continue
                
                # Use VoiceMemory if available, otherwise fallback to VoicesDatabase
                if self.voice_memory is not None:
                    # New VoiceMemory module - better cross-session identification
                    result = self.voice_memory.identify_speaker(
                        audio_segment.numpy() if hasattr(audio_segment, 'numpy') else audio_segment,
                        sample_rate=16000,
                        duration=seg_duration
                    )
                    speaker_id = result.speaker_id
                    speaker_label = result.label
                    confidence = result.confidence
                    is_new = result.is_new
                    print(f"    {pyannote_speaker} -> {speaker_id} ({speaker_label or 'unlabeled'}) conf={confidence:.3f} {'[NEW]' if is_new else ''}")
                    pyannote_to_persistent[pyannote_speaker] = (speaker_id, speaker_label, confidence)
                else:
                    # Legacy VoicesDatabase
                    embedding = self._get_speaker_embedding(audio_segment)
                    speaker_id, speaker_label, is_new = self.voices_db.find_or_create_speaker(embedding)
                    pyannote_to_persistent[pyannote_speaker] = (speaker_id, speaker_label, 0.0)
            
            # Step 3: Run ASR transcription
            config = get_config()
            asr_backend = config.get("models", {}).get("asr_backend", "funasr")
            
            if asr_backend == "funasr":
                print("\n[3/3] Running FunASR transcription...")
                funasr_model = get_funasr_model()
                
                # Check if using Fun-ASR model (needs language parameter)
                models_config = config.get("models", {})
                funasr_model_name = models_config.get("funasr_model", "")
                is_fun_asr = "Fun-ASR" in funasr_model_name or "FunAudioLLM" in funasr_model_name
                funasr_language = models_config.get("funasr_language", "中文")
                
                # Helper to parse FunASR outputs into timestamped words
                def _parse_funasr_items(items, default_window=(0.0, 999999.0)):
                    segments = []
                    if not items:
                        return segments
                    import re
                    for item in items:
                        text = item.get("text", "") or ""
                        timestamps = item.get("timestamp", []) or []
                        meta = {"language": None, "emotion": None, "audio_type": None}

                        if not text:
                            continue

                        normalized_text = re.sub(r'<\s*\|\s*', '<|', text)
                        normalized_text = re.sub(r'\s*\|\s*>', '|>', normalized_text)

                        metadata_tags = re.findall(r'<\|([^|]+)\|>', normalized_text)
                        for tag in metadata_tags:
                            tag_lower = tag.strip().lower()
                            if tag_lower in ['zh', 'en', 'ja', 'ko', 'yue']:
                                meta["language"] = tag_lower
                            elif tag_lower in ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']:
                                meta["emotion"] = tag_lower
                            elif tag_lower in ['speech', 'music', 'applause', 'laughter']:
                                meta["audio_type"] = tag_lower

                        clean_text = re.sub(r'<\s*\|\s*[^|]+\s*\|\s*>', '', text).strip()
                        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

                        if meta["language"]:
                            print(f"    Detected: language={meta['language']}, emotion={meta['emotion']}, type={meta['audio_type']}")

                        if timestamps:
                            words = clean_text.split()
                            for i, ts in enumerate(timestamps):
                                if i < len(words):
                                    segments.append({
                                        "start": ts[0] / 1000.0,
                                        "end": ts[1] / 1000.0,
                                        "word": words[i],
                                        "confidence": 0.9,
                                        "metadata": meta.copy()
                                    })
                        else:
                            start_default, end_default = default_window
                            segments.append({
                                "start": start_default,
                                "end": end_default,
                                "word": clean_text,
                                "confidence": 0.9,
                                "metadata": meta.copy()
                            })
                    return segments

                # For Fun-ASR-Nano, skip full-file ASR and go directly to per-segment processing
                # This is faster since Fun-ASR doesn't return timestamps anyway
                is_fun_asr_model = getattr(funasr_model, '_is_fun_asr', False)
                
                if is_fun_asr_model and diarization_segments:
                    # Fun-ASR: Direct per-segment processing with parallelization (skip full file)
                    import tempfile
                    import torchaudio
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    import multiprocessing
                    
                    # Use half of available CPUs for parallel ASR (leave some for system)
                    num_workers = max(1, min(len(diarization_segments), multiprocessing.cpu_count() // 2))
                    print(f"    Processing {len(diarization_segments)} segments in parallel ({num_workers} workers)...")
                    
                    waveform, sample_rate = torchaudio.load(str(audio_path))
                    
                    def process_segment(args):
                        """Process a single segment - runs in thread."""
                        idx, dia_seg = args
                        start_sample = int(dia_seg["start"] * sample_rate)
                        end_sample = int(dia_seg["end"] * sample_rate)
                        start_sample = max(0, start_sample)
                        end_sample = min(waveform.shape[1], end_sample)
                        
                        if end_sample <= start_sample:
                            return idx, []
                        
                        segment_wave = waveform[:, start_sample:end_sample]
                        tmp_path = None
                        try:
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                                tmp_path = tmp_wav.name
                                torchaudio.save(tmp_path, segment_wave, sample_rate)
                            
                            seg_result = funasr_model.generate(
                                input=tmp_path,
                                cache={},
                                batch_size=1,
                                language=funasr_language,
                                itn=True,
                            )
                            return idx, _parse_funasr_items(
                                seg_result,
                                default_window=(dia_seg["start"], dia_seg["end"])
                            )
                        except Exception as e:
                            print(f"    [WARN] Segment {idx} failed: {e}")
                            return idx, []
                        finally:
                            if tmp_path and Path(tmp_path).exists():
                                try:
                                    os.unlink(tmp_path)
                                except Exception:
                                    pass
                    
                    # Process segments in parallel using ThreadPoolExecutor
                    segment_results = {}
                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        futures = {
                            executor.submit(process_segment, (idx, seg)): idx 
                            for idx, seg in enumerate(diarization_segments)
                        }
                        for future in as_completed(futures):
                            idx, result = future.result()
                            segment_results[idx] = result
                    
                    # Combine results in order
                    asr_segments = []
                    for idx in sorted(segment_results.keys()):
                        asr_segments.extend(segment_results[idx])
                    
                else:
                    # Standard path: Run full file ASR first
                    if is_fun_asr:
                        result = funasr_model.generate(
                            input=str(audio_path),
                            cache={},
                            batch_size=1,
                            language=funasr_language,
                            itn=True,
                        )
                    else:
                        result = funasr_model.generate(
                            input=str(audio_path),
                            batch_size_s=300,
                        )

                    asr_segments = _parse_funasr_items(result)

                    # If no timestamps were returned, re-run ASR on each diarization segment
                    if asr_segments and all(seg.get("end", 0) >= 999000 for seg in asr_segments):
                        print("    No ASR timestamps returned; re-running ASR per diarization segment for alignment...")
                        import tempfile
                        import torchaudio

                        waveform, sample_rate = torchaudio.load(str(audio_path))
                        aligned_segments = []

                        for idx, dia_seg in enumerate(diarization_segments):
                            start_sample = int(dia_seg["start"] * sample_rate)
                            end_sample = int(dia_seg["end"] * sample_rate)
                            start_sample = max(0, start_sample)
                            end_sample = min(waveform.shape[1], end_sample)

                            if end_sample <= start_sample:
                                continue

                            segment_wave = waveform[:, start_sample:end_sample]

                            tmp_path = None
                            try:
                                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                                    tmp_path = tmp_wav.name
                                    torchaudio.save(tmp_path, segment_wave, sample_rate)

                                if is_fun_asr:
                                    seg_result = funasr_model.generate(
                                        input=tmp_path,
                                        cache={},
                                        batch_size=1,
                                        language=funasr_language,
                                        itn=True,
                                    )
                                else:
                                    seg_result = funasr_model.generate(input=tmp_path, batch_size_s=60)
                                aligned_segments.extend(
                                    _parse_funasr_items(
                                        seg_result,
                                        default_window=(dia_seg["start"], dia_seg["end"])
                                    )
                                )
                            finally:
                                if tmp_path and Path(tmp_path).exists():
                                    try:
                                        os.unlink(tmp_path)
                                    except Exception:
                                        pass

                        if aligned_segments:
                            asr_segments = aligned_segments

                print(f"    Transcribed: {len(asr_segments)} segments")
                
            else:
                # Whisper backend
                print("\n[3/3] Running Whisper transcription...")
                whisper_model = get_whisper_model(self.whisper_model)
                
                # Get language settings from config
                whisper_config = config.get("whisper", {})
                language = whisper_config.get("language")  # None = auto-detect
                task = whisper_config.get("task", "transcribe")  # "transcribe" or "translate"
                
                # Build transcription options
                transcribe_opts = {
                    "word_timestamps": True,
                    "task": task
                }
                if language:
                    transcribe_opts["language"] = language
                    print(f"    Language: {language}, Task: {task}")
                else:
                    print(f"    Language: auto-detect, Task: {task}")
                
                # Transcribe full audio
                whisper_result = whisper_model.transcribe(str(audio_path), **transcribe_opts)
                
                # Show detected language if auto-detected
                if not language and "language" in whisper_result:
                    print(f"    Detected language: {whisper_result['language']}")
                
                # Convert Whisper result to common format
                asr_segments = []
                for seg in whisper_result.get("segments", []):
                    for word_info in seg.get("words", []):
                        asr_segments.append({
                            "start": word_info.get("start", 0),
                            "end": word_info.get("end", 0),
                            "word": word_info.get("word", "").strip(),
                            "confidence": word_info.get("probability", 0.5)
                        })
            
            # Build final segments combining diarization with transcription
            print("\n[COMBINING] Merging diarization with transcription...")
            
            final_segments = []
            speakers_seen = set()
            
            for dia_seg in diarization_segments:
                seg_start = dia_seg["start"]
                seg_end = dia_seg["end"]
                pyannote_speaker = dia_seg["pyannote_speaker"]
                
                # Get persistent speaker ID (tuple: speaker_id, speaker_label, confidence)
                speaker_info = pyannote_to_persistent.get(
                    pyannote_speaker, 
                    (f"UNKNOWN_{pyannote_speaker}", None, 0.0)
                )
                speaker_id = speaker_info[0]
                speaker_label = speaker_info[1]
                # speaker_confidence = speaker_info[2]  # Available if needed
                speakers_seen.add(speaker_id)
                
                # Find ASR words that fall within this segment
                segment_text = []
                segment_confidence = []
                segment_metadata = {"language": None, "emotion": None, "audio_type": None}
                
                for word_info in asr_segments:
                    word_start = word_info.get("start", 0)
                    word_end = word_info.get("end", 0)
                    
                    # Check if word overlaps with diarization segment
                    overlap_start = max(seg_start, word_start)
                    overlap_end = min(seg_end, word_end)
                    
                    if overlap_end > overlap_start:
                        word = word_info.get("word", "").strip()
                        if word:
                            segment_text.append(word)
                            segment_confidence.append(word_info.get("confidence", 0.5))
                            # Extract metadata from FunASR segments
                            if "metadata" in word_info:
                                meta = word_info["metadata"]
                                if meta.get("language"):
                                    segment_metadata["language"] = meta["language"]
                                if meta.get("emotion"):
                                    segment_metadata["emotion"] = meta["emotion"]
                                if meta.get("audio_type"):
                                    segment_metadata["audio_type"] = meta["audio_type"]
                
                text = " ".join(segment_text)
                confidence = np.mean(segment_confidence) if segment_confidence else 0.5
                
                if text:  # Only add non-empty segments
                    final_segments.append(TranscriptionSegment(
                        start_time=seg_start,
                        end_time=seg_end,
                        speaker_id=speaker_id,
                        speaker_label=speaker_label,
                        text=text,
                        confidence=float(confidence),
                        language=segment_metadata["language"],
                        emotion=segment_metadata["emotion"],
                        audio_type=segment_metadata["audio_type"]
                    ))
            
            # Get audio duration
            import torchaudio
            waveform, sample_rate = torchaudio.load(str(audio_path))
            duration = waveform.shape[1] / sample_rate
            
            # Create result
            result_obj = TranscriptionResult(
                file_path=str(audio_path),
                file_hash=file_hash,
                timestamp=datetime.now().isoformat(),
                duration=float(duration),
                segments=final_segments,
                speakers_detected=list(speakers_seen)
            )
            
            # Save to history
            self.history.add_transcription(result_obj)
            
            # Print results
            print(f"\n{'='*60}")
            print("TRANSCRIPTION RESULTS")
            print(f"{'='*60}")
            print(f"Duration: {duration:.2f}s")
            print(f"Speakers: {', '.join(speakers_seen)}")
            print(f"{'='*60}\n")
            
            for seg in final_segments:
                speaker_display = seg.speaker_label or seg.speaker_id
                print(f"[{seg.start_time:6.2f} - {seg.end_time:6.2f}] {speaker_display}:")
                # Show metadata if available
                meta_parts = []
                if seg.language:
                    meta_parts.append(f"lang={seg.language}")
                if seg.emotion:
                    meta_parts.append(f"emotion={seg.emotion}")
                if seg.audio_type:
                    meta_parts.append(f"type={seg.audio_type}")
                if meta_parts:
                    print(f"    [{', '.join(meta_parts)}]")
                print(f"    {seg.text}\n")
            
            return result_obj
            
        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_directory(self, dir_path: str, pattern: str = "*.wav"):
        """Process all matching audio files in directory."""
        dir_path = Path(dir_path)
        
        if not dir_path.exists():
            print(f"[ERROR] Directory not found: {dir_path}")
            return
        
        files = sorted(dir_path.glob(pattern))
        print(f"Found {len(files)} audio files to process")
        
        for audio_file in files:
            self.process_file(str(audio_file))
    
    def watch_directory(self, dir_path: str, pattern: str = "*.wav", interval: float = 5.0):
        """Watch directory for new audio files and process them."""
        dir_path = Path(dir_path)
        
        if not dir_path.exists():
            print(f"[ERROR] Directory not found: {dir_path}")
            return
        
        print(f"[WATCH] Monitoring {dir_path} for new {pattern} files...")
        print("        Press Ctrl+C to stop\n")
        
        processed_files = set()
        
        try:
            while True:
                files = set(dir_path.glob(pattern))
                new_files = files - processed_files
                
                for audio_file in sorted(new_files):
                    # Check if file is still being written
                    try:
                        initial_size = audio_file.stat().st_size
                        time.sleep(0.5)
                        if audio_file.stat().st_size != initial_size:
                            continue  # File still being written
                    except:
                        continue
                    
                    self.process_file(str(audio_file))
                    processed_files.add(audio_file)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n[WATCH] Stopped monitoring")


def main():
    parser = argparse.ArgumentParser(
        description="Dual-Model Audio Transcription (Pyannote + Whisper with WeSpeaker embeddings)"
    )
    
    parser.add_argument("--file", "-f", help="Process single audio file")
    parser.add_argument("--dir", "-d", help="Process all audio files in directory")
    parser.add_argument("--watch", "-w", help="Watch directory for new files")
    parser.add_argument("--pattern", "-p", default="*.wav", help="File pattern (default: *.wav)")
    parser.add_argument("--model", "-m", default="base", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--force", action="store_true", help="Force reprocess already-processed files")
    parser.add_argument("--list-speakers", action="store_true", help="List known speakers")
    parser.add_argument("--label-speaker", nargs=2, metavar=("ID", "NAME"),
                        help="Label a speaker (e.g., --label-speaker SPK_0001 'John Doe')")
    parser.add_argument("--voices-db", default="transcripts/voices.json",
                        help="Path to voices database")
    parser.add_argument("--history", default="transcripts/conversation_history.json",
                        help="Path to conversation history")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="Force diarization to exactly this many speakers")
    parser.add_argument("--min-speakers", type=int, default=None,
                        help="Set minimum expected speakers (helps split close voices)")
    parser.add_argument("--max-speakers", type=int, default=None,
                        help="Set maximum expected speakers")
    parser.add_argument("--clustering-threshold", type=float, default=None,
                        help="Pyannote VBx threshold (lower = more speakers, higher = fewer)")
    parser.add_argument("--preload-models", action="store_true",
                        help="Download/load diarization, embedding, and ASR models then exit (warm start)")
    
    args = parser.parse_args()
    
    # Handle speaker management commands
    if args.list_speakers:
        db = VoicesDatabase(args.voices_db)
        speakers = db.list_speakers()
        
        if not speakers:
            print("No known speakers in database")
            return
        
        print(f"\nKnown Speakers ({len(speakers)}):")
        print("-" * 60)
        for s in speakers:
            print(f"  {s['id']}: {s['label']}")
            print(f"      First seen: {s['first_seen']}")
            print(f"      Last seen:  {s['last_seen']}")
            print(f"      Samples:    {s['samples']}")
            print()
        return
    
    if args.label_speaker:
        db = VoicesDatabase(args.voices_db)
        speaker_id, name = args.label_speaker
        if db.label_speaker(speaker_id, name):
            print(f"Successfully labeled {speaker_id} as '{name}'")
        else:
            print(f"Speaker {speaker_id} not found")
        return

    # Apply diarization overrides before any model loads
    config = get_config()
    py_cfg = config.setdefault("pyannote", {})
    if args.num_speakers is not None:
        py_cfg["num_speakers"] = args.num_speakers
    if args.min_speakers is not None:
        py_cfg["min_speakers"] = args.min_speakers
    if args.max_speakers is not None:
        py_cfg["max_speakers"] = args.max_speakers
    if args.clustering_threshold is not None:
        py_cfg["clustering_threshold"] = args.clustering_threshold
    
    # Check HF_TOKEN
    if not get_hf_token():
        print("[ERROR] HuggingFace token not found!")
        print("        Option 1: Create hf_token.txt with your token")
        print("        Option 2: export HF_TOKEN='hf_your_token_here'")
        sys.exit(1)

    if args.preload_models:
        print("[PRELOAD] Loading Pyannote diarization pipeline...")
        get_pyannote_pipeline()
        print("[PRELOAD] Loading Pyannote embedding model...")
        get_pyannote_embedding_model()
        backend = get_config().get("models", {}).get("asr_backend", "funasr")
        if backend == "funasr":
            print("[PRELOAD] Loading FunASR model...")
            get_funasr_model()
        else:
            print("[PRELOAD] Loading Whisper model...")
            get_whisper_model(args.model)
        print("[PRELOAD] Done. Models cached; rerun without --preload-models to transcribe.")
        return
    
    # Create transcriber
    transcriber = AudioTranscriber(
        whisper_model=args.model,
        voices_db_path=args.voices_db,
        history_path=args.history
    )
    
    # Process based on mode
    if args.file:
        transcriber.process_file(args.file, force=args.force)
    elif args.dir:
        transcriber.process_directory(args.dir, pattern=args.pattern)
    elif args.watch:
        transcriber.watch_directory(args.watch, pattern=args.pattern)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python audio_to_text.py --file cache/audio_001.wav")
        print("  python audio_to_text.py --dir cache/ --model small")
        print("  python audio_to_text.py --watch cache/")
        print("  python audio_to_text.py --list-speakers")
        print("  python audio_to_text.py --label-speaker SPK_0001 'John Doe'")


if __name__ == "__main__":
    main()
