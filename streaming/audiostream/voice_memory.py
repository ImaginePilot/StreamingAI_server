#!/usr/bin/env python3
"""
Voice Memory Module - Persistent Speaker Recognition Across Sessions

This module provides cross-context speaker identification by maintaining a database
of voice embeddings (voiceprints). When a new speaker segment is encountered, it
computes the voice embedding and compares against stored voiceprints to identify
returning speakers.

Features:
- Persistent voice embedding storage (JSON + optional numpy cache)
- Rolling average embedding updates for improved accuracy over time
- Configurable similarity thresholds
- Multiple embedding models support (WeSpeaker, ECAPA-TDNN)
- Speaker labeling and management

Usage:
    from voice_memory import VoiceMemory
    
    vm = VoiceMemory()
    
    # Identify a speaker from audio segment
    speaker_id, label, confidence, is_new = vm.identify_speaker(audio_waveform, sample_rate)
    
    # Label a speaker
    vm.label_speaker("SPK_0001", "John Doe")
    
    # List all known speakers
    speakers = vm.list_speakers()
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
import warnings
import threading

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
DEFAULT_DB_PATH = SCRIPT_DIR / "transcripts" / "voices.json"
DEFAULT_EMBEDDINGS_DIR = SCRIPT_DIR / "transcripts" / "voice_embeddings"

# Default thresholds (can be overridden in config)
DEFAULT_SIMILARITY_THRESHOLD = 0.65  # Lower than before for cross-context matching
DEFAULT_HIGH_CONFIDENCE_THRESHOLD = 0.80  # High confidence match
DEFAULT_MIN_SEGMENT_DURATION = 0.5  # Minimum seconds of audio for embedding

# Embedding model cache
_embedding_model = None
_embedding_lock = threading.Lock()


def _load_config() -> dict:
    """Load configuration from audio_env.json if available."""
    config_file = SCRIPT_DIR / "audio_env.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                return json.load(f)
        except:
            pass
    return {}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class VoicePrint:
    """A stored voice embedding with metadata."""
    speaker_id: str
    label: Optional[str]
    embedding: List[float]
    embedding_count: int  # How many samples contributed to this embedding
    first_seen: str
    last_seen: str
    total_duration: float  # Total seconds of audio used
    confidence_history: List[float] = field(default_factory=list)  # Recent match confidences
    
    def get_embedding_array(self) -> np.ndarray:
        """Return embedding as numpy array."""
        return np.array(self.embedding, dtype=np.float32)


@dataclass  
class IdentificationResult:
    """Result of speaker identification."""
    speaker_id: str
    label: Optional[str]
    confidence: float
    is_new: bool
    matched_count: int  # How many stored voiceprints were compared
    

# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------

def get_embedding_model():
    """
    Get the voice embedding model (lazy loaded, thread-safe).
    Uses pyannote/wespeaker-voxceleb-resnet34-LM by default.
    """
    global _embedding_model
    
    if _embedding_model is not None:
        return _embedding_model
    
    with _embedding_lock:
        if _embedding_model is not None:
            return _embedding_model
        
        try:
            from pyannote.audio import Model
            import torch
            
            # Get HF token
            hf_token = _get_hf_token()
            if not hf_token:
                print("[WARN] No HuggingFace token found, embedding model may fail")
            
            model_name = "pyannote/wespeaker-voxceleb-resnet34-LM"
            print(f"[VoiceMemory] Loading embedding model: {model_name}")
            
            model = Model.from_pretrained(model_name, token=hf_token)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                model = model.to(torch.device("cuda"))
                print("[VoiceMemory] Embedding model on GPU")
            else:
                print("[VoiceMemory] Embedding model on CPU")
            
            _embedding_model = model
            print("[VoiceMemory] Embedding model loaded")
            
        except Exception as e:
            print(f"[VoiceMemory] Failed to load embedding model: {e}")
            raise
    
    return _embedding_model


def _get_hf_token() -> Optional[str]:
    """Get HuggingFace token from various sources."""
    # Try environment variable
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    
    # Try audio_env.json
    config_file = SCRIPT_DIR / "audio_env.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
            token = config.get("huggingface_token")
            if token:
                return token
        except:
            pass
    
    # Try hf_token.txt
    token_file = SCRIPT_DIR / "hf_token.txt"
    if token_file.exists():
        try:
            token = token_file.read_text().strip()
            if token:
                return token
        except:
            pass
    
    return None


def compute_embedding(
    audio_waveform: np.ndarray,
    sample_rate: int = 16000
) -> Optional[np.ndarray]:
    """
    Compute voice embedding from audio waveform.
    
    Args:
        audio_waveform: Audio samples as numpy array (channels, samples) or (samples,)
        sample_rate: Sample rate of audio
        
    Returns:
        Embedding vector as numpy array, or None if failed
    """
    try:
        import torch
        from pyannote.audio import Inference
        
        model = get_embedding_model()
        
        # Ensure correct shape
        if audio_waveform.ndim == 1:
            audio_waveform = audio_waveform.reshape(1, -1)
        elif audio_waveform.ndim == 3:
            audio_waveform = audio_waveform.squeeze(0)
        
        # Convert to mono if stereo
        if audio_waveform.shape[0] > 1:
            audio_waveform = audio_waveform.mean(axis=0, keepdims=True)
        
        # Create inference object
        inference = Inference(model, window="whole")
        
        # Prepare audio dict for Inference
        audio_dict = {
            "waveform": torch.from_numpy(audio_waveform).float(),
            "sample_rate": sample_rate
        }
        
        # Compute embedding
        embedding = inference(audio_dict)
        return embedding.flatten().astype(np.float32)
        
    except Exception as e:
        print(f"[VoiceMemory] Failed to compute embedding: {e}")
        return None


# ---------------------------------------------------------------------------
# Voice Memory Database
# ---------------------------------------------------------------------------

class VoiceMemory:
    """
    Persistent database of voice embeddings for cross-session speaker identification.
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        high_confidence_threshold: Optional[float] = None,
        min_segment_duration: float = DEFAULT_MIN_SEGMENT_DURATION
    ):
        """
        Initialize VoiceMemory.
        
        Args:
            db_path: Path to voices.json database
            similarity_threshold: Minimum cosine similarity to match a speaker (default from config)
            high_confidence_threshold: Threshold for high-confidence matches (default from config)
            min_segment_duration: Minimum audio duration (seconds) for reliable embedding
        """
        # Load config for defaults
        config = _load_config()
        settings = config.get("settings", {})
        
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use config values or defaults
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else \
            settings.get("speaker_similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
        self.high_confidence_threshold = high_confidence_threshold if high_confidence_threshold is not None else \
            settings.get("voice_memory_high_confidence_threshold", DEFAULT_HIGH_CONFIDENCE_THRESHOLD)
        self.min_segment_duration = min_segment_duration
        
        print(f"[VoiceMemory] Thresholds: similarity={self.similarity_threshold}, high_conf={self.high_confidence_threshold}")
        
        self.voiceprints: Dict[str, VoicePrint] = {}
        self.next_speaker_num = 1
        self._lock = threading.Lock()
        
        self._load()
    
    def _load(self):
        """Load voiceprints from database."""
        if not self.db_path.exists():
            print(f"[VoiceMemory] No database found at {self.db_path}, starting fresh")
            return
        
        try:
            raw = self.db_path.read_text().strip()
            if not raw:
                print("[VoiceMemory] Empty database file, starting fresh")
                return
            
            data = json.loads(raw)
            
            # Load voiceprints
            for sid, vp_data in data.get("voiceprints", data.get("speakers", {})).items():
                # Handle both old format (speakers) and new format (voiceprints)
                if "embedding" in vp_data:
                    self.voiceprints[sid] = VoicePrint(
                        speaker_id=sid,
                        label=vp_data.get("label"),
                        embedding=vp_data["embedding"],
                        embedding_count=vp_data.get("embedding_count", vp_data.get("sample_count", 1)),
                        first_seen=vp_data.get("first_seen", datetime.now().isoformat()),
                        last_seen=vp_data.get("last_seen", datetime.now().isoformat()),
                        total_duration=vp_data.get("total_duration", 0.0),
                        confidence_history=vp_data.get("confidence_history", [])
                    )
            
            self.next_speaker_num = data.get("next_speaker_num", len(self.voiceprints) + 1)
            
            # NOTE: Do NOT load thresholds from DB - config always takes precedence
            # This ensures audio_env.json changes are respected
            
            print(f"[VoiceMemory] Loaded {len(self.voiceprints)} voiceprints from database")
            
        except Exception as e:
            print(f"[VoiceMemory] Failed to load database: {e}")
            self.voiceprints = {}
            self.next_speaker_num = 1
    
    def _save(self):
        """Save voiceprints to database."""
        data = {
            "voiceprints": {
                sid: {
                    "speaker_id": vp.speaker_id,
                    "label": vp.label,
                    "embedding": vp.embedding,
                    "embedding_count": vp.embedding_count,
                    "first_seen": vp.first_seen,
                    "last_seen": vp.last_seen,
                    "total_duration": vp.total_duration,
                    "confidence_history": vp.confidence_history[-10:]  # Keep last 10
                }
                for sid, vp in self.voiceprints.items()
            },
            "next_speaker_num": self.next_speaker_num,
            "similarity_threshold": self.similarity_threshold,
            "high_confidence_threshold": self.high_confidence_threshold,
            "total_speakers": len(self.voiceprints),
            "updated": datetime.now().isoformat()
        }
        
        try:
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[VoiceMemory] Failed to save database: {e}")
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        emb1 = np.array(emb1, dtype=np.float32).flatten()
        emb2 = np.array(emb2, dtype=np.float32).flatten()
        
        dot = np.dot(emb1, emb2)
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        
        if norm < 1e-8:
            return 0.0
        
        return float(dot / norm)
    
    def _update_embedding(
        self,
        voiceprint: VoicePrint,
        new_embedding: np.ndarray,
        duration: float,
        confidence: float
    ):
        """Update voiceprint with new embedding using weighted rolling average."""
        old_emb = voiceprint.get_embedding_array()
        count = voiceprint.embedding_count
        
        # Weight based on count and confidence
        # Newer samples get more weight when count is low
        # High confidence matches get more weight
        alpha = min(0.3, 1.0 / (count + 1)) * (0.5 + 0.5 * confidence)
        
        # Rolling average
        new_emb_normalized = new_embedding / (np.linalg.norm(new_embedding) + 1e-8)
        old_emb_normalized = old_emb / (np.linalg.norm(old_emb) + 1e-8)
        
        updated = (1 - alpha) * old_emb_normalized + alpha * new_emb_normalized
        updated = updated / (np.linalg.norm(updated) + 1e-8)  # Re-normalize
        
        voiceprint.embedding = updated.tolist()
        voiceprint.embedding_count += 1
        voiceprint.total_duration += duration
        voiceprint.last_seen = datetime.now().isoformat()
        voiceprint.confidence_history.append(round(confidence, 3))
    
    def identify_speaker(
        self,
        audio_waveform: np.ndarray,
        sample_rate: int = 16000,
        duration: Optional[float] = None
    ) -> IdentificationResult:
        """
        Identify speaker from audio waveform.
        
        Args:
            audio_waveform: Audio samples (channels, samples) or (samples,)
            sample_rate: Sample rate
            duration: Duration of segment in seconds (for logging)
            
        Returns:
            IdentificationResult with speaker_id, label, confidence, is_new
        """
        with self._lock:
            # Compute embedding
            embedding = compute_embedding(audio_waveform, sample_rate)
            
            if embedding is None:
                # Failed to compute embedding, return unknown
                return IdentificationResult(
                    speaker_id="UNKNOWN",
                    label=None,
                    confidence=0.0,
                    is_new=False,
                    matched_count=0
                )
            
            # Calculate duration if not provided
            if duration is None:
                samples = audio_waveform.shape[-1] if audio_waveform.ndim > 1 else len(audio_waveform)
                duration = samples / sample_rate
            
            # Find best matching voiceprint
            best_match: Optional[str] = None
            best_similarity = 0.0
            
            for sid, vp in self.voiceprints.items():
                similarity = self._cosine_similarity(embedding, vp.get_embedding_array())
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = sid
            
            matched_count = len(self.voiceprints)
            
            # Debug output
            if best_match:
                print(f"[VoiceMemory] Best match: {best_match} (similarity: {best_similarity:.3f}, threshold: {self.similarity_threshold})")
            
            # Check if match exceeds threshold
            if best_match and best_similarity >= self.similarity_threshold:
                voiceprint = self.voiceprints[best_match]
                
                # Update embedding with new sample
                self._update_embedding(voiceprint, embedding, duration, best_similarity)
                self._save()
                
                confidence_level = "HIGH" if best_similarity >= self.high_confidence_threshold else "MEDIUM"
                print(f"[VoiceMemory] {confidence_level} confidence match: {best_match} -> {voiceprint.label or '(unlabeled)'}")
                
                return IdentificationResult(
                    speaker_id=best_match,
                    label=voiceprint.label,
                    confidence=best_similarity,
                    is_new=False,
                    matched_count=matched_count
                )
            
            # Create new speaker
            speaker_id = f"SPK_{self.next_speaker_num:04d}"
            self.next_speaker_num += 1
            
            self.voiceprints[speaker_id] = VoicePrint(
                speaker_id=speaker_id,
                label=None,
                embedding=embedding.tolist(),
                embedding_count=1,
                first_seen=datetime.now().isoformat(),
                last_seen=datetime.now().isoformat(),
                total_duration=duration,
                confidence_history=[]
            )
            
            self._save()
            
            print(f"[VoiceMemory] NEW speaker created: {speaker_id}")
            
            return IdentificationResult(
                speaker_id=speaker_id,
                label=None,
                confidence=1.0,  # Perfect match with self
                is_new=True,
                matched_count=matched_count
            )
    
    def identify_from_embedding(
        self,
        embedding: np.ndarray,
        duration: float = 1.0
    ) -> IdentificationResult:
        """
        Identify speaker from pre-computed embedding.
        
        Args:
            embedding: Voice embedding vector
            duration: Duration of source audio (for stats)
            
        Returns:
            IdentificationResult
        """
        with self._lock:
            best_match: Optional[str] = None
            best_similarity = 0.0
            
            for sid, vp in self.voiceprints.items():
                similarity = self._cosine_similarity(embedding, vp.get_embedding_array())
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = sid
            
            matched_count = len(self.voiceprints)
            
            if best_match and best_similarity >= self.similarity_threshold:
                voiceprint = self.voiceprints[best_match]
                self._update_embedding(voiceprint, embedding, duration, best_similarity)
                self._save()
                
                return IdentificationResult(
                    speaker_id=best_match,
                    label=voiceprint.label,
                    confidence=best_similarity,
                    is_new=False,
                    matched_count=matched_count
                )
            
            # Create new speaker
            speaker_id = f"SPK_{self.next_speaker_num:04d}"
            self.next_speaker_num += 1
            
            self.voiceprints[speaker_id] = VoicePrint(
                speaker_id=speaker_id,
                label=None,
                embedding=embedding.tolist(),
                embedding_count=1,
                first_seen=datetime.now().isoformat(),
                last_seen=datetime.now().isoformat(),
                total_duration=duration,
                confidence_history=[]
            )
            
            self._save()
            
            return IdentificationResult(
                speaker_id=speaker_id,
                label=None,
                confidence=1.0,
                is_new=True,
                matched_count=matched_count
            )
    
    def label_speaker(self, speaker_id: str, label: str) -> bool:
        """
        Assign a human-readable label to a speaker.
        
        Args:
            speaker_id: The speaker ID (e.g., "SPK_0001")
            label: Human-readable name (e.g., "John Doe")
            
        Returns:
            True if successful, False if speaker not found
        """
        with self._lock:
            if speaker_id not in self.voiceprints:
                print(f"[VoiceMemory] Speaker not found: {speaker_id}")
                return False
            
            self.voiceprints[speaker_id].label = label
            self._save()
            print(f"[VoiceMemory] Labeled {speaker_id} as '{label}'")
            return True
    
    def get_speaker(self, speaker_id: str) -> Optional[Dict[str, Any]]:
        """Get speaker info by ID."""
        if speaker_id not in self.voiceprints:
            return None
        
        vp = self.voiceprints[speaker_id]
        return {
            "speaker_id": vp.speaker_id,
            "label": vp.label,
            "embedding_count": vp.embedding_count,
            "first_seen": vp.first_seen,
            "last_seen": vp.last_seen,
            "total_duration": vp.total_duration,
            "avg_confidence": np.mean(vp.confidence_history) if vp.confidence_history else None
        }
    
    def list_speakers(self) -> List[Dict[str, Any]]:
        """List all known speakers."""
        return [
            {
                "speaker_id": vp.speaker_id,
                "label": vp.label or "(unlabeled)",
                "embedding_count": vp.embedding_count,
                "first_seen": vp.first_seen,
                "last_seen": vp.last_seen,
                "total_duration": round(vp.total_duration, 1),
                "avg_confidence": round(np.mean(vp.confidence_history), 3) if vp.confidence_history else None
            }
            for vp in sorted(self.voiceprints.values(), key=lambda x: x.last_seen, reverse=True)
        ]
    
    def delete_speaker(self, speaker_id: str) -> bool:
        """Delete a speaker from the database."""
        with self._lock:
            if speaker_id not in self.voiceprints:
                return False
            
            del self.voiceprints[speaker_id]
            self._save()
            print(f"[VoiceMemory] Deleted speaker: {speaker_id}")
            return True
    
    def merge_speakers(self, source_id: str, target_id: str) -> bool:
        """
        Merge one speaker into another.
        
        Args:
            source_id: Speaker to merge from (will be deleted)
            target_id: Speaker to merge into
            
        Returns:
            True if successful
        """
        with self._lock:
            if source_id not in self.voiceprints or target_id not in self.voiceprints:
                return False
            
            source = self.voiceprints[source_id]
            target = self.voiceprints[target_id]
            
            # Weighted average of embeddings
            total_count = source.embedding_count + target.embedding_count
            source_weight = source.embedding_count / total_count
            target_weight = target.embedding_count / total_count
            
            source_emb = source.get_embedding_array()
            target_emb = target.get_embedding_array()
            
            merged_emb = source_weight * source_emb + target_weight * target_emb
            merged_emb = merged_emb / (np.linalg.norm(merged_emb) + 1e-8)
            
            target.embedding = merged_emb.tolist()
            target.embedding_count = total_count
            target.total_duration += source.total_duration
            target.confidence_history.extend(source.confidence_history)
            
            # Keep earliest first_seen
            if source.first_seen < target.first_seen:
                target.first_seen = source.first_seen
            
            # Delete source
            del self.voiceprints[source_id]
            self._save()
            
            print(f"[VoiceMemory] Merged {source_id} into {target_id}")
            return True
    
    def set_threshold(self, threshold: float):
        """Set similarity threshold."""
        self.similarity_threshold = threshold
        self._save()
        print(f"[VoiceMemory] Similarity threshold set to {threshold}")
    
    def clear(self):
        """Clear all voiceprints."""
        with self._lock:
            self.voiceprints = {}
            self.next_speaker_num = 1
            self._save()
            print("[VoiceMemory] Database cleared")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Command-line interface for voice memory management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Memory - Speaker Database Management")
    parser.add_argument("--db", default=None, help="Path to voices.json database")
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # List speakers
    list_parser = subparsers.add_parser("list", help="List all known speakers")
    
    # Label speaker
    label_parser = subparsers.add_parser("label", help="Label a speaker")
    label_parser.add_argument("speaker_id", help="Speaker ID (e.g., SPK_0001)")
    label_parser.add_argument("name", help="Human-readable name")
    
    # Delete speaker
    delete_parser = subparsers.add_parser("delete", help="Delete a speaker")
    delete_parser.add_argument("speaker_id", help="Speaker ID to delete")
    
    # Merge speakers
    merge_parser = subparsers.add_parser("merge", help="Merge speakers")
    merge_parser.add_argument("source", help="Source speaker ID (will be deleted)")
    merge_parser.add_argument("target", help="Target speaker ID (will be kept)")
    
    # Set threshold
    threshold_parser = subparsers.add_parser("threshold", help="Set similarity threshold")
    threshold_parser.add_argument("value", type=float, help="Threshold value (0.0-1.0)")
    
    # Clear database
    clear_parser = subparsers.add_parser("clear", help="Clear all speakers")
    clear_parser.add_argument("--confirm", action="store_true", help="Confirm clear")
    
    # Test with audio file
    test_parser = subparsers.add_parser("test", help="Test identification with audio file")
    test_parser.add_argument("audio_file", help="Path to audio file")
    
    args = parser.parse_args()
    
    vm = VoiceMemory(db_path=args.db)
    
    if args.command == "list":
        speakers = vm.list_speakers()
        if not speakers:
            print("No speakers in database")
        else:
            print(f"\n{'ID':<12} {'Label':<20} {'Samples':<8} {'Duration':<10} {'Last Seen':<20}")
            print("-" * 70)
            for s in speakers:
                print(f"{s['speaker_id']:<12} {s['label']:<20} {s['embedding_count']:<8} {s['total_duration']:<10} {s['last_seen'][:19]}")
    
    elif args.command == "label":
        vm.label_speaker(args.speaker_id, args.name)
    
    elif args.command == "delete":
        vm.delete_speaker(args.speaker_id)
    
    elif args.command == "merge":
        vm.merge_speakers(args.source, args.target)
    
    elif args.command == "threshold":
        vm.set_threshold(args.value)
    
    elif args.command == "clear":
        if args.confirm:
            vm.clear()
        else:
            print("Use --confirm to clear the database")
    
    elif args.command == "test":
        import torchaudio
        
        waveform, sr = torchaudio.load(args.audio_file)
        waveform = waveform.numpy()
        
        result = vm.identify_speaker(waveform, sr)
        print(f"\nIdentification Result:")
        print(f"  Speaker ID: {result.speaker_id}")
        print(f"  Label: {result.label or '(unlabeled)'}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Is New: {result.is_new}")
        print(f"  Compared Against: {result.matched_count} voiceprints")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
