#!/usr/bin/env python3
"""
Memory Module - Short-term and Long-term Memory Management

Short-term: LanceDB with vector search and rotation control
Long-term: LightRAG integration for knowledge graph-based memory

Features:
- LanceDB for recent VL results and conversations
- Automatic rotation (max items or time-based)
- Vector embeddings using qwen3-embedding
- Export to LightRAG for graph-based long-term memory
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib

import lancedb
import pyarrow as pa
import requests

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
LLM_DIR = BASE_DIR / "llm"
LANCEDB_PATH = LLM_DIR / "memory_db"
LIGHTRAG_WORKING_DIR = LLM_DIR / "lightrag_data"

# LanceDB Configuration
MAX_SHORT_TERM_ITEMS = int(os.getenv("MEMORY_MAX_ITEMS", "2000"))  # Max items before rotation
SHORT_TERM_TTL_HOURS = int(os.getenv("MEMORY_TTL_HOURS", "24"))   # Hours before rotation
ROTATION_BATCH_SIZE = int(os.getenv("MEMORY_ROTATION_BATCH", "50"))  # Items to rotate at once

# LightRAG Configuration
LIGHTRAG_URL = os.getenv("LIGHTRAG_URL", "http://localhost:9621")
LIGHTRAG_API_KEY = os.getenv("LIGHTRAG_API_KEY", "")

# Ollama for embeddings
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.2.6:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))  # qwen3-embedding:0.6b dimension


@dataclass
class MemoryEntry:
    """A single memory entry for short-term storage."""
    id: str
    timestamp: float
    entry_type: str  # "vl_result", "llm_response", "conversation"
    
    # Content
    clip_path: Optional[str] = None
    transcription: Optional[str] = None
    scene: Optional[str] = None
    speech: Optional[str] = None
    summary: Optional[str] = None
    llm_response: Optional[str] = None
    
    # Metadata
    speakers: Optional[List[str]] = None
    duration: Optional[float] = None
    model: Optional[str] = None
    
    # Vector embedding (computed from summary/content)
    embedding: Optional[List[float]] = None
    
    # Rotation tracking
    rotated: bool = False
    rotated_at: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def get_text_for_embedding(self) -> str:
        """Get the text to use for embedding generation."""
        parts = []
        if self.summary:
            parts.append(self.summary)
        if self.scene:
            parts.append(f"Scene: {self.scene}")
        if self.speech:
            parts.append(f"Speech: {self.speech}")
        if self.transcription:
            parts.append(f"Transcription: {self.transcription}")
        if self.llm_response:
            parts.append(f"Response: {self.llm_response}")
        return " ".join(parts) if parts else ""
    
    def to_document(self) -> str:
        """Convert entry to a document string for LightRAG."""
        timestamp_str = datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        parts = [f"[{timestamp_str}] {self.entry_type}"]
        
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        if self.scene:
            parts.append(f"Scene: {self.scene}")
        if self.speech:
            parts.append(f"Speech: {self.speech}")
        if self.transcription:
            parts.append(f"Transcription: {self.transcription}")
        if self.llm_response:
            parts.append(f"LLM Response: {self.llm_response}")
        if self.speakers:
            parts.append(f"Speakers: {', '.join(self.speakers)}")
        
        return "\n".join(parts)


class EmbeddingService:
    """Generate embeddings using Ollama with qwen3-embedding."""
    
    def __init__(self, ollama_url: str = OLLAMA_URL, model: str = EMBEDDING_MODEL):
        self.ollama_url = ollama_url
        self.model = model
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using Ollama."""
        if not text.strip():
            return None
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model, "prompt": text, "keep_alive": -1},
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("embedding")
        except requests.RequestException as e:
            print(f"[WARN] Embedding generation failed: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if embedding service is available."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                return any(self.model in m for m in models)
            return False
        except requests.RequestException:
            return False


class ShortTermMemory:
    """
    LanceDB-based short-term memory with rotation control.
    
    Stores recent VL results, LLM responses, and conversations
    with vector embeddings for similarity search.
    """
    
    def __init__(
        self,
        db_path: Path = LANCEDB_PATH,
        max_items: int = MAX_SHORT_TERM_ITEMS,
        ttl_hours: int = SHORT_TERM_TTL_HOURS,
        embedding_dim: int = EMBEDDING_DIM,
    ):
        self.db_path = db_path
        self.max_items = max_items
        self.ttl_hours = ttl_hours
        self.embedding_dim = embedding_dim
        self.embedding_service = EmbeddingService()
        
        # Define schema with configurable embedding dimension
        self.schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("timestamp", pa.float64()),
            pa.field("entry_type", pa.string()),
            pa.field("clip_path", pa.string()),
            pa.field("transcription", pa.string()),
            pa.field("scene", pa.string()),
            pa.field("speech", pa.string()),
            pa.field("summary", pa.string()),
            pa.field("llm_response", pa.string()),
            pa.field("speakers", pa.string()),  # JSON encoded
            pa.field("duration", pa.float64()),
            pa.field("model", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), embedding_dim)),
            pa.field("rotated", pa.bool_()),
            pa.field("rotated_at", pa.float64()),
        ])
        
        # Initialize LanceDB
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        
        # Create or open table
        self._init_table()
    
    def _init_table(self):
        """Initialize the memory table."""
        try:
            self.table = self.db.open_table("short_term_memory")
        except Exception:
            # Create empty table with schema
            self.table = self.db.create_table(
                "short_term_memory",
                schema=self.schema,
            )
    
    def _generate_id(self, entry: MemoryEntry) -> str:
        """Generate unique ID for an entry."""
        content = f"{entry.timestamp}:{entry.entry_type}:{entry.clip_path or ''}:{entry.summary or ''}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _entry_to_record(self, entry: MemoryEntry) -> Dict[str, Any]:
        """Convert MemoryEntry to LanceDB record."""
        return {
            "id": entry.id,
            "timestamp": entry.timestamp,
            "entry_type": entry.entry_type,
            "clip_path": entry.clip_path or "",
            "transcription": entry.transcription or "",
            "scene": entry.scene or "",
            "speech": entry.speech or "",
            "summary": entry.summary or "",
            "llm_response": entry.llm_response or "",
            "speakers": json.dumps(entry.speakers or []),
            "duration": entry.duration or 0.0,
            "model": entry.model or "",
            "embedding": entry.embedding or [0.0] * self.embedding_dim,
            "rotated": entry.rotated,
            "rotated_at": entry.rotated_at or 0.0,
        }
    
    def _record_to_entry(self, record: Dict[str, Any]) -> MemoryEntry:
        """Convert LanceDB record to MemoryEntry."""
        return MemoryEntry(
            id=record["id"],
            timestamp=record["timestamp"],
            entry_type=record["entry_type"],
            clip_path=record["clip_path"] or None,
            transcription=record["transcription"] or None,
            scene=record["scene"] or None,
            speech=record["speech"] or None,
            summary=record["summary"] or None,
            llm_response=record["llm_response"] or None,
            speakers=json.loads(record["speakers"]) if record["speakers"] else None,
            duration=record["duration"] or None,
            model=record["model"] or None,
            embedding=list(record["embedding"]) if record["embedding"] is not None else None,
            rotated=record["rotated"],
            rotated_at=record["rotated_at"] or None,
        )
    
    def add(self, entry: MemoryEntry, generate_embedding: bool = True) -> str:
        """
        Add a memory entry to short-term storage.
        
        Args:
            entry: MemoryEntry to add
            generate_embedding: Whether to generate vector embedding
            
        Returns:
            ID of the added entry
        """
        # Generate ID if not set
        if not entry.id:
            entry.id = self._generate_id(entry)
        
        # Set timestamp if not set
        if not entry.timestamp:
            entry.timestamp = time.time()
        
        # Generate embedding
        if generate_embedding and not entry.embedding:
            text = entry.get_text_for_embedding()
            if text:
                entry.embedding = self.embedding_service.get_embedding(text)
        
        # Add to table
        record = self._entry_to_record(entry)
        self.table.add([record])
        
        return entry.id
    
    def add_vl_result(
        self,
        clip_path: str,
        transcription: str,
        scene: str,
        speech: str,
        summary: str,
        speakers: List[str] = None,
        duration: float = None,
        model: str = None,
        timestamp: float = None,
    ) -> str:
        """Convenience method to add a VL processing result."""
        entry = MemoryEntry(
            id="",
            timestamp=timestamp or time.time(),
            entry_type="vl_result",
            clip_path=clip_path,
            transcription=transcription,
            scene=scene,
            speech=speech,
            summary=summary,
            speakers=speakers,
            duration=duration,
            model=model,
        )
        return self.add(entry)
    
    def add_llm_response(
        self,
        response: str,
        context_summary: str = None,
        model: str = None,
        timestamp: float = None,
    ) -> str:
        """Convenience method to add an LLM response."""
        entry = MemoryEntry(
            id="",
            timestamp=timestamp or time.time(),
            entry_type="llm_response",
            llm_response=response,
            summary=context_summary,
            model=model,
        )
        return self.add(entry)
    
    def get_recent(self, limit: int = 10, entry_type: str = None) -> List[MemoryEntry]:
        """
        Get most recent memory entries.
        
        Args:
            limit: Maximum number of entries to return
            entry_type: Filter by entry type (optional)
            
        Returns:
            List of MemoryEntry objects, newest first
        """
        try:
            # Query all non-rotated entries
            df = self.table.to_pandas()
            df = df[df["rotated"] == False]
            
            if entry_type:
                df = df[df["entry_type"] == entry_type]
            
            df = df.sort_values("timestamp", ascending=False).head(limit)
            
            return [self._record_to_entry(row.to_dict()) for _, row in df.iterrows()]
        except Exception as e:
            print(f"[WARN] Failed to get recent entries: {e}")
            return []
    
    def search_similar(
        self,
        query: str,
        limit: int = 5,
        entry_type: str = None,
    ) -> List[tuple[MemoryEntry, float]]:
        """
        Search for similar memories using vector similarity.
        
        Args:
            query: Text to search for
            limit: Maximum number of results
            entry_type: Filter by entry type (optional)
            
        Returns:
            List of (MemoryEntry, similarity_score) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_service.get_embedding(query)
        if not query_embedding:
            return []
        
        try:
            # Vector search
            results = self.table.search(query_embedding).limit(limit * 2).to_pandas()
            
            # Filter non-rotated and by type
            results = results[results["rotated"] == False]
            if entry_type:
                results = results[results["entry_type"] == entry_type]
            
            results = results.head(limit)
            
            entries = []
            for _, row in results.iterrows():
                entry = self._record_to_entry(row.to_dict())
                score = row.get("_distance", 0.0)
                entries.append((entry, 1.0 - score))  # Convert distance to similarity
            
            return entries
        except Exception as e:
            print(f"[WARN] Vector search failed: {e}")
            return []
    
    def get_context_window(
        self,
        time_window_minutes: int = 5,
        max_items: int = 10,
    ) -> List[MemoryEntry]:
        """
        Get recent context within a time window.
        
        Args:
            time_window_minutes: Minutes to look back
            max_items: Maximum items to return
            
        Returns:
            List of MemoryEntry objects in chronological order
        """
        cutoff = time.time() - (time_window_minutes * 60)
        
        try:
            df = self.table.to_pandas()
            df = df[(df["rotated"] == False) & (df["timestamp"] >= cutoff)]
            df = df.sort_values("timestamp", ascending=True).tail(max_items)
            
            return [self._record_to_entry(row.to_dict()) for _, row in df.iterrows()]
        except Exception as e:
            print(f"[WARN] Failed to get context window: {e}")
            return []
    
    def count(self, include_rotated: bool = False) -> int:
        """Count entries in memory."""
        try:
            df = self.table.to_pandas()
            if not include_rotated:
                df = df[df["rotated"] == False]
            return len(df)
        except Exception:
            return 0
    
    def get_entries_for_rotation(self) -> List[MemoryEntry]:
        """
        Get entries that should be rotated to long-term memory.
        
        Returns entries that exceed max_items or TTL.
        """
        entries_to_rotate = []
        
        try:
            df = self.table.to_pandas()
            df = df[df["rotated"] == False].sort_values("timestamp", ascending=True)
            
            # Check max items
            total_active = len(df)
            if total_active > self.max_items:
                excess = total_active - self.max_items + ROTATION_BATCH_SIZE
                old_entries = df.head(excess)
                for _, row in old_entries.iterrows():
                    entries_to_rotate.append(self._record_to_entry(row.to_dict()))
            
            # Check TTL
            cutoff = time.time() - (self.ttl_hours * 3600)
            expired = df[df["timestamp"] < cutoff]
            for _, row in expired.iterrows():
                entry = self._record_to_entry(row.to_dict())
                if entry.id not in [e.id for e in entries_to_rotate]:
                    entries_to_rotate.append(entry)
            
            return entries_to_rotate
        except Exception as e:
            print(f"[WARN] Failed to get entries for rotation: {e}")
            return []
    
    def mark_rotated(self, entry_ids: List[str]) -> int:
        """
        Mark entries as rotated (moved to long-term memory).
        
        Args:
            entry_ids: List of entry IDs to mark
            
        Returns:
            Number of entries marked
        """
        if not entry_ids:
            return 0
        
        try:
            df = self.table.to_pandas()
            rotated_at = time.time()
            
            # Update rotated entries
            mask = df["id"].isin(entry_ids)
            df.loc[mask, "rotated"] = True
            df.loc[mask, "rotated_at"] = rotated_at
            
            # Recreate table with updated data
            self.db.drop_table("short_term_memory")
            self.table = self.db.create_table("short_term_memory", df)
            
            return mask.sum()
        except Exception as e:
            print(f"[WARN] Failed to mark entries as rotated: {e}")
            return 0
    
    def cleanup_rotated(self, older_than_hours: int = 48) -> int:
        """
        Delete rotated entries older than specified hours.
        
        Args:
            older_than_hours: Delete rotated entries older than this
            
        Returns:
            Number of entries deleted
        """
        try:
            df = self.table.to_pandas()
            cutoff = time.time() - (older_than_hours * 3600)
            
            # Find entries to delete
            to_delete = df[(df["rotated"] == True) & (df["rotated_at"] < cutoff)]
            delete_count = len(to_delete)
            
            if delete_count > 0:
                # Keep non-deleted entries
                df = df[~((df["rotated"] == True) & (df["rotated_at"] < cutoff))]
                
                # Recreate table
                self.db.drop_table("short_term_memory")
                self.table = self.db.create_table("short_term_memory", df)
            
            return delete_count
        except Exception as e:
            print(f"[WARN] Failed to cleanup rotated entries: {e}")
            return 0


class LightRAGClient:
    """
    Client for LightRAG long-term memory integration.
    
    LightRAG uses knowledge graphs to store and retrieve information,
    providing better context understanding through entity relationships.
    """
    
    def __init__(
        self,
        base_url: str = LIGHTRAG_URL,
        api_key: str = LIGHTRAG_API_KEY,
        working_dir: Path = LIGHTRAG_WORKING_DIR,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.working_dir = working_dir
        self.working_dir.mkdir(parents=True, exist_ok=True)
    
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def is_available(self) -> bool:
        """Check if LightRAG server is available."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self._headers(),
                timeout=5
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def insert_text(self, text: str) -> bool:
        """
        Insert text into LightRAG for knowledge graph construction.
        
        Args:
            text: Text content to insert
            
        Returns:
            True if successful
        """
        try:
            response = requests.post(
                f"{self.base_url}/documents/text",
                headers=self._headers(),
                json={"text": text},
                timeout=120
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            print(f"[WARN] Failed to insert into LightRAG: {e}")
            return False
    
    def insert_file(self, file_path: str) -> bool:
        """
        Insert a file into LightRAG.
        
        Args:
            file_path: Path to file to insert
            
        Returns:
            True if successful
        """
        try:
            with open(file_path, "rb") as f:
                response = requests.post(
                    f"{self.base_url}/insert_file",
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                    files={"file": f},
                    timeout=120
                )
            response.raise_for_status()
            return True
        except (requests.RequestException, IOError) as e:
            print(f"[WARN] Failed to insert file into LightRAG: {e}")
            return False
    
    def query(
        self,
        question: str,
        mode: str = "hybrid",
        only_need_context: bool = False,
    ) -> Optional[str]:
        """
        Query the LightRAG knowledge graph.
        
        Args:
            question: Question to ask
            mode: Query mode - "naive", "local", "global", or "hybrid"
            only_need_context: If True, return only retrieved context without LLM response
            
        Returns:
            Response string or None if failed
        """
        try:
            response = requests.post(
                f"{self.base_url}/query",
                headers=self._headers(),
                json={
                    "query": question,
                    "mode": mode,
                    "only_need_context": only_need_context,
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response") or result.get("context")
        except requests.RequestException as e:
            print(f"[WARN] Failed to query LightRAG: {e}")
            return None
    
    def query_with_context(self, question: str, mode: str = "hybrid") -> Dict[str, Any]:
        """
        Query LightRAG and get both response and context.
        
        Args:
            question: Question to ask
            mode: Query mode
            
        Returns:
            Dict with "response" and "context" keys
        """
        result = {"response": None, "context": None}
        
        # Get context
        result["context"] = self.query(question, mode=mode, only_need_context=True)
        
        # Get response
        result["response"] = self.query(question, mode=mode, only_need_context=False)
        
        return result
    
    def get_knowledge_graph_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics about the knowledge graph."""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self._headers(),
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            return None


class MemoryManager:
    """
    Unified memory manager for short-term and long-term memory.
    
    Handles:
    - Adding new memories to short-term (LanceDB)
    - Automatic rotation to long-term (LightRAG)
    - Querying both memory stores
    """
    
    def __init__(
        self,
        db_path: Path = LANCEDB_PATH,
        max_items: int = MAX_SHORT_TERM_ITEMS,
        ttl_hours: int = SHORT_TERM_TTL_HOURS,
        auto_rotate: bool = True,
    ):
        self.short_term = ShortTermMemory(db_path, max_items, ttl_hours)
        self.long_term = LightRAGClient()
        self.auto_rotate = auto_rotate
    
    def add_vl_result(self, **kwargs) -> str:
        """Add a VL processing result to short-term memory."""
        entry_id = self.short_term.add_vl_result(**kwargs)
        
        if self.auto_rotate:
            self._check_and_rotate()
        
        return entry_id
    
    def add_llm_response(self, **kwargs) -> str:
        """Add an LLM response to short-term memory."""
        entry_id = self.short_term.add_llm_response(**kwargs)
        
        if self.auto_rotate:
            self._check_and_rotate()
        
        return entry_id
    
    def get_recent_context(
        self,
        limit: int = 10,
        time_window_minutes: int = None,
    ) -> List[MemoryEntry]:
        """Get recent context from short-term memory."""
        if time_window_minutes:
            return self.short_term.get_context_window(time_window_minutes, limit)
        return self.short_term.get_recent(limit)
    
    def search(
        self,
        query: str,
        limit: int = 5,
        include_long_term: bool = True,
    ) -> Dict[str, Any]:
        """
        Search both short-term and long-term memory.
        
        Returns:
            Dict with "short_term" and "long_term" results
        """
        results = {
            "short_term": [],
            "long_term": None,
        }
        
        # Search short-term
        short_results = self.short_term.search_similar(query, limit)
        results["short_term"] = [
            {"entry": entry.to_dict(), "score": score}
            for entry, score in short_results
        ]
        
        # Search long-term
        if include_long_term and self.long_term.is_available():
            results["long_term"] = self.long_term.query(query, mode="hybrid")
        
        return results
    
    def _check_and_rotate(self) -> int:
        """Check if rotation is needed and perform it."""
        entries = self.short_term.get_entries_for_rotation()
        if not entries:
            return 0
        
        return self.rotate_to_long_term(entries)
    
    def rotate_to_long_term(self, entries: List[MemoryEntry] = None) -> int:
        """
        Rotate entries from short-term to long-term memory.
        
        Args:
            entries: Specific entries to rotate (or auto-select if None)
            
        Returns:
            Number of entries rotated
        """
        if entries is None:
            entries = self.short_term.get_entries_for_rotation()
        
        if not entries:
            return 0
        
        # Check if LightRAG is available
        if not self.long_term.is_available():
            print("[WARN] LightRAG not available, marking entries as rotated anyway")
            self.short_term.mark_rotated([e.id for e in entries])
            return len(entries)
        
        # Format entries as a document for LightRAG
        doc_content = self._format_entries_for_long_term(entries)
        
        # Insert into LightRAG
        success = self.long_term.insert_text(doc_content)
        
        if success:
            print(f"[MEMORY] Inserted {len(entries)} entries into LightRAG knowledge graph")
        
        # Mark as rotated in short-term
        self.short_term.mark_rotated([e.id for e in entries])
        
        print(f"[MEMORY] Rotated {len(entries)} entries to long-term memory")
        return len(entries)
    
    def _format_entries_for_long_term(self, entries: List[MemoryEntry]) -> str:
        """Format entries as text for LightRAG knowledge graph."""
        documents = []
        
        for entry in sorted(entries, key=lambda e: e.timestamp):
            documents.append(entry.to_document())
        
        # Create a structured document for better knowledge extraction
        header = f"""# Memory Archive - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This document contains {len(entries)} memory entries from the short-term memory buffer.
Time range: {datetime.fromtimestamp(min(e.timestamp for e in entries)).isoformat()} to {datetime.fromtimestamp(max(e.timestamp for e in entries)).isoformat()}

## Entries

"""
        return header + "\n\n---\n\n".join(documents)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        long_term_stats = self.long_term.get_knowledge_graph_stats()
        
        return {
            "short_term": {
                "active_count": self.short_term.count(include_rotated=False),
                "rotated_count": self.short_term.count(include_rotated=True) - self.short_term.count(include_rotated=False),
                "max_items": self.short_term.max_items,
                "ttl_hours": self.short_term.ttl_hours,
            },
            "long_term": {
                "available": self.long_term.is_available(),
                "url": self.long_term.base_url,
                "stats": long_term_stats,
            },
            "embedding": {
                "available": self.short_term.embedding_service.is_available(),
                "model": self.short_term.embedding_service.model,
            }
        }


# Convenience function for global access
_memory_manager: Optional[MemoryManager] = None

def get_memory() -> MemoryManager:
    """Get or create the global memory manager."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def main():
    """CLI for testing memory operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory Management CLI")
    parser.add_argument("--stats", action="store_true", help="Show memory statistics")
    parser.add_argument("--recent", type=int, default=0, help="Show N recent entries")
    parser.add_argument("--search", type=str, help="Search memories")
    parser.add_argument("--rotate", action="store_true", help="Force rotation to long-term")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup old rotated entries")
    parser.add_argument("--query-lightrag", type=str, help="Query LightRAG directly")
    
    args = parser.parse_args()
    
    memory = get_memory()
    
    if args.stats:
        stats = memory.get_stats()
        print(json.dumps(stats, indent=2))
    
    elif args.recent > 0:
        entries = memory.get_recent_context(limit=args.recent)
        for entry in entries:
            ts = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S")
            print(f"[{ts}] {entry.entry_type}: {entry.summary or entry.llm_response or '(no summary)'}")
    
    elif args.search:
        results = memory.search(args.search)
        print(f"Short-term results ({len(results['short_term'])}):")
        for r in results["short_term"]:
            print(f"  - [{r['score']:.2f}] {r['entry']['summary'] or r['entry']['llm_response']}")
        if results["long_term"]:
            print(f"\nLong-term (LightRAG) response:\n{results['long_term']}")
    
    elif args.rotate:
        count = memory.rotate_to_long_term()
        print(f"Rotated {count} entries to long-term memory")
    
    elif args.cleanup:
        count = memory.short_term.cleanup_rotated()
        print(f"Cleaned up {count} old rotated entries")
    
    elif args.query_lightrag:
        if memory.long_term.is_available():
            response = memory.long_term.query(args.query_lightrag)
            print(f"LightRAG response:\n{response}")
        else:
            print("LightRAG is not available")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
