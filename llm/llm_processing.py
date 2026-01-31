#!/usr/bin/env python3
"""
LLM Processing Module - Central Language Model Processing

Processes VL results and memory context to generate natural language responses
suitable for TTS output.

Data Flow:
1. Read unprocessed entries from vl_processed.json
2. Query LanceDB (short-term) and LightRAG (long-term) for context
3. Process with qwen3:30b LLM
4. Output JSON formatted text for TTS

Usage:
    python llm_processing.py --process           # Process pending VL results
    python llm_processing.py --query "question"  # Direct query with context
    python llm_processing.py --testrun           # Test with latest VL result
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Import memory module
try:
    from memory import get_memory, MemoryManager
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    print("[WARN] Memory module not available")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
LLM_DIR = BASE_DIR / "llm"
VL_PROCESSED_JSON = LLM_DIR / "vl_processed.json"
LLM_OUTPUT_JSON = LLM_DIR / "llm_output.json"
LLM_STATE_JSON = LLM_DIR / "llm_state.json"

# Ollama Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.2.6:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:30b")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_CONTEXT_SIZE = int(os.getenv("LLM_CONTEXT_SIZE", "8192"))

# LightRAG Configuration
LIGHTRAG_URL = os.getenv("LIGHTRAG_URL", "http://localhost:9621")


@dataclass
class LLMOutput:
    """Output from LLM processing, formatted for TTS."""
    timestamp: float
    think: str  # LLM's reasoning about whether/how to respond
    reply: str  # The actual response (can be empty if LLM decides not to speak)
    emotion: str  # neutral, happy, sad, excited, concerned, etc.
    priority: str  # low, normal, high, urgent
    should_speak: bool  # Whether TTS should vocalize this
    source_clips: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_tts_json(self) -> Dict[str, Any]:
        """Format specifically for TTS consumption."""
        return {
            "text": self.reply,
            "emotion": self.emotion,
            "priority": self.priority,
            "timestamp": self.timestamp,
            "should_speak": self.should_speak,
        }


class VLDataReader:
    """Reads and tracks VL processed data."""
    
    def __init__(self, vl_json_path: Path = VL_PROCESSED_JSON, state_path: Path = LLM_STATE_JSON):
        self.vl_json_path = vl_json_path
        self.state_path = state_path
        self.last_processed_ts = self._load_state()
    
    def _load_state(self) -> float:
        """Load last processed timestamp."""
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    state = json.load(f)
                    return state.get("last_processed_ts", 0.0)
            except (json.JSONDecodeError, IOError):
                pass
        return 0.0
    
    def _save_state(self, ts: float):
        """Save last processed timestamp."""
        with open(self.state_path, "w") as f:
            json.dump({"last_processed_ts": ts, "updated_at": datetime.now().isoformat()}, f)
        self.last_processed_ts = ts
    
    def get_unprocessed(self) -> List[Dict[str, Any]]:
        """Get VL results that haven't been processed by LLM yet."""
        if not self.vl_json_path.exists():
            return []
        
        try:
            with open(self.vl_json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
        
        results = data.get("results", {})
        unprocessed = []
        
        for key, entry in results.items():
            # Parse processed_at timestamp
            processed_at_str = entry.get("processed_at", "")
            try:
                processed_at = datetime.fromisoformat(processed_at_str).timestamp()
            except ValueError:
                processed_at = entry.get("start_ts", 0.0)
            
            # Check if newer than last processed
            if processed_at > self.last_processed_ts:
                entry["_key"] = key
                entry["_processed_at_ts"] = processed_at
                unprocessed.append(entry)
        
        # Sort by timestamp
        unprocessed.sort(key=lambda x: x.get("start_ts", 0))
        return unprocessed
    
    def mark_processed(self, entries: List[Dict[str, Any]]):
        """Mark entries as processed by updating state."""
        if not entries:
            return
        
        max_ts = max(e.get("_processed_at_ts", 0) for e in entries)
        if max_ts > self.last_processed_ts:
            self._save_state(max_ts)
    
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get the most recent VL result."""
        if not self.vl_json_path.exists():
            return None
        
        try:
            with open(self.vl_json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
        
        results = data.get("results", {})
        if not results:
            return None
        
        # Get most recent by start_ts
        latest = max(results.values(), key=lambda x: x.get("start_ts", 0))
        return latest


class ContextRetriever:
    """Retrieves context from LanceDB and LightRAG."""
    
    def __init__(self, memory: Optional[MemoryManager] = None):
        self.memory = memory or (get_memory() if MEMORY_AVAILABLE else None)
        self.lightrag_url = LIGHTRAG_URL
    
    def get_short_term_context(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent context from LanceDB (short-term memory)."""
        if not self.memory:
            return []
        
        try:
            results = self.memory.short_term.search_similar(query, limit=limit)
            return [
                {
                    "summary": entry.summary,
                    "scene": entry.scene,
                    "speech": entry.speech,
                    "timestamp": entry.timestamp,
                    "score": score,
                }
                for entry, score in results
            ]
        except Exception as e:
            print(f"[WARN] Short-term context retrieval failed: {e}")
            return []
    
    def get_long_term_context(self, query: str) -> Optional[str]:
        """Get context from LightRAG (long-term memory) - raw data only."""
        try:
            response = requests.post(
                f"{self.lightrag_url}/query",
                json={
                    "query": query,
                    "mode": "hybrid",
                    "only_need_context": True,  # Raw context, no LLM processing
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response") or result.get("context")
        except requests.RequestException as e:
            print(f"[WARN] Long-term context retrieval failed: {e}")
            return None
    
    def get_combined_context(self, vl_entries: List[Dict[str, Any]], max_short_term: int = 5) -> Dict[str, Any]:
        """
        Get combined context from all sources.
        
        Args:
            vl_entries: Recent VL processing results
            max_short_term: Max short-term memory entries to retrieve
            
        Returns:
            Combined context dictionary
        """
        # Build query from VL entries
        query_parts = []
        for entry in vl_entries[:3]:  # Use top 3 entries for query
            if entry.get("summary"):
                query_parts.append(entry["summary"])
            if entry.get("speech"):
                query_parts.append(entry["speech"])
        
        query = " ".join(query_parts) if query_parts else "recent events"
        
        # Get contexts
        short_term = self.get_short_term_context(query, limit=max_short_term)
        long_term = self.get_long_term_context(query)
        
        return {
            "vl_entries": vl_entries,
            "short_term_memory": short_term,
            "long_term_context": long_term,
            "query_used": query,
        }


class LLMProcessor:
    """Processes VL data with context using qwen3:30b."""
    
    SYSTEM_PROMPT = """You are a friendly AI companion observing a live video feed. You can see what's happening and may choose to comment, respond, or stay silent.

Your task:
1. Think about what you're observing and whether it warrants a response
2. Decide if you should say something (not everything needs a comment)
3. If you speak, be natural and conversational - talk TO the person, not about them

When to speak:
- When someone talks to you or asks something
- When something interesting, unusual, or noteworthy happens
- When you can offer helpful information or a friendly comment
- When there's a safety concern (high priority)

When to stay silent:
- Mundane activities that don't need commentary
- When the person is busy/focused and doesn't need interruption
- When you've already commented on similar things recently

Output JSON format:
```json
{
    "think": "Your reasoning about the scene and whether to respond",
    "reply": "What you say to the person (empty string if staying silent)",
    "emotion": "neutral|happy|curious|concerned|excited",
    "priority": "low|normal|high|urgent",
    "should_speak": true/false
}
```

Keep replies SHORT (1-2 sentences), warm and natural. Talk like a friend, not a narrator."""

    def __init__(self, ollama_url: str = OLLAMA_URL, model: str = LLM_MODEL):
        self.ollama_url = ollama_url
        self.model = model
    
    def _format_context_prompt(self, context: Dict[str, Any]) -> str:
        """Format context into a prompt for the LLM."""
        parts = []
        
        # Current VL entries
        vl_entries = context.get("vl_entries", [])
        if vl_entries:
            parts.append("## Current Scene(s)")
            for i, entry in enumerate(vl_entries, 1):
                scene = entry.get("scene", "")
                speech = entry.get("speech", "")
                summary = entry.get("summary", "")
                transcription = entry.get("transcription", "")
                
                parts.append(f"\n### Scene {i}")
                if scene:
                    parts.append(f"Visual: {scene}")
                if speech or transcription:
                    parts.append(f"Audio: {speech or transcription}")
                if summary and summary != scene:
                    parts.append(f"Summary: {summary}")
        
        # Short-term memory
        short_term = context.get("short_term_memory", [])
        if short_term:
            parts.append("\n## Recent Memory (Short-term)")
            for mem in short_term[:3]:  # Limit to top 3
                if mem.get("summary"):
                    parts.append(f"- {mem['summary']}")
        
        # Long-term context
        long_term = context.get("long_term_context")
        if long_term:
            parts.append("\n## Background Context (Long-term)")
            # Truncate if too long
            if len(long_term) > 500:
                long_term = long_term[:500] + "..."
            parts.append(long_term)
        
        return "\n".join(parts)
    
    def process(self, context: Dict[str, Any]) -> Optional[LLMOutput]:
        """
        Process context and generate LLM response.
        
        Args:
            context: Combined context from ContextRetriever
            
        Returns:
            LLMOutput or None if failed
        """
        prompt = self._format_context_prompt(context)
        
        # Call Ollama
        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {
                        "num_predict": LLM_MAX_TOKENS,
                        "temperature": LLM_TEMPERATURE,
                        "num_ctx": LLM_CONTEXT_SIZE,
                    },
                },
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as e:
            print(f"[ERROR] LLM request failed: {e}")
            return None
        
        # Parse response
        content = result.get("message", {}).get("content", "")
        
        # Try to extract JSON from response
        try:
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                parts = content.split("```")
                if len(parts) >= 2:
                    content = parts[1].strip()
            
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # If not valid JSON, create structured output from raw text
            parsed = {
                "think": "Unable to parse response",
                "reply": content.strip()[:200] if content.strip() else "",
                "emotion": "neutral",
                "priority": "normal",
                "should_speak": bool(content.strip()),
            }
        
        # Get source clips
        source_clips = [e.get("clip_path", "") for e in context.get("vl_entries", [])]
        
        # Determine should_speak from parsed data
        should_speak = parsed.get("should_speak", bool(parsed.get("reply", "").strip()))
        
        return LLMOutput(
            timestamp=time.time(),
            think=parsed.get("think", ""),
            reply=parsed.get("reply", ""),
            emotion=parsed.get("emotion", "neutral"),
            priority=parsed.get("priority", "normal"),
            should_speak=should_speak,
            source_clips=source_clips,
        )
    
    def direct_query(self, query: str, context: Dict[str, Any] = None) -> Optional[LLMOutput]:
        """
        Process a direct query with optional context.
        
        Args:
            query: User query
            context: Optional context dictionary
            
        Returns:
            LLMOutput or None if failed
        """
        prompt_parts = []
        
        if context:
            prompt_parts.append(self._format_context_prompt(context))
            prompt_parts.append("\n## User Query")
        
        prompt_parts.append(query)
        prompt = "\n".join(prompt_parts)
        
        # Call Ollama
        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {
                        "num_predict": LLM_MAX_TOKENS,
                        "temperature": LLM_TEMPERATURE,
                        "num_ctx": LLM_CONTEXT_SIZE,
                    },
                },
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as e:
            print(f"[ERROR] LLM request failed: {e}")
            return None
        
        content = result.get("message", {}).get("content", "")
        
        # Try to extract JSON
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                parts = content.split("```")
                if len(parts) >= 2:
                    content = parts[1].strip()
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {
                "think": query,
                "reply": content.strip()[:200] if content.strip() else "",
                "emotion": "neutral",
                "priority": "normal",
                "should_speak": bool(content.strip()),
            }
        
        should_speak = parsed.get("should_speak", bool(parsed.get("reply", "").strip()))
        
        return LLMOutput(
            timestamp=time.time(),
            think=parsed.get("think", ""),
            reply=parsed.get("reply", ""),
            emotion=parsed.get("emotion", "neutral"),
            priority=parsed.get("priority", "normal"),
            should_speak=should_speak,
            source_clips=[],
        )


class OutputManager:
    """Manages LLM output storage and retrieval."""
    
    def __init__(self, output_path: Path = LLM_OUTPUT_JSON):
        self.output_path = output_path
        self._ensure_file()
    
    def _ensure_file(self):
        """Ensure output file exists."""
        if not self.output_path.exists():
            self._save({"outputs": [], "last_updated": None})
    
    def _load(self) -> Dict[str, Any]:
        """Load output file."""
        try:
            with open(self.output_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"outputs": [], "last_updated": None}
    
    def _save(self, data: Dict[str, Any]):
        """Save output file."""
        with open(self.output_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def add_output(self, output: LLMOutput):
        """Add a new output entry."""
        data = self._load()
        data["outputs"].append(output.to_dict())
        data["last_updated"] = datetime.now().isoformat()
        
        # Keep only last 100 outputs
        if len(data["outputs"]) > 100:
            data["outputs"] = data["outputs"][-100:]
        
        self._save(data)
    
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get the most recent output."""
        data = self._load()
        outputs = data.get("outputs", [])
        return outputs[-1] if outputs else None
    
    def get_pending_tts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get outputs formatted for TTS."""
        data = self._load()
        outputs = data.get("outputs", [])
        
        # Return most recent outputs in TTS format
        tts_outputs = []
        for output in outputs[-limit:]:
            tts_outputs.append({
                "text": output.get("response_text", ""),
                "emotion": output.get("emotion", "neutral"),
                "priority": output.get("priority", "normal"),
                "timestamp": output.get("timestamp", 0),
            })
        
        return tts_outputs


def process_pending():
    """Process all pending VL results."""
    reader = VLDataReader()
    retriever = ContextRetriever()
    processor = LLMProcessor()
    output_mgr = OutputManager()
    
    unprocessed = reader.get_unprocessed()
    
    if not unprocessed:
        print("[INFO] No pending VL results to process")
        return
    
    print(f"[INFO] Processing {len(unprocessed)} VL result(s)...")
    
    # Get combined context
    context = retriever.get_combined_context(unprocessed)
    
    # Process with LLM
    result = processor.process(context)
    
    if result:
        output_mgr.add_output(result)
        reader.mark_processed(unprocessed)
        
        print(f"[SUCCESS] Generated response:")
        print(f"  Think: {result.think[:100]}..." if len(result.think) > 100 else f"  Think: {result.think}")
        print(f"  Reply: {result.reply if result.reply else '(silent)'}")
        print(f"  Should Speak: {result.should_speak}")
        print(f"  Emotion: {result.emotion}")
        print(f"  Priority: {result.priority}")
    else:
        print("[ERROR] Failed to generate response")


def run_testrun():
    """Test with the latest VL result."""
    reader = VLDataReader()
    retriever = ContextRetriever()
    processor = LLMProcessor()
    
    latest = reader.get_latest()
    
    if not latest:
        print("[ERROR] No VL results found")
        return
    
    print(f"[TEST] Processing latest VL result:")
    print(f"  Clip: {latest.get('clip_path', 'N/A')}")
    print(f"  Summary: {latest.get('summary', 'N/A')}")
    
    # Get context
    context = retriever.get_combined_context([latest])
    
    print(f"\n[CONTEXT] Short-term memory: {len(context.get('short_term_memory', []))} entries")
    print(f"[CONTEXT] Long-term context: {'Yes' if context.get('long_term_context') else 'No'}")
    
    # Process
    print("\n[PROCESSING] Calling LLM...")
    start = time.time()
    result = processor.process(context)
    elapsed = time.time() - start
    
    if result:
        print(f"\n[RESULT] ({elapsed:.1f}s)")
        print(f"  Think: {result.think}")
        print(f"  Reply: {result.reply if result.reply else '(silent)'}")
        print(f"  Should Speak: {result.should_speak}")
        print(f"  Emotion: {result.emotion}")
        print(f"  Priority: {result.priority}")
        
        print("\n[TTS JSON]")
        print(json.dumps(result.to_tts_json(), indent=2))
    else:
        print("[ERROR] Failed to generate response")


def run_query(query: str):
    """Run a direct query with context."""
    retriever = ContextRetriever()
    processor = LLMProcessor()
    
    # Get some context
    reader = VLDataReader()
    recent = reader.get_unprocessed() or ([reader.get_latest()] if reader.get_latest() else [])
    
    context = retriever.get_combined_context(recent[:3]) if recent else None
    
    print(f"[QUERY] {query}")
    if context:
        print(f"[CONTEXT] Using {len(context.get('vl_entries', []))} VL entries")
    
    result = processor.direct_query(query, context)
    
    if result:
        print(f"\n[RESPONSE]")
        print(f"  Think: {result.think}")
        print(f"  Reply: {result.reply if result.reply else '(silent)'}")
        print(f"  Should Speak: {result.should_speak}")
        print(f"  Emotion: {result.emotion}")
    else:
        print("[ERROR] Failed to generate response")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Processing Module")
    parser.add_argument("--process", action="store_true", help="Process pending VL results")
    parser.add_argument("--testrun", action="store_true", help="Test with latest VL result")
    parser.add_argument("--query", type=str, help="Direct query with context")
    parser.add_argument("--status", action="store_true", help="Show processing status")
    
    args = parser.parse_args()
    
    if args.process:
        process_pending()
    elif args.testrun:
        run_testrun()
    elif args.query:
        run_query(args.query)
    elif args.status:
        reader = VLDataReader()
        unprocessed = reader.get_unprocessed()
        latest = reader.get_latest()
        
        print(f"[STATUS]")
        print(f"  Pending VL results: {len(unprocessed)}")
        print(f"  Last processed timestamp: {reader.last_processed_ts}")
        if latest:
            print(f"  Latest VL result: {latest.get('summary', 'N/A')[:50]}...")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
