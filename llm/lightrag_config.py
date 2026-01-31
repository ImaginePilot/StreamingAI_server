#!/usr/bin/env python3
"""
LightRAG Server Configuration and Launcher

Configures LightRAG to use Ollama models:
- Embedding: qwen3-embedding:0.6b
- LLM: qwen3:30b
"""

import os
import subprocess
import sys
from pathlib import Path

# Ollama server
OLLAMA_HOST = "http://192.168.2.6:11434"

# LightRAG settings
LIGHTRAG_HOST = "0.0.0.0"
LIGHTRAG_PORT = 9621
WORKING_DIR = str(Path(__file__).parent / "lightrag_data")

# Model configuration
LLM_MODEL = "gpt-oss:120b"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
EMBEDDING_DIM = 1024

def get_env_config():
    """Get environment variables for LightRAG."""
    return {
        # Ollama LLM binding
        "LLM_BINDING": "ollama",
        "LLM_MODEL": LLM_MODEL,
        "LLM_BINDING_HOST": OLLAMA_HOST,
        
        # Ollama Embedding binding
        "EMBEDDING_BINDING": "ollama", 
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "EMBEDDING_BINDING_HOST": OLLAMA_HOST,
        "EMBEDDING_DIM": str(EMBEDDING_DIM),
        
        # Working directory
        "WORKING_DIR": WORKING_DIR,
        
        # Ollama-specific options (context size)
        "OLLAMA_LLM_NUM_CTX": "65536",
    }

def print_config():
    """Print the configuration."""
    print("LightRAG Configuration:")
    print("=" * 50)
    print(f"  Ollama Host:     {OLLAMA_HOST}")
    print(f"  LLM Model:       {LLM_MODEL}")
    print(f"  Embedding Model: {EMBEDDING_MODEL}")
    print(f"  Embedding Dim:   {EMBEDDING_DIM}")
    print(f"  Working Dir:     {WORKING_DIR}")
    print(f"  Server:          http://{LIGHTRAG_HOST}:{LIGHTRAG_PORT}")
    print("=" * 50)

def start_server():
    """Start the LightRAG API server."""
    os.makedirs(WORKING_DIR, exist_ok=True)
    
    env = os.environ.copy()
    env.update(get_env_config())
    
    print_config()
    print("\nStarting LightRAG server...\n")
    
    # LightRAG uses env vars for model config, CLI for server settings
    cmd = [
        "lightrag-server",
        "--host", LIGHTRAG_HOST,
        "--port", str(LIGHTRAG_PORT),
        "--working-dir", WORKING_DIR,
        "--llm-binding", "ollama",
        "--embedding-binding", "ollama",
    ]
    
    try:
        subprocess.run(cmd, env=env)
    except FileNotFoundError:
        print("lightrag-server not found. Try:")
        print("  uv tool install 'lightrag-hku[api]'")
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LightRAG Configuration")
    parser.add_argument("--start", action="store_true", help="Start the LightRAG server")
    parser.add_argument("--print-env", action="store_true", help="Print environment variables for shell export")
    parser.add_argument("--print-cmd", action="store_true", help="Print the full command with env vars")
    
    args = parser.parse_args()
    
    if args.print_env:
        for k, v in get_env_config().items():
            print(f"export {k}=\"{v}\"")
    elif args.print_cmd:
        env_vars = get_env_config()
        env_str = " \\\n  ".join(f"{k}=\"{v}\"" for k, v in env_vars.items())
        print(f"""{env_str} \\
  lightrag-server \\
    --host {LIGHTRAG_HOST} \\
    --port {LIGHTRAG_PORT} \\
    --working-dir {WORKING_DIR} \\
    --llm-binding ollama \\
    --embedding-binding ollama""")
    elif args.start:
        start_server()
    else:
        print_config()
        print("\nUse --start to launch the server")
        print("Use --print-env to export environment variables")
        print("Use --print-cmd to see the full command")
