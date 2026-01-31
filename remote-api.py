import base64
import json
import os
import tempfile
import time
import threading
import inspect
import traceback
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mlx_vlm import generate, load, prompt_utils, stream_generate
from mlx_vlm.utils import load_config
try:
	from mlx_lm import generate as lm_generate
	from mlx_lm import load as lm_load
	from mlx_lm import stream_generate as lm_stream_generate
except Exception:  # optional dependency
	lm_generate = None
	lm_load = None
	lm_stream_generate = None
try:
	from openai_harmony import (
		HarmonyEncodingName,
		load_harmony_encoding,
		Conversation,
		Message,
		Role,
		SystemContent,
		DeveloperContent,
	)
except Exception:  # optional dependency
	HarmonyEncodingName = None
	load_harmony_encoding = None
	Conversation = None
	Message = None
	Role = None
	SystemContent = None
	DeveloperContent = None


MODEL_ID = os.getenv("MODEL_ID", "mlx-community/Qwen3-VL-30B-A3B-Instruct-8bit")
MODEL_IDS = os.getenv(
	"MODEL_IDS",
	" mlx-community/GLM-4.7-REAP-50-mixed-3-4-bits,mlx-community/MiniMax-M2.1-4bit",
)
LM_MODEL_IDS = os.getenv("LM_MODEL_IDS", "mlx-community/GLM-4.7-REAP-50-mxfp4, mlx-community/GLM-4.7-4bit,mlx-community/GLM-4.7-REAP-50-mixed-3-4-bits,mlx-community/MiniMax-M2.1-4bit")
OSS_MODEL_IDS = os.getenv("OSS_MODEL_IDS", "txgsync/gpt-oss-120b-Derestricted-mxfp4-mlx")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "3000"))
DEBUG_LOG = os.getenv("DEBUG_LOG", "1") == "1"
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "256000"))
THINK_TAGS_MODE = os.getenv("THINK_TAGS_MODE", "auto")
THINK_PREFIX_MODELS = os.getenv("THINK_PREFIX_MODELS", "mlx-community/GLM-4.7-REAP-50-mixed-3-4-bits")

app = FastAPI(title="Qwen3-VL MLX API")

model = None
processor = None
config = None
load_error: Optional[str] = None
current_model_id: Optional[str] = None
current_backend: Optional[str] = None
model_cache: Dict[str, Dict[str, Any]] = {}
PRELOAD_ALL = os.getenv("PRELOAD_ALL", "0") == "1"
GENERATION_LOCK = threading.Lock()


class ChatMessage(BaseModel):
	role: str
	content: str


class GenerateRequest(BaseModel):
	prompt: Optional[str] = None
	messages: Optional[List[ChatMessage]] = None
	system: Optional[str] = None
	image: Optional[str] = None
	image_base64: Optional[str] = None
	max_tokens: int = 100
	temperature: float = 0.0
	top_p: float = 1.0
	resize_shape: Optional[List[int]] = None


class VideoGenerateRequest(BaseModel):
	prompt: str
	messages: Optional[List[ChatMessage]] = None
	system: Optional[str] = None
	video_base64: Optional[str] = None
	video_path: Optional[str] = None
	max_tokens: int = 1000
	temperature: float = 0.0
	top_p: float = 1.0
	num_frames: int = 8  # Number of frames to extract from video
	resize_shape: Optional[List[int]] = None


class GenerateResponse(BaseModel):
	text: str


class OpenAIMessage(BaseModel):
	role: str
	content: Any


class OpenAIChatRequest(BaseModel):
	model: Optional[str] = None
	messages: List[OpenAIMessage]
	tools: Optional[List[Dict[str, Any]]] = None
	tool_choice: Optional[Any] = None
	max_tokens: Optional[int] = None
	temperature: Optional[float] = None
	top_p: Optional[float] = None
	stream: Optional[bool] = False


class OpenAICompletionRequest(BaseModel):
	model: Optional[str] = None
	prompt: str
	tools: Optional[List[Dict[str, Any]]] = None
	tool_choice: Optional[Any] = None
	max_tokens: Optional[int] = None
	temperature: Optional[float] = None
	top_p: Optional[float] = None
	stream: Optional[bool] = False


@app.on_event("startup")
def load_model(model_id: Optional[str] = None):
	global model, processor, config, load_error, current_model_id, current_backend
	try:
		model_id = model_id or MODEL_ID
		if model_id in model_cache:
			bundle = model_cache[model_id]
			model = bundle["model"]
			processor = bundle["processor"]
			config = bundle["config"]
			current_backend = bundle.get("backend", "vlm")
			load_error = None
			current_model_id = model_id
			return
		try:
			if _is_lm_model(model_id):
				model, processor = _lm_load_model(model_id)
				config = {}
				load_error = None
				current_model_id = model_id
				current_backend = "lm"
				model_cache[model_id] = {
					"model": model,
					"processor": processor,
					"config": config,
					"backend": "lm",
				}
				return
			model, processor = load(model_id, lazy=False, trust_remote_code=True)
			config = load_config(model_id, trust_remote_code=True)
			load_error = None
			current_model_id = model_id
			current_backend = "vlm"
			model_cache[model_id] = {
				"model": model,
				"processor": processor,
				"config": config,
				"backend": "vlm",
			}
			return
		except Exception as exc:
			vlm_error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
			# Don't fall back to LM for vision models - they must load as VLM
			if _is_vlm_model(model_id):
				load_error = f"VLM load failed for vision model {model_id}:\n{vlm_error}"
				print(f"[load_model] VLM load failed for {model_id}, not falling back to LM:\n{vlm_error}", flush=True)
				return
			print(f"[load_model] VLM load failed for {model_id}, falling back to LM:\n{vlm_error}", flush=True)
			model, processor = _lm_load_model(model_id)
			config = {}
			load_error = None
			current_model_id = model_id
			current_backend = "lm"
			model_cache[model_id] = {
				"model": model,
				"processor": processor,
				"config": config,
				"backend": "lm",
			}
	except Exception as exc:
		load_error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
		print(f"[load_model] failed for {model_id}\n{load_error}", flush=True)

	if PRELOAD_ALL and not model_cache:
		for mid in _get_model_ids():
			try:
				if _is_lm_model(mid):
					model, processor = _lm_load_model(mid)
					model_cache[mid] = {
						"model": model,
						"processor": processor,
						"config": {},
						"backend": "lm",
					}
					continue
				model, processor = load(mid, lazy=False, trust_remote_code=True)
				config = load_config(mid, trust_remote_code=True)
				model_cache[mid] = {
					"model": model,
					"processor": processor,
					"config": config,
					"backend": "vlm",
				}
			except Exception:
				continue


@app.get("/health")
def health_check():
	return {
		"status": "ok" if load_error is None else "error",
		"model": current_model_id or MODEL_ID,
		"backend": current_backend,
		"available_models": _get_model_ids(),
		"loaded_models": list(model_cache.keys()),
		"model_backends": {k: v.get("backend") for k, v in model_cache.items()},
		"loaded": load_error is None,
		"error": load_error,
	}


@app.get("/v1/models")
def list_models():
	models = _get_model_ids()
	return {
		"object": "list",
		"data": [
			{
				"id": model_id,
				"object": "model",
				"owned_by": "local",
			}
			for model_id in models
		],
	}


def _decode_base64_to_tempfile(image_base64: str) -> str:
	if image_base64.startswith("data:"):
		image_base64 = image_base64.split(",", 1)[-1]
	try:
		data = base64.b64decode(image_base64)
	except Exception as exc:
		raise HTTPException(status_code=400, detail="Invalid image_base64") from exc

	temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
	temp.write(data)
	temp.flush()
	temp.close()
	return temp.name


def _decode_video_to_tempfile(video_base64: str) -> str:
	"""Decode base64 video data to a temporary file."""
	if video_base64.startswith("data:"):
		video_base64 = video_base64.split(",", 1)[-1]
	try:
		data = base64.b64decode(video_base64)
	except Exception as exc:
		raise HTTPException(status_code=400, detail="Invalid video_base64") from exc

	# Detect format from magic bytes
	if data[:4] == b'\x00\x00\x00':
		suffix = ".mp4"
	elif data[:4] == b'RIFF':
		suffix = ".webm"
	else:
		suffix = ".mp4"  # default

	temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
	temp.write(data)
	temp.flush()
	temp.close()
	return temp.name


def _extract_frames_from_video(video_path: str, num_frames: int = 8) -> List[str]:
	"""
	Extract frames from a video file and return list of temporary image paths.
	Uses FFmpeg to extract evenly spaced frames.
	"""
	import subprocess

	# Get video duration
	cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
	result = subprocess.run(cmd, capture_output=True, text=True)
	if result.returncode != 0:
		raise HTTPException(status_code=400, detail=f"Failed to get video duration: {result.stderr}")

	try:
		duration = float(result.stdout.strip())
	except ValueError:
		duration = 1.0

	if duration <= 0:
		duration = 1.0

	# Calculate frame extraction points (evenly spaced, excluding very end)
	frame_times = [duration * i / num_frames for i in range(num_frames)]

	frame_paths = []
	for i, timestamp in enumerate(frame_times):
		output_path = tempfile.mktemp(suffix=f"_frame_{i}.png")
		cmd = [
			"ffmpeg", "-ss", str(timestamp),
			"-i", video_path,
			"-vframes", "1",
			"-q:v", "2",  # High quality
			"-y",  # Overwrite output
			output_path
		]
		subprocess.run(cmd, capture_output=True, check=False)
		if os.path.exists(output_path):
			frame_paths.append(output_path)

	return frame_paths


def _debug_log(message: str) -> None:
	if DEBUG_LOG:
		print(message, flush=True)


def _get_model_ids() -> List[str]:
	ids = [m.strip() for m in MODEL_IDS.split(",") if m.strip()]
	if MODEL_ID and MODEL_ID not in ids:
		ids.insert(0, MODEL_ID)
	return ids


def _is_lm_model(model_id: str) -> bool:
	if model_id in [m.strip() for m in LM_MODEL_IDS.split(",") if m.strip()]:
		return True
	return "glm" in model_id.lower()


def _is_vlm_model(model_id: str) -> bool:
	"""Check if a model is explicitly a vision-language model that should not fall back to LM."""
	vlm_patterns = ["qwen3-vl", "qwen2-vl", "qwen-vl", "llava", "pixtral", "molmo"]
	model_lower = model_id.lower()
	return any(p in model_lower for p in vlm_patterns)


def _is_oss_model(model_id: Optional[str]) -> bool:
	if not model_id:
		return False
	ids = [m.strip() for m in OSS_MODEL_IDS.split(",") if m.strip()]
	return model_id in ids or "gpt-oss" in model_id.lower()


def _lm_load_model(model_id: str):
	if lm_load is None:
		raise RuntimeError("mlx-lm is not installed")
	try:
		return lm_load(model_id, tokenizer_config={"fix_mistral_regex": True})
	except Exception:
		return lm_load(model_id)


def _render_harmony_prefill(messages: List[Dict[str, Any]]) -> Optional[List[int]]:
	if load_harmony_encoding is None or HarmonyEncodingName is None:
		return None
	if Conversation is None or Message is None or Role is None:
		return None
	encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
	h_messages = []
	for m in messages:
		role = m.get("role")
		content = m.get("content", "")
		if role == "system":
			h_messages.append(_harmony_message(Role.SYSTEM, str(content), SystemContent))
			continue
		if role == "developer" and DeveloperContent is not None:
			h_messages.append(_harmony_message(Role.DEVELOPER, str(content), DeveloperContent))
			continue
		if role == "user":
			h_messages.append(Message.from_role_and_content(Role.USER, str(content)))
			continue
		if role == "assistant":
			h_messages.append(Message.from_role_and_content(Role.ASSISTANT, str(content)))
			continue
		# fallback
		h_messages.append(Message.from_role_and_content(Role.USER, str(content)))
	convo = Conversation.from_messages(h_messages)
	return encoding.render_conversation_for_completion(convo, Role.ASSISTANT)


def _harmony_message(role: Any, content: str, content_cls: Optional[Any]):
	if Message is None:
		return None
	if content_cls is None:
		return Message.from_role_and_content(role, content)
	try:
		obj = content_cls.new() if hasattr(content_cls, "new") else content_cls()
		for method_name in ("with_instructions", "with_text", "with_content"):
			method = getattr(obj, method_name, None)
			if callable(method):
				obj = method(content)
				return Message.from_role_and_content(role, obj)
		for method_name in ("set_instructions", "set_text", "set_content"):
			method = getattr(obj, method_name, None)
			if callable(method):
				method(content)
				return Message.from_role_and_content(role, obj)
		return Message.from_role_and_content(role, content)
	except Exception:
		return Message.from_role_and_content(role, content)


def _parse_harmony_output(tokenizer, output_text: str) -> Optional[Dict[str, Any]]:
	if load_harmony_encoding is None or HarmonyEncodingName is None:
		return None
	if Role is None:
		return None
	encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
	try:
		completion_ids = tokenizer.encode(output_text)
	except Exception:
		return None
	try:
		entries = encoding.parse_messages_from_completion_tokens(completion_ids, Role.ASSISTANT)
	except Exception:
		return None
	assistant = None
	for entry in entries:
		data = entry.to_dict() if hasattr(entry, "to_dict") else getattr(entry, "__dict__", {})
		if data.get("role") == "assistant":
			assistant = data
	if not assistant:
		return None
	content = assistant.get("content") or ""
	thinking = assistant.get("thinking") or assistant.get("reasoning")
	if isinstance(content, list):
		parts: List[str] = []
		for item in content:
			if isinstance(item, dict) and item.get("type") == "text":
				text = item.get("text")
				if isinstance(text, str):
					parts.append(text)
			elif isinstance(item, str):
				parts.append(item)
			else:
				parts.append(str(item))
		content = "".join(parts)
	channel_parsed = _extract_channel_content(str(content))
	if channel_parsed:
		content = channel_parsed.get("content", content)
		if channel_parsed.get("thinking") and not thinking:
			thinking = channel_parsed["thinking"]
	result = {"content": content}
	if thinking:
		result["thinking"] = thinking
	return result


def _extract_channel_content(text: str) -> Optional[Dict[str, str]]:
	if "<|channel|>" not in text:
		return None
	marker = "<|channel|>"
	message_marker = "<|message|>"
	channels: Dict[str, str] = {}
	idx = 0
	while True:
		start = text.find(marker, idx)
		if start == -1:
			break
		name_start = start + len(marker)
		name_end = text.find(message_marker, name_start)
		if name_end == -1:
			break
		channel = text[name_start:name_end].strip()
		content_start = name_end + len(message_marker)
		next_start = text.find(marker, content_start)
		if next_start == -1:
			segment = text[content_start:]
			idx = len(text)
		else:
			segment = text[content_start:next_start]
			idx = next_start
		if channel:
			channels[channel] = segment
	if not channels:
		return None
	result: Dict[str, str] = {}
	if "final" in channels:
		result["content"] = channels["final"].strip()
	elif "analysis" in channels:
		result["content"] = channels["analysis"].strip()
	else:
		first_channel = next(iter(channels.values()))
		result["content"] = first_channel.strip()
	if "analysis" in channels:
		result["thinking"] = channels["analysis"].strip()
	return result


def _is_think_prefix_model(model_id: Optional[str]) -> bool:
	if not model_id:
		return False
	models = [m.strip() for m in THINK_PREFIX_MODELS.split(",") if m.strip()]
	return model_id in models


def _needs_think_prefix(prompt: Any, text: Optional[str] = None, model_id: Optional[str] = None) -> bool:
	if THINK_TAGS_MODE == "off":
		return False
	if THINK_TAGS_MODE == "on":
		return True
	if _is_think_prefix_model(model_id):
		return True
	if isinstance(prompt, str) and prompt.rstrip().endswith("<think>"):
		return True
	if text and "</think>" in text and "<think>" not in text:
		return True
	return False


@app.post("/generate", response_model=GenerateResponse)
def generate_text(payload: GenerateRequest):
	_debug_log(f"[generate] in prompt={bool(payload.prompt)} messages={bool(payload.messages)} image={bool(payload.image or payload.image_base64)}")
	global model, processor, config, load_error
	if model is None or processor is None or config is None:
		if load_error is None:
			load_model()
		if load_error is not None:
			raise HTTPException(
				status_code=500,
				detail=f"Model load failed: {load_error}. Update mlx-vlm to a version that supports qwen3_vl_moe or set MODEL_ID to a supported model.",
			)
	if payload.messages is None and not payload.prompt:
		raise HTTPException(
			status_code=400, detail="Provide either prompt or messages."
		)

	backend = current_backend or "vlm"
	image_path = None
	if payload.image_base64:
		image_path = _decode_base64_to_tempfile(payload.image_base64)
	elif payload.image:
		image_path = payload.image
	if backend == "lm" and image_path:
		raise HTTPException(status_code=400, detail="Image inputs are not supported for this model.")

	kwargs = {
		"max_tokens": payload.max_tokens or DEFAULT_MAX_TOKENS,
		"temperature": payload.temperature,
		"top_p": payload.top_p,
	}
	if payload.resize_shape is not None:
		if len(payload.resize_shape) not in [1, 2]:
			raise HTTPException(status_code=400, detail="resize_shape must be 1 or 2 integers")
		kwargs["resize_shape"] = (
			(payload.resize_shape[0],) * 2
			if len(payload.resize_shape) == 1
			else tuple(payload.resize_shape)
		)

	try:
		if payload.messages is not None:
			messages = [
				(m.model_dump() if hasattr(m, "model_dump") else m.dict())
				for m in payload.messages
			]
			if backend == "vlm":
				prompt = prompt_utils.apply_chat_template(
					processor, config, messages, num_images=1
				)
			else:
				prompt = _build_lm_prompt_from_messages(messages)
		else:
			if payload.system:
				messages = [
					{"role": "system", "content": payload.system},
					{"role": "user", "content": payload.prompt},
				]
				if backend == "vlm":
					prompt = prompt_utils.apply_chat_template(
						processor, config, messages, num_images=1
					)
				else:
					prompt = _build_lm_prompt_from_messages(messages)
			else:
				if backend == "vlm":
					prompt = prompt_utils.apply_chat_template(
						processor, config, payload.prompt, num_images=1
					)
				else:
					prompt = payload.prompt

		if backend == "vlm":
			text = generate(
				model,
				processor,
				prompt,
				image=image_path,
				**kwargs,
			)
		else:
			if lm_generate is None:
				raise HTTPException(status_code=500, detail="mlx-lm is not installed.")
			text = lm_generate(model, processor, prompt=prompt, **kwargs)
	finally:
		if payload.image_base64 and image_path:
			try:
				os.remove(image_path)
			except OSError:
				pass

	return GenerateResponse(text=text)


@app.post("/generate/video", response_model=GenerateResponse)
def generate_from_video(payload: VideoGenerateRequest):
	"""
	Generate text from video input using Qwen3-VL Instruct.
	Extracts frames from the video and passes them to the vision-language model.
	"""
	_debug_log(f"[video] in num_frames={payload.num_frames}")
	global model, processor, config, load_error

	if model is None or processor is None or config is None:
		if load_error is None:
			load_model()
		if load_error is not None:
			raise HTTPException(
				status_code=500,
				detail=f"Model load failed: {load_error}. Update mlx-vlm to a version that supports qwen3_vl_moe or set MODEL_ID to a supported model.",
			)

	backend = current_backend or "vlm"
	if backend == "lm":
		raise HTTPException(status_code=400, detail="Video input requires VLM backend (Qwen3-VL). LM models do not support video.")

	# Get video path from base64 or file path
	video_path = None
	frame_paths = []
	try:
		if payload.video_base64:
			video_path = _decode_video_to_tempfile(payload.video_base64)
		elif payload.video_path:
			video_path = payload.video_path
		else:
			raise HTTPException(status_code=400, detail="Provide either video_base64 or video_path.")

		# Extract frames from video
		frame_paths = _extract_frames_from_video(video_path, num_frames=payload.num_frames)
		if not frame_paths:
			raise HTTPException(status_code=400, detail="Failed to extract frames from video.")

		kwargs = {
			"max_tokens": payload.max_tokens or DEFAULT_MAX_TOKENS,
			"temperature": payload.temperature,
			"top_p": payload.top_p,
		}
		if payload.resize_shape is not None:
			if len(payload.resize_shape) not in [1, 2]:
				raise HTTPException(status_code=400, detail="resize_shape must be 1 or 2 integers")
			kwargs["resize_shape"] = (
				(payload.resize_shape[0],) * 2
				if len(payload.resize_shape) == 1
				else tuple(payload.resize_shape)
			)

		# Build messages with images (frames)
		image_contents = []
		for frame_path in frame_paths:
			with open(frame_path, "rb") as f:
				frame_b64 = base64.b64encode(f.read()).decode("utf-8")
				image_contents.append({
					"type": "image_url",
					"image_url": {
						"url": f"data:image/png;base64,{frame_b64}"
					}
				})

		# Build user content with images and prompt
		user_content = payload.prompt
		if image_contents:
			user_content = [
				*image_contents,
				{"type": "text", "text": payload.prompt}
			]

		messages_list = []
		if payload.system:
			messages_list.append({"role": "system", "content": payload.system})
		messages_list.append({"role": "user", "content": user_content})

		prompt = prompt_utils.apply_chat_template(
			processor, config, messages_list, num_images=len(frame_paths)
		)

		text = generate(
			model,
			processor,
			prompt,
			**kwargs,
		)

	finally:
		# Cleanup temporary files
		for frame_path in frame_paths:
			try:
				os.remove(frame_path)
			except OSError:
				pass
		if payload.video_base64 and video_path and video_path.startswith(tempfile.gettempdir()):
			try:
				os.remove(video_path)
			except OSError:
				pass

	return GenerateResponse(text=text)


def _ensure_loaded(model_id: Optional[str] = None):
	global model, processor, config, load_error, current_model_id, current_backend
	requested_model = model_id or MODEL_ID
	if requested_model in model_cache:
		bundle = model_cache[requested_model]
		model = bundle["model"]
		processor = bundle["processor"]
		config = bundle["config"]
		current_backend = bundle.get("backend", "vlm")
		current_model_id = requested_model
		load_error = None
		return
	if current_model_id != requested_model or model is None or processor is None or config is None:
		load_model(requested_model)
		if load_error is not None:
			print(f"[_ensure_loaded] failed for {requested_model}\n{load_error}", flush=True)
			raise HTTPException(
				status_code=500,
				detail=f"Model load failed: {load_error}. Update mlx-vlm to a version that supports qwen3_vl_moe or set MODEL_ID to a supported model.",
			)


def _extract_image_from_messages(messages: List[OpenAIMessage]) -> Optional[str]:
	for message in messages:
		content = message.content
		if isinstance(content, list):
			for part in content:
				if not isinstance(part, dict):
					continue
				if part.get("type") == "image_url":
					image_url = part.get("image_url", {})
					url = image_url.get("url") if isinstance(image_url, dict) else None
					if not url:
						continue
					if url.startswith("data:"):
						return _decode_base64_to_tempfile(url)
					return url
	return None


def _content_to_text(content: Any) -> str:
	if isinstance(content, str):
		return content
	if isinstance(content, list):
		parts: List[str] = []
		for part in content:
			if isinstance(part, dict) and part.get("type") == "text":
				text = part.get("text")
				if isinstance(text, str):
					parts.append(text)
		return "\n".join(parts).strip()
	return str(content)


def _normalize_messages(messages: List[OpenAIMessage]) -> List[Dict[str, Any]]:
	result = []
	for message in messages:
		result.append(
			{
				"role": message.role,
				"content": _content_to_text(message.content),
			}
		)
	return result


def _build_lm_prompt_from_messages(messages: List[Dict[str, Any]]) -> str:
	chat_template = getattr(processor, "chat_template", None)
	if chat_template is not None:
		return processor.apply_chat_template(
			messages,
			add_generation_prompt=True,
			return_dict=False,
			enable_thinking=True,
		)
	return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
	try:
		params = inspect.signature(fn).parameters
	except Exception:
		return kwargs
	allowed = {k for k in params.keys() if k != "self"}
	return {k: v for k, v in kwargs.items() if k in allowed}


def _strip_code_fences(text: str) -> str:
	text = text.strip()
	if text.startswith("```"):
		lines = text.splitlines()
		if len(lines) >= 2:
			lines = lines[1:]
			if lines and lines[-1].strip().startswith("```"):
				lines = lines[:-1]
			text = "\n".join(lines).strip()
	return text


def _try_parse_tool_calls(text: str) -> Optional[List[Dict[str, Any]]]:
	try:
		payload = json.loads(_strip_code_fences(text))
	except Exception:
		return None
	if not isinstance(payload, dict):
		return None
	tool_calls = payload.get("tool_calls")
	if not isinstance(tool_calls, list):
		return None
	for call in tool_calls:
		if isinstance(call, dict):
			call.setdefault("id", f"call_{uuid4().hex}")
			call.setdefault("type", "function")
			function = call.get("function")
			if isinstance(function, dict):
				args = function.get("arguments")
				if isinstance(args, dict):
					function["arguments"] = json.dumps(args)
	return tool_calls


@app.post("/v1/chat/completions")
def openai_chat_completions(payload: OpenAIChatRequest):
	request_id = f"chatcmpl-{uuid4().hex}"
	_debug_log(
		f"[chat] in id={request_id} stream={bool(payload.stream)} messages={len(payload.messages)} tools={bool(payload.tools)}"
	)
	_debug_log(f"[chat] in id={request_id} raw_messages={payload.messages}")
	_ensure_loaded(payload.model)
	backend = current_backend or "vlm"

	image_path = _extract_image_from_messages(payload.messages)
	if backend == "lm" and image_path:
		raise HTTPException(status_code=400, detail="Image inputs are not supported for this model.")
	kwargs = {
		"max_tokens": payload.max_tokens or DEFAULT_MAX_TOKENS,
		"temperature": payload.temperature if payload.temperature is not None else 0.0,
		"top_p": payload.top_p if payload.top_p is not None else 1.0,
	}

	messages = _normalize_messages(payload.messages)
	if backend == "lm" and _is_oss_model(payload.model):
		oss_instruction = (
			"Respond with the final answer only. Do not include analysis, reasoning, or meta commentary. "
			"If a JSON response is requested, output only valid JSON."
		)
		messages = [{"role": "system", "content": oss_instruction}] + messages
	if payload.tools:
		tool_instructions = (
			"You may call tools. If you need a tool, respond with a JSON object: "
			"{\"tool_calls\":[{\"id\":\"call_x\",\"type\":\"function\","
			"\"function\":{\"name\":\"tool_name\",\"arguments\":\"{...}\"}}]}. "
			"Otherwise respond normally. Available tools: "
			+ json.dumps(payload.tools)
		)
		messages = [{"role": "system", "content": tool_instructions}] + messages
	if backend == "vlm":
		prompt = prompt_utils.apply_chat_template(
			processor, config, messages, num_images=1
		)
	else:
		prompt = _build_lm_prompt_from_messages(messages)
		if _is_oss_model(payload.model):
			prefill_ids = _render_harmony_prefill(messages)
			if prefill_ids is not None:
				prompt = prefill_ids
	_debug_log(
		f"[chat] in id={request_id} backend={backend} prompt_type={type(prompt).__name__} prompt_preview={str(prompt)[:200]}"
	)
	_debug_log(f"[chat] in id={request_id} needs_think_prefix={_needs_think_prefix(prompt, model_id=payload.model or MODEL_ID)}")
	needs_think_prefix = _needs_think_prefix(prompt, model_id=payload.model or MODEL_ID)

	if payload.stream:
		def event_stream():
			try:
				GENERATION_LOCK.acquire()
				yield "data: " + json.dumps(
					{
						"id": request_id,
						"object": "chat.completion.chunk",
						"model": payload.model or MODEL_ID,
						"choices": [
							{
								"index": 0,
								"delta": {"role": "assistant"},
								"finish_reason": None,
							}
						],
					}
				) + "\n\n"
				stream_fn = stream_generate if backend == "vlm" else lm_stream_generate
				if stream_fn is None:
					raise HTTPException(status_code=400, detail="stream=true is not supported for this model.")
				if backend == "vlm":
					chunk_iter = stream_fn(
						model,
						processor,
						prompt,
						image=image_path,
						**kwargs,
					)
				else:
					lm_kwargs = _filter_kwargs_for_callable(stream_fn, kwargs)
					chunk_iter = stream_fn(
						model,
						processor,
						prompt,
						**lm_kwargs,
					)
				if backend == "lm" and _is_oss_model(payload.model):
					buffer = []
					for chunk in chunk_iter:
						delta_text = chunk.text if hasattr(chunk, "text") else chunk
						if not delta_text:
							continue
						buffer.append(str(delta_text))
					raw_text = "".join(buffer)
					parsed = _parse_harmony_output(processor, raw_text)
					final_text = parsed.get("content", raw_text) if parsed else raw_text
					if needs_think_prefix and "<think>" not in str(final_text):
						final_text = "<think>" + str(final_text)
					_debug_log(f"[chat] out id={request_id} chunk_len={len(str(final_text))}")
					yield "data: " + json.dumps(
						{
							"id": request_id,
							"object": "chat.completion.chunk",
							"model": payload.model or MODEL_ID,
							"choices": [
								{
									"index": 0,
									"delta": {"content": str(final_text)},
									"finish_reason": None,
								}
							],
						}
					) + "\n\n"
				else:
					think_prefix_emitted = False
					for chunk in chunk_iter:
						delta_text = chunk.text if hasattr(chunk, "text") else chunk
						if not delta_text:
							continue
						if needs_think_prefix and not think_prefix_emitted:
							if str(delta_text).lstrip().startswith("<think>"):
								think_prefix_emitted = True
							else:
								think_prefix_emitted = True
								yield "data: " + json.dumps(
									{
										"id": request_id,
										"object": "chat.completion.chunk",
										"model": payload.model or MODEL_ID,
										"choices": [
											{
												"index": 0,
												"delta": {"content": "<think>"},
												"finish_reason": None,
											}
										],
									}
								) + "\n\n"
						_debug_log(f"[chat] out id={request_id} chunk_len={len(str(delta_text))}")
						yield "data: " + json.dumps(
							{
								"id": request_id,
								"object": "chat.completion.chunk",
								"model": payload.model or MODEL_ID,
								"choices": [
									{
										"index": 0,
										"delta": {"content": str(delta_text)},
										"finish_reason": None,
									}
								],
							}
						) + "\n\n"
				yield "data: " + json.dumps(
					{
						"id": request_id,
						"object": "chat.completion.chunk",
						"model": payload.model or MODEL_ID,
						"choices": [
							{
								"index": 0,
								"delta": {},
								"finish_reason": "stop",
							}
						],
					}
				) + "\n\n"
				yield "data: [DONE]\n\n"
				_debug_log(f"[chat] out id={request_id} done")
			finally:
				try:
					GENERATION_LOCK.release()
				except RuntimeError:
					pass
				if image_path and image_path.startswith(tempfile.gettempdir()):
					try:
						os.remove(image_path)
					except OSError:
						pass
		return StreamingResponse(event_stream(), media_type="text/event-stream")

	try:
		with GENERATION_LOCK:
			if backend == "vlm":
				text = generate(model, processor, prompt, image=image_path, **kwargs)
			else:
				if lm_generate is None:
					raise HTTPException(status_code=500, detail="mlx-lm is not installed.")
				lm_kwargs = _filter_kwargs_for_callable(lm_generate, kwargs)
				text = lm_generate(model, processor, prompt=prompt, **lm_kwargs)
			text = text.text if hasattr(text, "text") else text
			if needs_think_prefix and "<think>" not in str(text):
				text = "<think>" + str(text)
			_debug_log(f"[chat] out id={request_id} raw_text_type={type(text).__name__} raw_text_preview={str(text)[:200]}")
	finally:
		if image_path and image_path.startswith(tempfile.gettempdir()):
			try:
				os.remove(image_path)
			except OSError:
				pass

	_debug_log(f"[chat] out id={request_id} text_len={len(str(text))}")
	tool_calls = _try_parse_tool_calls(str(text)) if payload.tools else None
	message: Dict[str, Any] = {"role": "assistant", "content": str(text)}
	if backend == "lm" and _is_oss_model(payload.model):
		parsed = _parse_harmony_output(processor, str(text))
		_debug_log(f"[chat] out id={request_id} harmony_parsed={parsed}")
		if parsed:
			message = {"role": "assistant", "content": parsed.get("content", "")}
			if parsed.get("thinking"):
				message["thinking"] = parsed["thinking"]
	if tool_calls:
		message = {"role": "assistant", "content": None, "tool_calls": tool_calls}

	return {
		"id": request_id,
		"object": "chat.completion",
		"model": payload.model or MODEL_ID,
		"choices": [
			{
				"index": 0,
				"message": message,
				"finish_reason": "stop",
			}
		],
	}


@app.post("/v1/completions")
def openai_completions(payload: OpenAICompletionRequest):
	request_id = f"cmpl-{uuid4().hex}"
	_debug_log(f"[completion] in id={request_id} stream={bool(payload.stream)}")
	_ensure_loaded(payload.model)
	backend = current_backend or "vlm"
	kwargs = {
		"max_tokens": payload.max_tokens or DEFAULT_MAX_TOKENS,
		"temperature": payload.temperature if payload.temperature is not None else 0.0,
		"top_p": payload.top_p if payload.top_p is not None else 1.0,
	}
	if backend == "vlm":
		prompt = prompt_utils.apply_chat_template(
			processor, config, payload.prompt, num_images=1
		)
	else:
		prompt = payload.prompt
	if payload.stream:
		def event_stream():
			GENERATION_LOCK.acquire()
			yield "data: " + json.dumps(
				{
					"id": request_id,
					"object": "text_completion",
					"model": payload.model or MODEL_ID,
					"choices": [
						{
							"index": 0,
							"text": "",
							"finish_reason": None,
						}
					],
				}
			) + "\n\n"
			stream_fn = stream_generate if backend == "vlm" else lm_stream_generate
			if stream_fn is None:
				raise HTTPException(status_code=400, detail="stream=true is not supported for this model.")
			if backend == "vlm":
				chunk_iter = stream_fn(model, processor, prompt, **kwargs)
			else:
				lm_kwargs = _filter_kwargs_for_callable(stream_fn, kwargs)
				chunk_iter = stream_fn(model, processor, prompt, **lm_kwargs)
			for chunk in chunk_iter:
				delta_text = chunk.text if hasattr(chunk, "text") else chunk
				if not delta_text:
					continue
				_debug_log(f"[completion] out id={request_id} chunk_len={len(str(delta_text))}")
				yield "data: " + json.dumps(
					{
						"id": request_id,
						"object": "text_completion",
						"model": payload.model or MODEL_ID,
						"choices": [
							{
								"index": 0,
								"text": str(delta_text),
								"finish_reason": None,
							}
						],
					}
				) + "\n\n"
			yield "data: " + json.dumps(
				{
					"id": request_id,
					"object": "text_completion",
					"model": payload.model or MODEL_ID,
					"choices": [
						{
							"index": 0,
							"text": "",
							"finish_reason": "stop",
						}
					],
				}
			) + "\n\n"
			yield "data: [DONE]\n\n"
			_debug_log(f"[completion] out id={request_id} done")
			try:
				GENERATION_LOCK.release()
			except RuntimeError:
				pass
		return StreamingResponse(event_stream(), media_type="text/event-stream")

	with GENERATION_LOCK:
		if backend == "vlm":
			text = generate(model, processor, prompt, **kwargs)
		else:
			if lm_generate is None:
				raise HTTPException(status_code=500, detail="mlx-lm is not installed.")
			lm_kwargs = _filter_kwargs_for_callable(lm_generate, kwargs)
			text = lm_generate(model, processor, prompt=prompt, **lm_kwargs)
	text = text.text if hasattr(text, "text") else text
	_debug_log(f"[completion] out id={request_id} text_len={len(str(text))}")
	return {
		"id": request_id,
		"object": "text_completion",
		"model": payload.model or MODEL_ID,
		"choices": [
			{
				"index": 0,
				"text": str(text),
				"finish_reason": "stop",
			}
		],
	}


if __name__ == "__main__":
	import argparse
	import uvicorn

	parser = argparse.ArgumentParser(description="Qwen3-VL MLX API")
	parser.add_argument("--benchmark", action="store_true", help="Run a quick TPS benchmark")
	args = parser.parse_args()

	if args.benchmark:
		load_model()
		if load_error is not None or model is None or processor is None or config is None:
			print("Model load failed during benchmark:")
			print(load_error or "Unknown error")
			raise SystemExit(1)
		prompt = prompt_utils.apply_chat_template(
			processor, config, "Generate a poem introducing yourself", num_images=1
		)
		start = time.perf_counter()
		result = generate(
			model,
			processor,
			prompt,
			max_tokens=200,
			temperature=0.0,
			top_p=1.0,
		)
		text = result.text if hasattr(result, "text") else result
		elapsed = max(time.perf_counter() - start, 1e-6)
		tokenizer = getattr(processor, "tokenizer", None)
		if tokenizer is not None:
			try:
				output_tokens = len(tokenizer.encode(str(text)))
			except Exception:
				output_tokens = len(str(text).split())
		else:
			output_tokens = len(str(text).split())
		tps = output_tokens / elapsed
		print("=== Benchmark ===")
		print(f"Prompt: Generate a poem introducing yourself")
		print(f"Output tokens: {output_tokens}")
		print(f"Elapsed: {elapsed:.3f}s")
		print(f"Tokens/sec: {tps:.2f}")
		print("=== Output ===")
		print(text)
	else:
		uvicorn.run("api:app", host=HOST, port=PORT)
