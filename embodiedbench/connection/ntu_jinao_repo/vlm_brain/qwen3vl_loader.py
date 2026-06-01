"""Qwen3-VL loading and generation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_qwen3vl_model(config: dict[str, Any]):
    """Load Qwen3-VL with transformers, surfacing dependency/OOM errors clearly."""

    try:
        import torch
        from transformers import AutoProcessor
    except Exception as exc:
        raise RuntimeError(
            "Missing Qwen3-VL inference dependencies. Install vlm_brain/requirements_vlm.txt first."
        ) from exc

    try:
        from transformers import Qwen3VLForConditionalGeneration
    except Exception as exc:
        raise RuntimeError(
            "Your transformers version does not expose Qwen3VLForConditionalGeneration. "
            "Install the latest source version: pip install git+https://github.com/huggingface/transformers"
        ) from exc

    model_name = str(config.get("model_name_or_path") or config.get("base_model"))
    dtype = _resolve_dtype(config.get("torch_dtype", "auto"), torch)
    kwargs: dict[str, Any] = {
        "device_map": config.get("device", "auto"),
        "trust_remote_code": bool(config.get("trust_remote_code", True)),
    }
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype
    else:
        kwargs["torch_dtype"] = "auto"
    if bool(config.get("load_in_4bit", False)):
        kwargs["load_in_4bit"] = True

    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message or "cuda" in message:
            raise RuntimeError(
                "Failed to load Qwen3-VL, likely due to insufficient GPU memory. "
                "Try load_in_4bit=true, a smaller model, or CPU/offload settings."
            ) from exc
        raise
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=bool(config.get("trust_remote_code", True)))
    return model, processor


def load_processor(config: dict[str, Any]):
    try:
        from transformers import AutoProcessor
    except Exception as exc:
        raise RuntimeError("Missing transformers. Install vlm_brain/requirements_vlm.txt first.") from exc
    model_name = str(config.get("model_name_or_path") or config.get("base_model"))
    return AutoProcessor.from_pretrained(model_name, trust_remote_code=bool(config.get("trust_remote_code", True)))


def prepare_messages(image_path: str | None, text_prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if image_path:
        content.append({"type": "image", "image": str(Path(image_path).expanduser())})
    content.append({"type": "text", "text": text_prompt})
    return [{"role": "user", "content": content}]


def generate_response(model, processor, image_path: str | None, prompt: str, config: dict[str, Any]) -> str:
    try:
        import torch
        from qwen_vl_utils import process_vision_info
    except Exception as exc:
        raise RuntimeError("Missing torch or qwen-vl-utils. Install vlm_brain/requirements_vlm.txt first.") from exc

    messages = prepare_messages(image_path, prompt)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generation_kwargs = {
        "max_new_tokens": int(config.get("max_new_tokens", 4096)),
        "do_sample": bool(config.get("do_sample", False)),
        "temperature": float(config.get("temperature", 0.1)),
        "top_p": float(config.get("top_p", 0.9)),
    }
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids_trimmed = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def _resolve_dtype(value: Any, torch_module) -> Any:
    if value in (None, "auto"):
        return "auto"
    value = str(value).lower()
    if value in {"float16", "fp16"}:
        return torch_module.float16
    if value in {"bfloat16", "bf16"}:
        return torch_module.bfloat16
    if value in {"float32", "fp32"}:
        return torch_module.float32
    raise ValueError(f"Unsupported torch_dtype: {value}")
