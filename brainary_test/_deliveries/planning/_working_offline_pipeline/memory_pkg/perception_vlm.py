from __future__ import annotations

import argparse
import json
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    from qwen_vl_utils import process_vision_info
except ImportError:  # pragma: no cover - runtime dependency hint
    process_vision_info = None

try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, pipeline
except ImportError:  # pragma: no cover - runtime dependency hint
    AutoProcessor = None
    Qwen2_5_VLForConditionalGeneration = None
    pipeline = None


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MEMORY_PATH = Path("perception_vlm_memory.json")
DEFAULT_AUXILIARY_IMAGE_SIZE = 768
DEFAULT_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

SYSTEM_PROMPT = """
你是一个用于图像识别的视觉语言模型助手。请优先依据图像证据回答，
不要把不确定的类别说成确定事实。输出必须是 JSON，不要添加 Markdown。
""".strip()


# 说明：
# 本算法基于Qwen2.5-VL-3B-Instruct 模型实现以下相关的能力：
# 1. 跨模态关联：模型把图像 token 与文本 token 放在同一对话/生成框架中推理。
# 2. 提取不同低维度的语义特征，如轮廓/颜色/纹理/深度等，进行语义动态融合
# 3. 动态分辨率选取和适配，针对不同样本选择不同分辨率
# 4. 多层次视觉结构：底层视觉编码器提取不同层次的局部 patch/区域信息，语言模型层再做语义级推理。
# 5. 短时记忆：同一次对话上下文、当前图像、多轮提示中的中间观察都可作为工作记忆。
# 6. 长时记忆：从经验学习，预训练和指令微调阶段已经从大规模图文经验中学习。

@dataclass
class RecognitionResult:
    """单张图像的结构化识别结果。"""

    image_path: str
    primary_label: str
    confidence: float
    objects: List[Dict[str, Any]] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    scene: str = ""
    reasoning: str = ""
    uncertainty: str = ""
    raw_response: str = ""


@dataclass
class PerceptionMemory:
    """不微调条件下的外部记忆系统。"""

    memory_path: Path = DEFAULT_MEMORY_PATH
    long_term: Dict[str, Any] = field(default_factory=lambda: {"records": [], "label_stats": {}})
    short_term: Dict[str, Any] = field(default_factory=dict)

    def load(self) -> None:
        if not self.memory_path.exists():
            return
        try:
            self.long_term = json.loads(self.memory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"警告：长期记忆文件读取失败，将使用空记忆。原因：{exc}")

    def save(self) -> None:
        self.memory_path.write_text(
            json.dumps(self.long_term, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def reset_short_term(self) -> None:
        self.short_term = {}

    def remember(self, result: RecognitionResult) -> None:
        """把高置信结果写入外部长期记忆，用作以后推理的经验提示。"""

        record = asdict(result)
        self.long_term.setdefault("records", []).append(record)
        stats = self.long_term.setdefault("label_stats", {})
        label = result.primary_label.strip()
        if label:
            item = stats.setdefault(label, {"count": 0, "avg_confidence": 0.0})
            old_count = item["count"]
            item["count"] = old_count + 1
            item["avg_confidence"] = round(
                (item["avg_confidence"] * old_count + result.confidence) / item["count"],
                4,
            )

    def rank_candidate_labels(self, labels: Sequence[str]) -> List[str]:
        """用历史经验给候选标签排序；无历史时保持原顺序。"""

        stats = self.long_term.get("label_stats", {})

        def score(label: str) -> float:
            item = stats.get(label, {})
            return float(item.get("count", 0)) + float(item.get("avg_confidence", 0.0))

        return sorted(labels, key=score, reverse=True)

    def build_experience_hint(self, top_k: int = 8) -> str:
        """把长期记忆压缩成短提示，避免把大量历史内容塞进上下文。"""

        stats = self.long_term.get("label_stats", {})
        if not stats:
            return "暂无历史经验。"
        ranked = sorted(
            stats.items(),
            key=lambda item: (item[1].get("count", 0), item[1].get("avg_confidence", 0.0)),
            reverse=True,
        )[:top_k]
        return "历史高频类别：" + "；".join(
            f"{label}(次数={meta.get('count', 0)}, 平均置信={meta.get('avg_confidence', 0.0)})"
            for label, meta in ranked
        )


class QwenVLMPerception:
    """基于 Qwen2.5-VL-3B-Instruct 的直接图像识别器。"""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        memory_path: Path | str = DEFAULT_MEMORY_PATH,
        device: str = DEVICE,
        max_new_tokens: int = 512,
        use_auxiliary_views: bool = True,
        enable_depth: bool = False,
        depth_model_id: Optional[str] = None,
        auxiliary_image_size: int = DEFAULT_AUXILIARY_IMAGE_SIZE,
    ) -> None:
        if AutoProcessor is None or Qwen2_5_VLForConditionalGeneration is None:
            raise RuntimeError(
                "缺少 transformers。请先安装：pip install -U transformers accelerate"
            )
        if process_vision_info is None:
            raise RuntimeError(
                "缺少 qwen-vl-utils。请先安装：pip install qwen-vl-utils"
            )

        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.use_auxiliary_views = use_auxiliary_views
        self.enable_depth = enable_depth
        self.depth_model_id = depth_model_id or DEFAULT_DEPTH_MODEL_ID
        self.auxiliary_image_size = auxiliary_image_size
        self.depth_estimator = None
        self.depth_view_description = "Depth auxiliary view is disabled."
        self.memory = PerceptionMemory(Path(memory_path))
        self.memory.load()

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if not torch.cuda.is_available():
            self.model.to(device)
        self.model.eval()

    def recognize_batch(
        self,
        image_paths: Iterable[str],
        candidate_labels: Optional[Sequence[str]] = None,
        save_memory: bool = True,
    ) -> List[RecognitionResult]:
        """批量识别图像，并用短期记忆保存本批次上下文。"""

        paths = [str(Path(path)) for path in image_paths]
        self.memory.reset_short_term()
        self.memory.short_term["current_batch"] = paths

        labels = list(candidate_labels or [])
        if labels:
            labels = self.memory.rank_candidate_labels(labels)
        self.memory.short_term["candidate_labels"] = labels

        results = [self.recognize_one(path, labels) for path in paths]

        # 不微调增强：批内复核。让模型看到第一轮结果，再对低置信/候选冲突样本复查。
        # 这相当于显式短时记忆，不改变模型参数，但能减少单轮生成的偶然错误。
        reviewed = [self.review_result(result, labels) for result in results]

        if save_memory:
            for result in reviewed:
                if result.confidence >= 0.70:
                    self.memory.remember(result)
            self.memory.save()

        return reviewed

    def recognize_one(
        self,
        image_path: str,
        candidate_labels: Optional[Sequence[str]] = None,
    ) -> RecognitionResult:
        """单张图像识别。候选标签可为空，模型会开放式识别。"""

        self._validate_image(image_path)
        prompt = self._build_recognition_prompt(candidate_labels)
        raw = self._generate(image_path, prompt)
        return self._parse_result(image_path, raw)

    def review_result(
        self,
        result: RecognitionResult,
        candidate_labels: Optional[Sequence[str]] = None,
    ) -> RecognitionResult:
        """基于第一轮结果做一次自检复核，不需要微调。"""

        if result.confidence >= 0.85 and result.primary_label:
            return result

        prompt = self._build_review_prompt(result, candidate_labels)
        raw = self._generate(result.image_path, prompt)
        reviewed = self._parse_result(result.image_path, raw)
        reviewed.raw_response = f"FIRST_PASS:\n{result.raw_response}\n\nREVIEW:\n{raw}"

        # 如果复核更不确定，则保留原结果，并记录不确定性。
        if reviewed.confidence + 0.05 < result.confidence:
            result.uncertainty = (
                result.uncertainty + "；复核结果置信度更低，保留第一轮判断"
            ).strip("；")
            return result
        return reviewed

    def _build_recognition_prompt(self, candidate_labels: Optional[Sequence[str]]) -> str:
        experience_hint = self.memory.build_experience_hint()
        labels_text = "、".join(candidate_labels or [])
        label_instruction = (
            f"候选类别按历史经验排序如下：{labels_text}。"
            "如图像证据支持，请优先从候选类别中选择；如不支持，可以给出新类别。"
            if labels_text
            else "没有预设候选类别，请开放式识别主要对象或场景。"
        )
        return f"""
任务：直接识别图像中的主要对象/类别，并列出关键可见对象。

模型本身可用能力：
- 跨模态关联：将图像内容和文本类别/问题共同推理。
- 多层次视觉理解：从局部细节、对象、场景到语义类别逐层判断。
- 短时记忆：当前提示、候选标签和本轮观察会作为上下文参与推理。
- 经验学习：模型预训练/指令微调中已学习大量图文经验。

外部非微调增强：
- 历史经验提示：{experience_hint}
- {label_instruction}

请只输出 JSON，字段如下：
{{
  "primary_label": "最可能类别或主要对象",
  "confidence": 0.0到1.0之间的数字,
  "objects": [
    {{"name": "对象名", "confidence": 0.0到1.0, "evidence": "可见证据"}}
  ],
  "attributes": {{
    "color": "主要颜色/不确定",
    "shape": "形状/不确定",
    "texture": "纹理/不确定"
  }},
  "scene": "场景描述",
  "reasoning": "简短说明依据哪些可见证据判断",
  "uncertainty": "不确定点；若无则为空字符串"
}}
""".strip()

    def _build_review_prompt(
        self,
        result: RecognitionResult,
        candidate_labels: Optional[Sequence[str]],
    ) -> str:
        labels_text = "、".join(candidate_labels or [])
        return f"""
请对上一轮图像识别结果做视觉证据复核，重点检查：
1. 主要类别是否真的被图像支持。
2. 候选类别中是否有更合适的标签。
3. 置信度是否过高或过低。

候选类别：{labels_text if labels_text else "无"}
上一轮结果：
{json.dumps(asdict(result), ensure_ascii=False, indent=2)}

请仍然只输出同样结构的 JSON。
""".strip()

    def _generate(self, image_path: str, prompt: str) -> str:
        with tempfile.TemporaryDirectory(prefix="qwen_vlm_aux_") as aux_dir:
            content = [{"type": "image", "image": image_path}]
            prompt_text = prompt
            if self.use_auxiliary_views:
                auxiliary_images = self._prepare_auxiliary_view_paths(
                    image_path,
                    Path(aux_dir),
                )
                content.extend(
                    {"type": "image", "image": str(auxiliary_image)}
                    for auxiliary_image in auxiliary_images
                )
                prompt_text = self._build_multiview_prompt(prompt)

            content.append({"type": "text", "text": prompt_text})
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )

            generated_ids_trimmed = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return output_text.strip()

    def _build_multiview_prompt(self, prompt: str) -> str:
        depth_instruction = (
            f"\n4. Depth-estimation view. {self.depth_view_description}"
            if self.enable_depth
            else ""
        )
        return f"""
You will receive multiple images of the same sample:
1. Original RGB image. Use this as the primary evidence.
2. Contour/edge view. Use it to check object outline and boundary evidence.
3. Texture-enhanced view. Use it to check surface texture and fine details.
{depth_instruction}

Only trust auxiliary views when they are consistent with the original image.
Do not classify from an auxiliary view alone.

{prompt}
""".strip()

    def _prepare_auxiliary_view_paths(
        self,
        image_path: str,
        output_dir: Path,
    ) -> List[Path]:
        image = self._load_auxiliary_base_image(image_path)
        auxiliary_views = [
            ("contour.png", self._create_contour_view(image)),
            ("texture.png", self._create_texture_view(image)),
        ]
        if self.enable_depth:
            auxiliary_views.append(("depth.png", self._create_depth_view(image)))
        output_paths: List[Path] = []
        for filename, auxiliary_image in auxiliary_views:
            output_path = output_dir / filename
            auxiliary_image.save(output_path)
            output_paths.append(output_path)
        return output_paths

    def _load_auxiliary_base_image(self, image_path: str) -> Image.Image:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.auxiliary_image_size > 0:
                image = ImageOps.contain(
                    image,
                    (self.auxiliary_image_size, self.auxiliary_image_size),
                    method=Image.Resampling.LANCZOS,
                )
            return image

    @staticmethod
    def _create_contour_view(image: Image.Image) -> Image.Image:
        gray = image.convert("L")
        contour = gray.filter(ImageFilter.FIND_EDGES)
        contour = ImageOps.autocontrast(contour)
        return contour.convert("RGB")

    @staticmethod
    def _create_texture_view(image: Image.Image) -> Image.Image:
        texture = image.filter(ImageFilter.DETAIL).filter(ImageFilter.SHARPEN)
        texture = ImageEnhance.Contrast(texture).enhance(1.6)
        return texture.convert("RGB")

    def _create_depth_view(self, image: Image.Image) -> Image.Image:
        depth_image = self._create_model_depth_view(image)
        self.depth_view_description = (
            f"The fourth image is a monocular depth-estimation result from {self.depth_model_id}."
        )
        return depth_image

    def _create_model_depth_view(self, image: Image.Image) -> Image.Image:
        if pipeline is None:
            raise RuntimeError(
                "Depth auxiliary view requires transformers pipeline. "
                "Please install transformers with depth-estimation support."
            )
        try:
            if self.depth_estimator is None:
                self.depth_estimator = pipeline(
                    task="depth-estimation",
                    model=self.depth_model_id,
                    device=0 if torch.cuda.is_available() else -1,
                )
            result = self.depth_estimator(image)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load or run depth model {self.depth_model_id!r}. "
                "Transformers will auto-download this model when network access is available."
            ) from exc

        depth = result.get("depth") if isinstance(result, dict) else None
        if not isinstance(depth, Image.Image):
            raise RuntimeError(
                f"Depth model {self.depth_model_id!r} did not return a PIL depth image."
            )
        return ImageOps.autocontrast(depth.convert("L")).convert("RGB")

    @staticmethod
    def _validate_image(image_path: str) -> None:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图像不存在：{image_path}")
        with Image.open(path) as image:
            image.verify()

    @staticmethod
    def _parse_result(image_path: str, raw: str) -> RecognitionResult:
        data = extract_json_object(raw)
        if not data:
            return RecognitionResult(
                image_path=image_path,
                primary_label="unknown",
                confidence=0.0,
                uncertainty="模型未返回可解析 JSON",
                raw_response=raw,
            )

        confidence = clamp_float(data.get("confidence", 0.0))
        objects = data.get("objects", [])
        if not isinstance(objects, list):
            objects = []
        attributes = data.get("attributes", {})
        if not isinstance(attributes, dict):
            attributes = {}

        return RecognitionResult(
            image_path=image_path,
            primary_label=str(data.get("primary_label", "unknown")),
            confidence=confidence,
            objects=objects,
            attributes=attributes,
            scene=str(data.get("scene", "")),
            reasoning=str(data.get("reasoning", "")),
            uncertainty=str(data.get("uncertainty", "")),
            raw_response=raw,
        )


def extract_json_object(text: str) -> Dict[str, Any]:
    """从模型输出中提取 JSON；兼容偶尔出现的额外文字。"""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def clamp_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def run_demo(
    image_paths: Sequence[str],
    candidate_labels: Optional[Sequence[str]] = None,
    memory_path: Path | str = DEFAULT_MEMORY_PATH,
    use_auxiliary_views: bool = True,
    enable_depth: bool = False,
    depth_model_id: Optional[str] = None,
    auxiliary_image_size: int = DEFAULT_AUXILIARY_IMAGE_SIZE,
) -> List[RecognitionResult]:
    recognizer = QwenVLMPerception(
        memory_path=memory_path,
        use_auxiliary_views=use_auxiliary_views,
        enable_depth=enable_depth,
        depth_model_id=depth_model_id,
        auxiliary_image_size=auxiliary_image_size,
    )
    results = recognizer.recognize_batch(image_paths, candidate_labels=candidate_labels)
    print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="图像识别推理")
    parser.add_argument("images", nargs="+", help="待识别图像路径")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="可选候选类别；提供后会做跨模态候选识别与复核",
    )
    parser.add_argument(
        "--memory",
        default=str(DEFAULT_MEMORY_PATH),
        help="外部长期记忆 JSON 路径",
    )
    parser.add_argument(
        "--no-auxiliary-views",
        action="store_true",
        help="Use only the original image, without contour/texture/depth auxiliary views.",
    )
    parser.add_argument(
        "--enable-depth",
        default=True,
        help=f"Enable depth-estimation auxiliary view; downloads {DEFAULT_DEPTH_MODEL_ID} if needed.",
    )
    parser.add_argument(
        "--depth-model",
        default=None,
        help="Optional transformers depth-estimation model id; also enables the depth view.",
    )
    parser.add_argument(
        "--auxiliary-size",
        type=int,
        default=DEFAULT_AUXILIARY_IMAGE_SIZE,
        help="Maximum side length of generated auxiliary views; <=0 keeps original size.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(
        args.images,
        candidate_labels=args.labels,
        memory_path=args.memory,
        use_auxiliary_views=not args.no_auxiliary_views,
        enable_depth=args.enable_depth or args.depth_model is not None,
        depth_model_id=args.depth_model,
        auxiliary_image_size=args.auxiliary_size,
    )
