"""
VisionPerceptionAdapter: bridges QwenVLMPerception → EmbodiedManipulationMemorySystem.

This adapter wraps QwenVLMPerception and translates its structured RecognitionResult
outputs into the (T, P, D) feature array expected by PerceptionInterface, while also
populating working memory with visible objects, scene text, and attribute metadata.

Two usage modes:

  (A) Full adapter mode — call adapter.perceive() directly:
        adapter = VisionPerceptionAdapter(perceiver, memory)
        results = adapter.perceive(image_paths, candidate_labels, current_location="table")
        # Working memory is fully updated; results returned for downstream use.

  (B) Plugin mode — register with the memory system and call memory.run_perception():
        memory.register_perception(adapter)
        feats = memory.run_perception({"image_paths": ["rgb.png"]})
        # Working memory features updated; call update_observation() separately.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from perception_vlm import QwenVLMPerception, RecognitionResult
    _HAS_PERCEPTION = True
except ImportError:
    QwenVLMPerception = None  # type: ignore
    RecognitionResult = None  # type: ignore
    _HAS_PERCEPTION = False

try:
    from embodiedbench.memory_manip.interfaces import PerceptionInterface
    from embodiedbench.memory_manip.agent_memory import EmbodiedManipulationMemorySystem
    _HAS_MEMORY = True
except ImportError:
    PerceptionInterface = object  # type: ignore — fallback for standalone testing
    EmbodiedManipulationMemorySystem = None  # type: ignore
    _HAS_MEMORY = False


# Dimension of the per-patch feature vector (must match what working memory stores)
FEATURE_DIM = 64

# Minimum object detection confidence to include in visible_objects list
MIN_OBJ_CONFIDENCE = 0.40


# ---------------------------------------------------------------------------
# Feature encoding helpers
# ---------------------------------------------------------------------------

def _hash_to_vec(text: str, dim: int) -> np.ndarray:
    """Hash a string into a deterministic float32 vector in [-1, 1]."""
    digest = hashlib.md5(text.encode("utf-8")).digest()
    n_bytes = dim * 4
    extended = (digest * (n_bytes // len(digest) + 1))[:n_bytes]
    arr = np.frombuffer(extended, dtype=np.uint8).astype(np.float32)
    arr = arr.reshape(dim, 4).mean(axis=1)
    return (arr / 127.5 - 1.0).astype(np.float32)


def encode_recognition_results(results: List[Any]) -> np.ndarray:
    """
    Encode a list of RecognitionResult objects into a (T=1, P, D) float32 array.

    P = 1 global scene token + 1 primary-label patch per image + N sub-object patches
    D = FEATURE_DIM (64)

    Encoding layout per patch:
      dims  0–15 : name / label hash (semantic identity)
      dims 16–23 : confidence broadcast (8 dims, value = confidence)
      dims 24–39 : attribute embedding (color+shape+texture hashes)
      dims 40–47 : scene-type hash (8 dims)
      dims 48–63 : uncertainty penalty (−confidence_gap broadcast to 16 dims)

    Parameters
    ----------
    results : list of RecognitionResult from QwenVLMPerception.recognize_batch()

    Returns
    -------
    np.ndarray of shape (1, P, FEATURE_DIM), dtype float32
    """
    patches: List[np.ndarray] = []

    # Global scene token: aggregated hash of all scene descriptions
    all_scenes = " ".join(r.scene for r in results if r.scene)
    global_vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    if all_scenes:
        global_vec[:16] = _hash_to_vec(all_scenes, 16)
    patches.append(global_vec)

    for result in results:
        # Primary label patch
        patch = np.zeros(FEATURE_DIM, dtype=np.float32)
        patch[:16] = _hash_to_vec(result.primary_label or "unknown", 16)
        patch[16:24] = np.full(8, result.confidence, dtype=np.float32)
        attr_text = " ".join(f"{k}:{v}" for k, v in result.attributes.items())
        patch[24:40] = _hash_to_vec(attr_text, 16)
        patch[40:48] = _hash_to_vec(result.scene or "", 8)
        uncertainty_penalty = -(1.0 - result.confidence)
        patch[48:64] = np.full(16, uncertainty_penalty, dtype=np.float32)
        patches.append(patch)

        # Sub-object patches
        for obj_dict in result.objects:
            name = obj_dict.get("name", "unknown")
            conf = float(obj_dict.get("confidence", 0.0))
            evidence = obj_dict.get("evidence", "")
            sub = np.zeros(FEATURE_DIM, dtype=np.float32)
            sub[:16] = _hash_to_vec(name, 16)
            sub[16:24] = np.full(8, conf, dtype=np.float32)
            sub[24:40] = _hash_to_vec(evidence, 16)
            patches.append(sub)

    arr = np.stack(patches, axis=0)   # (P, D)
    return arr[np.newaxis, :, :]      # (T=1, P, D)


# ---------------------------------------------------------------------------
# VisionPerceptionAdapter
# ---------------------------------------------------------------------------

class VisionPerceptionAdapter(PerceptionInterface if _HAS_MEMORY else object):
    """
    Adapts QwenVLMPerception outputs for the EmbodiedManipulationMemorySystem.

    Implements PerceptionInterface so it can be registered via:
        memory.register_perception(adapter)

    Core method is perceive(), which runs the full cycle:
        QwenVLMPerception.recognize_batch()
          → encode_recognition_results() → (T, P, D) array
          → memory.on_perception_features()
          → memory.update_observation()
    """

    def __init__(
        self,
        perceiver: Any,   # QwenVLMPerception instance
        memory: Any,      # EmbodiedManipulationMemorySystem instance
    ) -> None:
        if perceiver is None:
            raise RuntimeError(
                "perceiver is None. Ensure perception_vlm.py is importable and "
                "QwenVLMPerception has been instantiated."
            )
        self._perceiver = perceiver
        self._memory = memory
        self._last_results: List[Any] = []

    # ------------------------------------------------------------------
    # Primary API: full perception → memory update cycle
    # ------------------------------------------------------------------

    def perceive(
        self,
        image_paths: Sequence[str],
        candidate_labels: Optional[List[str]] = None,
        current_location: str = "unknown",
        save_memory: bool = True,
    ) -> List[Any]:
        """
        Run the full perception→memory update cycle.

        Calls QwenVLMPerception.recognize_batch(), encodes the results as a
        (T, P, D) feature array, and pushes both features and structured info
        (visible objects, scene description, object attributes) into working memory.

        Parameters
        ----------
        image_paths      : paths to RGB images captured at the current timestep
        candidate_labels : expected object class names (ranked by historical freq)
        current_location : semantic robot location (e.g., "table", "shelf")
        save_memory      : persist VLM-level long-term memory to disk after batch

        Returns
        -------
        List[RecognitionResult] — the raw perception outputs for downstream use
        """
        results = self._perceiver.recognize_batch(
            image_paths,
            candidate_labels=candidate_labels,
            save_memory=save_memory,
        )
        self._last_results = results

        # Encode and push feature array into memory
        features = encode_recognition_results(results)
        self._memory.on_perception_features(features)

        # Push structured observation into working memory
        visible_objects = _extract_visible_objects(results)
        scene_text = _build_scene_text(results)
        memory_objects = _build_memory_objects(results)

        self._memory.update_observation(
            visible_objects=visible_objects,
            memory_objects=memory_objects,
            text=scene_text,
            current_location=current_location,
        )

        return results

    # ------------------------------------------------------------------
    # PerceptionInterface implementation (plugin mode)
    # ------------------------------------------------------------------

    def extract_features(
        self, observation: Any, *, meta: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        PerceptionInterface implementation for use with memory.run_perception().

        Accepts:
          - list of RecognitionResult objects (already-computed)
          - dict with "image_paths" key
          - list of image path strings
        """
        if isinstance(observation, list) and len(observation) > 0:
            # Already-computed RecognitionResult objects
            if hasattr(observation[0], "primary_label"):
                self._last_results = observation
                return encode_recognition_results(observation)
            # List of file path strings
            image_paths = [str(p) for p in observation]
            results = self._perceiver.recognize_batch(image_paths)
            self._last_results = results
            return encode_recognition_results(results)

        if isinstance(observation, dict):
            image_paths = observation.get("image_paths", [])
            if image_paths:
                results = self._perceiver.recognize_batch(image_paths)
                self._last_results = results
                return encode_recognition_results(results)

        return np.zeros((1, 1, FEATURE_DIM), dtype=np.float32)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def last_results(self) -> List[Any]:
        """Last batch of RecognitionResult objects (read-only copy)."""
        return list(self._last_results)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_visible_objects(results: List[Any]) -> List[str]:
    """Deduplicated, confidence-filtered list of visible object names."""
    seen: set = set()
    visible: List[str] = []
    for r in results:
        # Sub-objects first (more specific)
        for obj in r.objects:
            name = obj.get("name", "").strip()
            conf = float(obj.get("confidence", 0.0))
            if name and conf >= MIN_OBJ_CONFIDENCE and name not in seen:
                seen.add(name)
                visible.append(name)
        # Primary label
        label = r.primary_label.strip()
        if label and label not in ("unknown", "") and label not in seen:
            seen.add(label)
            visible.append(label)
    return visible


def _build_scene_text(results: List[Any]) -> str:
    """Compact scene text summary across all images."""
    parts = []
    for i, r in enumerate(results):
        parts.append(
            f"[img{i+1}] {r.primary_label}(conf={r.confidence:.2f})"
            f" scene={r.scene} reasoning={r.reasoning[:80]}"
        )
    return " | ".join(parts)


def _build_memory_objects(results: List[Any]) -> Dict[str, Any]:
    """
    Build the memory_objects dict for working memory.

    Format: {obj_name: {confidence, attributes/evidence, scene, source}}
    """
    objects: Dict[str, Any] = {}
    for r in results:
        label = r.primary_label.strip()
        if label and label not in ("unknown", ""):
            objects[label] = {
                "confidence": r.confidence,
                "attributes": dict(r.attributes),
                "scene": r.scene,
                "source": "vlm_primary",
            }
        for obj in r.objects:
            name = obj.get("name", "").strip()
            conf = float(obj.get("confidence", 0.0))
            if name and conf >= MIN_OBJ_CONFIDENCE:
                objects[name] = {
                    "confidence": conf,
                    "evidence": obj.get("evidence", ""),
                    "source": "vlm_detected",
                }
    return objects
