from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

try:
    import clip
except ImportError:
    clip = None

try:
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor
except ImportError:
    AutoImageProcessor = None
    AutoModel = None
    AutoProcessor = None


# -------------------------- 1. 可配置参数 --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# backbone 可选："clip"、"siglip"、"dino"、"hybrid"
# hybrid 会优先用 SigLIP 做跨模态文本-图像关联，用 DINO/CLIP 提供图像高层特征。
BACKBONE_NAME = "hybrid"
CLIP_MODEL_NAME = "ViT-B/32"
SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"
DINO_MODEL_NAME = "facebook/dinov2-base"

# 相似度阈值，可根据具体场景调整。
SIM_THRESHOLD_SYNONYM = 0.85
SIM_THRESHOLD_ANTONYM = 0.20
SIM_THRESHOLD_METAPHOR = 0.60
SIM_THRESHOLD_CROSS_MODAL = 0.18
SIM_THRESHOLD_OBJECT_RELATION = 0.55
CLUSTER_THRESHOLD_HIERARCHY = 0.50
CLUSTER_THRESHOLD_DOMAIN = 0.40


# -------------------------- 2. 数据结构：知识图谱与记忆 --------------------------
@dataclass
class GraphNode:
    """图像内知识图谱的节点，以物体/候选概念为单位。"""

    node_id: str
    label: str
    image_path: str
    node_type: str = "object"
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """图像内知识图谱的边，用于表达物体之间或物体-图像之间的关系。"""

    source: str
    target: str
    relation: str
    weight: float
    evidence: str


@dataclass
class KnowledgeGraph:
    """图像内知识图谱容器。"""

    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[GraphEdge] = field(default_factory=list)

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)

    def print_graph(self) -> None:
        print("\n===== 图像内知识图谱（object 为 node，relation 为 edge）=====")
        for node in self.nodes.values():
            attrs = ", ".join(f"{k}={v}" for k, v in node.attributes.items())
            print(f"节点[{node.node_id}] | 标签={node.label} | 图像={node.image_path} | 属性: {attrs}")
        for edge in self.edges:
            print(
                f"边[{edge.source}] -> [{edge.target}] | 关系={edge.relation} "
                f"| 权重={edge.weight:.4f} | 证据={edge.evidence}"
            )


@dataclass
class MemorySystem:
    """
    三类 memory：
    - long_term_memory：长期保留的图像内知识图谱与稳定概念统计。
    - experience_memory：模型在多次感知中积累的经验，如高频关系、低层特征摘要。
    - short_term_memory：当前输入样本的中间结果与有用信息。
    """

    long_term_memory: Dict[str, Any] = field(default_factory=lambda: {"graphs": {}, "concept_stats": {}})
    experience_memory: Dict[str, Any] = field(default_factory=lambda: {"relations": [], "feature_summaries": []})
    short_term_memory: Dict[str, Any] = field(default_factory=dict)

    def reset_short_term(self) -> None:
        self.short_term_memory = {}

    def update_long_term_graph(self, image_path: str, graph: KnowledgeGraph) -> None:
        self.long_term_memory["graphs"][image_path] = graph
        for node in graph.nodes.values():
            stats = self.long_term_memory["concept_stats"].setdefault(node.label, {"count": 0})
            stats["count"] += 1

    def update_experience(self, graph: KnowledgeGraph, feature_summary: Dict[str, Any]) -> None:
        for edge in graph.edges:
            self.experience_memory["relations"].append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "weight": edge.weight,
                    "evidence": edge.evidence,
                }
            )
        self.experience_memory["feature_summaries"].append(feature_summary)

    def print_memory(self) -> None:
        print("\n===== Long-term Memory =====")
        print(f"已存图谱数量: {len(self.long_term_memory['graphs'])}")
        print(f"概念统计: {self.long_term_memory['concept_stats']}")
        print("\n===== Experience Memory =====")
        print(f"累计关系数量: {len(self.experience_memory['relations'])}")
        print(f"累计特征摘要数量: {len(self.experience_memory['feature_summaries'])}")
        print("\n===== Short-term Memory =====")
        for key, value in self.short_term_memory.items():
            print(f"{key}: {value}")


# -------------------------- 3. Backbone 封装：CLIP / SigLIP / DINO --------------------------
def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """对特征做 L2 归一化，便于余弦相似度计算。"""

    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class PerceptionBackbone:
    """统一封装不同视觉/跨模态 backbone 的编码接口。"""

    def __init__(self, backbone_name: str = BACKBONE_NAME):
        self.backbone_name = backbone_name.lower()
        self.clip_model = None
        self.clip_preprocess = None
        self.siglip_model = None
        self.siglip_processor = None
        self.dino_model = None
        self.dino_processor = None
        self._load_models()

    def _load_models(self) -> None:
        if self.backbone_name in {"clip", "hybrid"} and clip is not None:
            self.clip_model, self.clip_preprocess = clip.load(CLIP_MODEL_NAME, device=device)
            self.clip_model.eval()

        if self.backbone_name in {"siglip", "hybrid"} and AutoProcessor is not None and AutoModel is not None:
            try:
                self.siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
                self.siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).to(device).eval()
            except Exception as exc:
                print(f"警告：SigLIP 模型加载失败，将跳过 SigLIP。原因: {exc}")

        if self.backbone_name in {"dino", "hybrid"} and AutoImageProcessor is not None and AutoModel is not None:
            try:
                self.dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
                self.dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME).to(device).eval()
            except Exception as exc:
                print(f"警告：DINO 模型加载失败，将跳过 DINO。原因: {exc}")

        if self.clip_model is None and self.siglip_model is None and self.dino_model is None:
            raise RuntimeError(
                "没有可用 backbone。请安装 openai-clip 或 transformers，并确保模型权重可被加载。"
            )

    def encode_text(self, texts: List[str]) -> Tuple[Optional[torch.Tensor], List[str], str]:
        """优先使用 SigLIP/CLIP 提取文本特征；DINO 仅支持图像，不参与文本编码。"""

        if not texts:
            return None, [], ""

        if self.siglip_model is not None and self.siglip_processor is not None:
            inputs = self.siglip_processor(text=texts, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                features = self.siglip_model.get_text_features(**inputs)
            return normalize_features(features), texts.copy(), "siglip"

        if self.clip_model is not None and clip is not None:
            tokens = clip.tokenize(texts).to(device)
            with torch.no_grad():
                features = self.clip_model.encode_text(tokens)
            return normalize_features(features.float()), texts.copy(), "clip"

        print("警告：当前 backbone 不支持文本编码，无法计算文本相关关系。")
        return None, [], ""

    def encode_images(self, image_paths: List[str]) -> Tuple[Optional[torch.Tensor], List[str], str]:
        """提取整图高层特征，优先使用 DINO，其次 SigLIP/CLIP。"""

        images, valid_paths = load_images(image_paths)
        if not images:
            return None, [], ""

        if self.dino_model is not None and self.dino_processor is not None:
            inputs = self.dino_processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.dino_model(**inputs)
                features = outputs.last_hidden_state[:, 0, :]
            return normalize_features(features), valid_paths, "dino"

        if self.siglip_model is not None and self.siglip_processor is not None:
            inputs = self.siglip_processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                features = self.siglip_model.get_image_features(**inputs)
            return normalize_features(features), valid_paths, "siglip"

        if self.clip_model is not None and self.clip_preprocess is not None:
            image_batch = torch.cat([self.clip_preprocess(img).unsqueeze(0) for img in images]).to(device)
            with torch.no_grad():
                features = self.clip_model.encode_image(image_batch)
            return normalize_features(features.float()), valid_paths, "clip"

        return None, [], ""

    def encode_cross_modal_images(self, image_paths: List[str]) -> Tuple[Optional[torch.Tensor], List[str], str]:
        """跨模态关联需要图像和文本在同一嵌入空间，优先使用 SigLIP，其次 CLIP。"""

        images, valid_paths = load_images(image_paths)
        if not images:
            return None, [], ""

        if self.siglip_model is not None and self.siglip_processor is not None:
            inputs = self.siglip_processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                features = self.siglip_model.get_image_features(**inputs)
            return normalize_features(features), valid_paths, "siglip"

        if self.clip_model is not None and self.clip_preprocess is not None:
            image_batch = torch.cat([self.clip_preprocess(img).unsqueeze(0) for img in images]).to(device)
            with torch.no_grad():
                features = self.clip_model.encode_image(image_batch)
            return normalize_features(features.float()), valid_paths, "clip"

        print("警告：当前 backbone 仅有 DINO，无法直接做图像-文本跨模态相似度。")
        return None, [], ""


def load_images(image_paths: Iterable[str]) -> Tuple[List[Image.Image], List[str]]:
    """读取图像并返回 PIL 图像列表与有效路径列表。"""

    images: List[Image.Image] = []
    valid_paths: List[str] = []
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
            images.append(image)
            valid_paths.append(img_path)
        except Exception as exc:
            print(f"警告：图像 {img_path} 无法读取，已跳过。原因: {exc}")
    return images, valid_paths


# -------------------------- 4. Low-level 与 High-level 特征 --------------------------
def extract_low_level_features(image_path: str) -> Dict[str, Any]:
    """
    提取低层视觉特征：
    - color_hist：RGB 颜色直方图。
    - dominant_color：平均主色。
    - texture_stats：基于灰度边缘响应的纹理统计。
    """

    image = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.asarray(image).astype(np.float32) / 255.0
    color_hist = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=16, range=(0.0, 1.0), density=True)
        color_hist.extend(hist.tolist())

    gray = image.convert("L")
    edge = gray.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.asarray(edge).astype(np.float32) / 255.0
    texture_stats = {
        "edge_mean": round(float(edge_arr.mean()), 4),
        "edge_std": round(float(edge_arr.std()), 4),
        "edge_energy": round(float(np.mean(edge_arr**2)), 4),
    }

    return {
        "color_hist": np.asarray(color_hist, dtype=np.float32),
        "dominant_color": tuple(np.round(arr.mean(axis=(0, 1)), 4).tolist()),
        "texture_stats": texture_stats,
    }


def extract_image_feature_bank(
    backbone: PerceptionBackbone, image_paths: List[str]
) -> Dict[str, Dict[str, Any]]:
    """为每张图像同时构建 low-level 与 high-level 特征，供后续选择使用。"""

    high_features, valid_paths, high_backend = backbone.encode_images(image_paths)
    feature_bank: Dict[str, Dict[str, Any]] = {}
    for idx, img_path in enumerate(valid_paths):
        feature_bank[img_path] = {
            "low_level": extract_low_level_features(img_path),
            "high_level": high_features[idx] if high_features is not None else None,
            "high_level_backbone": high_backend,
        }
        low = feature_bank[img_path]["low_level"]
        print(
            f"图像[{img_path}] 特征完成 | Low-level: dominant_color={low['dominant_color']}, "
            f"texture={low['texture_stats']} | High-level backbone={high_backend}"
        )
    return feature_bank


def cosine_similarity_matrix(features: torch.Tensor) -> np.ndarray:
    """计算特征矩阵的余弦相似度矩阵。"""

    sim_matrix = torch.matmul(features, features.T).detach().cpu().numpy()
    return np.clip(sim_matrix, -1.0, 1.0)


def safe_linkage_from_similarity(sim_matrix: np.ndarray) -> np.ndarray:
    """将相似度矩阵转换为层次聚类所需的 linkage 矩阵。"""

    dist_matrix = np.clip(1 - sim_matrix, 0.0, 2.0)
    np.fill_diagonal(dist_matrix, 0.0)
    return linkage(squareform(dist_matrix, checks=False), method="average")


# -------------------------- 5. 自动关联挖掘函数 --------------------------
def auto_domain_clustering(texts: List[str], backbone: PerceptionBackbone) -> Tuple[Dict[str, int], Dict[int, List[str]]]:
    """自动文本领域聚类，用于辅助语言学关系判断。"""

    text_feats, text_labels, used_backend = backbone.encode_text(texts)
    if text_feats is None or len(text_labels) < 2:
        return {t: 0 for t in texts}, {0: texts}

    sim_matrix = cosine_similarity_matrix(text_feats)
    linkage_matrix = safe_linkage_from_similarity(sim_matrix)
    domain_labels = fcluster(linkage_matrix, t=CLUSTER_THRESHOLD_DOMAIN, criterion="distance")

    text2domain: Dict[str, int] = {}
    domain2texts: Dict[int, List[str]] = {}
    for text, domain_id in zip(text_labels, domain_labels):
        domain_id = int(domain_id)
        text2domain[text] = domain_id
        domain2texts.setdefault(domain_id, []).append(text)

    print(f"\n===== 自动划分的文本领域（backbone={used_backend}）=====")
    for domain_id, texts_in_domain in domain2texts.items():
        print(f"领域ID {domain_id}: {texts_in_domain}")
    return text2domain, domain2texts


def auto_linguistic_relation(texts: List[str], backbone: PerceptionBackbone) -> Dict[str, List[Tuple[str, str, float]]]:
    """自动挖掘语言学关联：同义、反义、比喻。"""

    print("\n===== 自动挖掘语言学关联（同义/反义/比喻）=====")
    text2domain, _ = auto_domain_clustering(texts, backbone)
    text_feats, text_labels, used_backend = backbone.encode_text(texts)
    if text_feats is None:
        return {}

    sim_matrix = cosine_similarity_matrix(text_feats)
    linguistic_rels: Dict[str, List[Tuple[str, str, float]]] = {"同义": [], "反义": [], "比喻": []}

    for i in range(len(text_labels)):
        for j in range(i + 1, len(text_labels)):
            text1, text2 = text_labels[i], text_labels[j]
            sim = round(float(sim_matrix[i][j]), 4)
            same_domain = text2domain[text1] == text2domain[text2]
            if sim >= SIM_THRESHOLD_SYNONYM and same_domain:
                linguistic_rels["同义"].append((text1, text2, sim))
                print(f"[{text1}] -> [{text2}] | 同义 | 相似度={sim} | backbone={used_backend}")
            elif sim <= SIM_THRESHOLD_ANTONYM and same_domain:
                linguistic_rels["反义"].append((text1, text2, sim))
                print(f"[{text1}] -> [{text2}] | 反义 | 相似度={sim} | backbone={used_backend}")
            elif SIM_THRESHOLD_METAPHOR <= sim < SIM_THRESHOLD_SYNONYM and not same_domain:
                linguistic_rels["比喻"].append((text1, text2, sim))
                print(f"[{text1}] -> [{text2}] | 比喻 | 相似度={sim} | backbone={used_backend}")

    return linguistic_rels


def auto_hierarchy_relation(texts: List[str], backbone: PerceptionBackbone) -> Dict[str, List[Tuple[str, str, float]]]:
    """自动挖掘层次关联：同层次与跨层次。"""

    print("\n===== 自动挖掘层次关联（同层次/跨层次）=====")
    text_feats, text_labels, used_backend = backbone.encode_text(texts)
    if text_feats is None or len(text_labels) < 2:
        return {}

    sim_matrix = cosine_similarity_matrix(text_feats)
    linkage_matrix = safe_linkage_from_similarity(sim_matrix)
    cluster_labels = fcluster(linkage_matrix, t=CLUSTER_THRESHOLD_HIERARCHY, criterion="distance")

    cluster2texts: Dict[int, List[str]] = {}
    for text, cluster_id in zip(text_labels, cluster_labels):
        cluster2texts.setdefault(int(cluster_id), []).append(text)

    same_level_rels: List[Tuple[str, str, float]] = []
    for cluster_id, texts_in_cluster in cluster2texts.items():
        for i in range(len(texts_in_cluster)):
            for j in range(i + 1, len(texts_in_cluster)):
                text1, text2 = texts_in_cluster[i], texts_in_cluster[j]
                sim = round(float(sim_matrix[text_labels.index(text1)][text_labels.index(text2)]), 4)
                same_level_rels.append((text1, text2, sim))
                print(f"同层次 | [{text1}] -> [{text2}] | 相似度={sim} | 聚类ID={cluster_id}")

    cross_level_rels: List[Tuple[str, str, float]] = []
    sorted_clusters = sorted(cluster2texts.items(), key=lambda item: len(item[1]), reverse=True)
    for i in range(len(sorted_clusters)):
        big_cluster_id, big_texts = sorted_clusters[i]
        for j in range(i + 1, len(sorted_clusters)):
            small_cluster_id, small_texts = sorted_clusters[j]
            big_feats = text_feats[[text_labels.index(text) for text in big_texts]]
            small_feats = text_feats[[text_labels.index(text) for text in small_texts]]
            avg_sim = torch.matmul(
                normalize_features(big_feats.mean(dim=0, keepdim=True)),
                normalize_features(small_feats.mean(dim=0, keepdim=True)).T,
            )
            avg_sim_value = round(float(avg_sim.detach().cpu().numpy()[0][0]), 4)
            if avg_sim_value > SIM_THRESHOLD_METAPHOR:
                cross_level_rels.append((big_texts[0], small_texts[0], avg_sim_value))
                print(
                    f"跨层次 | [{big_texts[0]}] -> [{small_texts[0]}] | 平均相似度={avg_sim_value} "
                    f"| 大类{big_cluster_id}->小类{small_cluster_id} | backbone={used_backend}"
                )

    return {"同层次": same_level_rels, "跨层次": cross_level_rels}


def auto_cross_modal_relation(
    image_paths: List[str], texts: List[str], backbone: PerceptionBackbone
) -> List[Tuple[str, str, float]]:
    """自动挖掘跨模态关联：图像-文本。"""

    print("\n===== 自动挖掘跨模态关联（图像-文本）=====")
    img_feats, img_labels, img_backend = backbone.encode_cross_modal_images(image_paths)
    text_feats, text_labels, text_backend = backbone.encode_text(texts)
    if img_feats is None or text_feats is None:
        return []

    sim_matrix = torch.matmul(img_feats, text_feats.T).detach().cpu().numpy()
    cross_modal_rels: List[Tuple[str, str, float]] = []
    for i, img_path in enumerate(img_labels):
        for j, text in enumerate(text_labels):
            sim = round(float(sim_matrix[i][j]), 4)
            cross_modal_rels.append((img_path, text, sim))
            print(
                f"图像[{img_path}] -> 文本[{text}] | 相似度={sim} "
                f"| image_backbone={img_backend} | text_backbone={text_backend}"
            )
    return cross_modal_rels


# -------------------------- 6. 图像内知识图谱构建 --------------------------
def infer_object_nodes_from_texts(
    image_path: str, cross_modal_rels: List[Tuple[str, str, float]], top_k: int = 5
) -> List[Tuple[str, float]]:
    """用跨模态得分从候选文本中选出当前图像的物体/概念节点。"""

    candidates = [(text, score) for img, text, score in cross_modal_rels if img == image_path]
    candidates = sorted(candidates, key=lambda item: item[1], reverse=True)
    return [(text, score) for text, score in candidates[:top_k] if score >= SIM_THRESHOLD_CROSS_MODAL]


def build_image_knowledge_graph(
    image_path: str,
    object_candidates: List[Tuple[str, float]],
    feature_bank: Dict[str, Dict[str, Any]],
    text_features: Optional[torch.Tensor],
    text_labels: List[str],
) -> KnowledgeGraph:
    """基于候选物体、低层特征和高层特征构建图像内知识图谱。"""

    graph = KnowledgeGraph()
    low_level = feature_bank[image_path]["low_level"]
    high_backend = feature_bank[image_path]["high_level_backbone"]

    image_node_id = f"{Path(image_path).stem}:image"
    graph.add_node(
        GraphNode(
            node_id=image_node_id,
            label="整图",
            image_path=image_path,
            node_type="image",
            attributes={
                "dominant_color": low_level["dominant_color"],
                "texture": low_level["texture_stats"],
                "high_level_backbone": high_backend,
            },
        )
    )

    object_node_ids: List[str] = []
    for idx, (label, score) in enumerate(object_candidates):
        node_id = f"{Path(image_path).stem}:object:{idx}:{label}"
        object_node_ids.append(node_id)
        graph.add_node(
            GraphNode(
                node_id=node_id,
                label=label,
                image_path=image_path,
                attributes={
                    "cross_modal_score": score,
                    "dominant_color": low_level["dominant_color"],
                    "texture": low_level["texture_stats"],
                },
            )
        )
        graph.add_edge(
            GraphEdge(
                source=image_node_id,
                target=node_id,
                relation="包含/感知到",
                weight=score,
                evidence="跨模态相似度超过阈值",
            )
        )

    if text_features is not None:
        for i in range(len(object_node_ids)):
            for j in range(i + 1, len(object_node_ids)):
                label_i = graph.nodes[object_node_ids[i]].label
                label_j = graph.nodes[object_node_ids[j]].label
                if label_i not in text_labels or label_j not in text_labels:
                    continue
                feat_i = text_features[text_labels.index(label_i)]
                feat_j = text_features[text_labels.index(label_j)]
                sim = torch.dot(feat_i, feat_j).detach().cpu().item()
                if sim >= SIM_THRESHOLD_OBJECT_RELATION:
                    graph.add_edge(
                        GraphEdge(
                            source=object_node_ids[i],
                            target=object_node_ids[j],
                            relation="语义相关/同场景共现",
                            weight=round(float(sim), 4),
                            evidence="候选物体文本特征相似",
                        )
                    )

    return graph


def build_memory_from_perception(
    image_paths: List[str],
    texts: List[str],
    backbone: PerceptionBackbone,
    feature_bank: Dict[str, Dict[str, Any]],
    cross_modal_rels: List[Tuple[str, str, float]],
    memory: MemorySystem,
) -> None:
    """把当前感知结果写入 short-term、long-term 与 experience memory。"""

    text_features, text_labels, text_backend = backbone.encode_text(texts)
    memory.short_term_memory["当前样本图像"] = image_paths
    memory.short_term_memory["当前候选文本"] = texts
    memory.short_term_memory["文本特征backbone"] = text_backend
    memory.short_term_memory["跨模态关系数量"] = len(cross_modal_rels)

    for image_path in feature_bank:
        object_candidates = infer_object_nodes_from_texts(image_path, cross_modal_rels)
        graph = build_image_knowledge_graph(image_path, object_candidates, feature_bank, text_features, text_labels)
        graph.print_graph()

        feature_summary = {
            "image_path": image_path,
            "dominant_color": feature_bank[image_path]["low_level"]["dominant_color"],
            "texture": feature_bank[image_path]["low_level"]["texture_stats"],
            "objects": object_candidates,
        }
        memory.short_term_memory[f"{image_path}:候选物体"] = object_candidates
        memory.update_long_term_graph(image_path, graph)
        memory.update_experience(graph, feature_summary)


# -------------------------- 7. 主推理流程 --------------------------
def main() -> None:
    # 示例输入：可替换为任意图像与候选文本，无需预定义领域。
    image_paths = [
        "house.jpg",    # 房屋实景图
        "kitchen.jpg",  # 厨房图
        "sun.jpg",      # 太阳图
    ]
    text_queries = [
        "太阳", "月亮", "火球", "日头",     # 语言学关联候选
        "房屋", "厨房", "卧室", "刀具", "碗筷",  # 层次与物体候选
    ]

    backbone = PerceptionBackbone(BACKBONE_NAME)
    memory = MemorySystem()
    memory.reset_short_term()

    # 1. 先对图像提取低层特征，并将整图 backbone 输出作为高层特征。
    print("\n===== 图像 Low-level / High-level 特征提取 =====")
    feature_bank = extract_image_feature_bank(backbone, image_paths)

    # 2. 自动挖掘跨模态关联。
    cross_modal_rels = auto_cross_modal_relation(list(feature_bank.keys()), text_queries, backbone)

    # 3. 自动挖掘语言学关联。
    linguistic_rels = auto_linguistic_relation(text_queries, backbone)

    # 4. 自动挖掘同层次与跨层次关联。
    hierarchy_rels = auto_hierarchy_relation(text_queries, backbone)

    # 5. 基于不同特征构建知识图谱与三类 memory。
    build_memory_from_perception(list(feature_bank.keys()), text_queries, backbone, feature_bank, cross_modal_rels, memory)
    memory.short_term_memory["语言学关联"] = linguistic_rels
    memory.short_term_memory["层次关联"] = hierarchy_rels
    memory.print_memory()


if __name__ == "__main__":
    main()
