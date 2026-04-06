"""
Long-term memory modules from the architecture figure:
- Conceptual model of the world (entities, dynamics priors)
- Temporal knowledge (events, routines)
- Spatial knowledge (persistent maps / topology)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import mimetypes
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
import uuid
from urllib import error, parse, request


NodeId = str


@dataclass
class ConceptualWorldModel:
    """
    Persistent conceptual structure: object entities, affordance tags, and simple rules.
    Represented as an entity table plus optional relation edges for prototyping.
    """

    entities: Dict[NodeId, Dict[str, Any]] = field(default_factory=dict)
    relations: List[Tuple[NodeId, str, NodeId]] = field(default_factory=list)

    def upsert_entity(self, eid: NodeId, record: Dict[str, Any]) -> None:
        base = self.entities.get(eid, {})
        base.update(record)
        self.entities[eid] = base

    def add_relation(self, src: NodeId, rel_type: str, dst: NodeId) -> None:
        self.relations.append((src, rel_type, dst))


@dataclass
class TemporalKnowledgeBase:
    """
    Sequences and histories: keyed episodes, ordered event strings, optional timestamps.
    """

    episodes: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=256))

    def record_episode_summary(self, summary: Dict[str, Any]) -> None:
        self.episodes.append(summary)


@dataclass
class SpatialKnowledgeGraph:
    """
    Large-scale spatial memory: metric landmarks or topological nodes.
    For sim2real, you can back this with a pose graph SLAM module behind the same API.
    """

    nodes: Dict[NodeId, Dict[str, Any]] = field(default_factory=dict)
    edges: List[Tuple[NodeId, NodeId, Dict[str, Any]]] = field(default_factory=list)

    def add_node(self, nid: NodeId, attrs: Dict[str, Any]) -> None:
        self.nodes[nid] = attrs

    def add_edge(self, a: NodeId, b: NodeId, attrs: Optional[Dict[str, Any]] = None) -> None:
        self.edges.append((a, b, attrs or {}))


@dataclass
class EmbodiedLTMClient:
    """Tiny HTTP adapter for EmbodiedLTM FastAPI service."""

    base_url: str = "http://127.0.0.1:8000"
    timeout_sec: float = 8.0

    def _safe_request(self, req: request.Request) -> Dict[str, Any]:
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except TimeoutError:
            return {"status": "error", "message": "EmbodiedLTM request timeout"}
        except error.URLError as e:
            return {"status": "error", "message": f"EmbodiedLTM unreachable: {e}"}
        except json.JSONDecodeError:
            return {"status": "error", "message": "EmbodiedLTM returned non-JSON payload"}

    def _post(
        self,
        endpoint: str,
        *,
        form: Dict[str, Any],
        files: Optional[Dict[str, Optional[str]]] = None,
    ) -> Dict[str, Any]:
        files = files or {}
        has_file = any(files.values())

        if not has_file:
            payload = parse.urlencode(form).encode("utf-8")
            req = request.Request(
                url=f"{self.base_url.rstrip('/')}{endpoint}",
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            return self._safe_request(req)

        boundary = f"----EmbodiedLTMBoundary{uuid.uuid4().hex}"
        body = bytearray()

        for key, value in form.items():
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8")
            )
            body.extend(str(value).encode("utf-8"))
            body.extend(b"\r\n")

        for field_name, file_path in files.items():
            if not file_path:
                continue
            path_obj = Path(file_path)
            if not path_obj.exists() or not path_obj.is_file():
                return {
                    "status": "error",
                    "message": f"file not found for field '{field_name}': {file_path}",
                }

            content_type = mimetypes.guess_type(path_obj.name)[0] or "application/octet-stream"
            file_bytes = path_obj.read_bytes()

            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(
                (
                    f'Content-Disposition: form-data; name="{field_name}"; '
                    f'filename="{path_obj.name}"\r\n'
                ).encode("utf-8")
            )
            body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
            body.extend(file_bytes)
            body.extend(b"\r\n")

        body.extend(f"--{boundary}--\r\n".encode("utf-8"))

        req = request.Request(
            url=f"{self.base_url.rstrip('/')}{endpoint}",
            data=bytes(body),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        return self._safe_request(req)

    def insert_memory(
        self,
        *,
        query_text: str,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        files = {
            "image": image_path,
            "video": video_path,
            "audio": audio_path,
        }
        return self._post(
            "/insert",
            form={"query": query_text},
            files=files,
        )

    def query_memory(
        self,
        query: str,
    ) -> Dict[str, Any]:
        return self._post(
            "/query",
            form={
                "query": query,
            },
        )


@dataclass
class ManipulationLongTermMemory:
    """Bundles the three long-term compartments."""

    conceptual: ConceptualWorldModel = field(default_factory=ConceptualWorldModel)
    temporal: TemporalKnowledgeBase = field(default_factory=TemporalKnowledgeBase)
    spatial: SpatialKnowledgeGraph = field(default_factory=SpatialKnowledgeGraph)
    remote_client: Optional[EmbodiedLTMClient] = None

    def configure_embodiedltm(self, *, base_url: str, timeout_sec: float = 8.0) -> None:
        self.remote_client = EmbodiedLTMClient(base_url=base_url, timeout_sec=timeout_sec)

    def insert_longterm_memory(
        self,
        summary: Dict[str, Any],
        *,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Push a lightweight textual episode summary to EmbodiedLTM /insert.
        Returns a status dict and never raises.
        """
        if self.remote_client is None:
            return {"status": "skipped", "message": "remote client not configured"}
        text = json.dumps(summary, ensure_ascii=False)
        return self.remote_client.insert_memory(
            query_text=text,
            image_path=image_path,
            video_path=video_path,
            audio_path=audio_path,
        )

    def remote_query(
        self,
        query: str,
    ) -> Dict[str, Any]:
        if self.remote_client is None:
            return {"status": "skipped", "message": "remote client not configured"}
        return self.remote_client.query_memory(query)
