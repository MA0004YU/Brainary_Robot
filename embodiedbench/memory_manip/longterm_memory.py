"""
EmbodiedLTM HTTP client for optional remote memory service.

The three in-process long-term stores (ConceptualWorldModel, TemporalKnowledgeBase,
SpatialKnowledgeGraph) have been superseded by EpisodicMemory and SemanticMemory.
This module now contains only the HTTP adapter for the external EmbodiedLTM service.
"""
from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional
import uuid
from urllib import error, request


class EmbodiedLTMClient:
    """Lightweight HTTP adapter for the EmbodiedLTM FastAPI service."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout_sec: float = 8.0) -> None:
        self.base_url = base_url
        self.timeout_sec = timeout_sec

    def _safe_request(self, req: request.Request) -> Dict[str, Any]:
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
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
        has_file = any(v for v in files.values())

        if not has_file:
            payload = "&".join(
                f"{k}={json.dumps(v) if not isinstance(v, str) else v}" for k, v in form.items()
            ).encode("utf-8")
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
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode())
            body.extend(str(value).encode())
            body.extend(b"\r\n")

        for field_name, file_path in files.items():
            if not file_path:
                continue
            path_obj = Path(file_path)
            if not path_obj.exists() or not path_obj.is_file():
                return {"status": "error", "message": f"file not found: {file_path}"}
            content_type = mimetypes.guess_type(path_obj.name)[0] or "application/octet-stream"
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(
                f'Content-Disposition: form-data; name="{field_name}"; filename="{path_obj.name}"\r\n'.encode()
            )
            body.extend(f"Content-Type: {content_type}\r\n\r\n".encode())
            body.extend(path_obj.read_bytes())
            body.extend(b"\r\n")

        body.extend(f"--{boundary}--\r\n".encode())
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
        return self._post(
            "/insert",
            form={"query": query_text},
            files={"image": image_path, "video": video_path, "audio": audio_path},
        )

    def query_memory(self, query: str, mode: str = "hybrid") -> Dict[str, Any]:
        return self._post("/query", form={"query": query, "mode": mode, "use_pm": "false"})

    def update_state(self, observation_desc: str) -> Dict[str, Any]:
        """Wrap a state-change description as a structured LTM insert (mirrors Planning_module LTMClient)."""
        prompt = (
            f"[STATE UPDATE] {observation_desc}. "
            "Please update the exact locations and states of the objects in your memory "
            "to reflect this exact new state. Remove any old locations that contradict this new state."
        )
        return self.insert_memory(query_text=prompt)
