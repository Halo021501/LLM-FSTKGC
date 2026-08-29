"""Fail-closed Alibaba Model Studio Batch API helpers.

This module deliberately contains no experiment or model code.  Offline stages
can import its validators without reading credentials or opening a socket.  A
network client is constructed only by an explicit CLI action after the caller
has passed the project confirmation gates.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_BATCH_MODEL = "qwen3.7-flash-2026-07-15"
ROLLING_BATCH_MODEL = "qwen-flash"
PROVIDER_NAME = "aliyun_qwen_batch"
OFFICIAL_HOST = "dashscope.aliyuncs.com"
MAX_REQUESTS_PER_FILE = 50_000
MAX_FILE_BYTES = 500 * 1024 * 1024
# Two official pages document different per-line ceilings.  Use the stricter
# one so a locally accepted job is accepted by both interpretations.
MAX_LINE_BYTES = 1 * 1024 * 1024
MAX_CUSTOM_ID_CHARS = 256
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_base_url(value: str) -> str:
    parsed = urllib.parse.urlparse(str(value).rstrip("/"))
    if (
        parsed.scheme != "https"
        or parsed.hostname != OFFICIAL_HOST
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/compatible-mode/v1"
    ):
        raise ValueError(
            "ALIYUN_QWEN_BASE_URL must be the official Beijing HTTPS endpoint "
            f"{DEFAULT_BASE_URL}; Flash Batch is not routed through an arbitrary host"
        )
    return DEFAULT_BASE_URL


def validate_api_key(value: str) -> str:
    key = str(value).strip()
    if not key or key.startswith(("YOUR_", "sk-REPLACE", "REPLACE_")):
        raise ValueError("DASHSCOPE_API_KEY is missing; fill the gitignored local environment file")
    if any(char.isspace() for char in key):
        raise ValueError("DASHSCOPE_API_KEY must not contain whitespace")
    return key


def _atomic_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")


def write_bytes_atomic(path: Path | str, data: bytes, *, replace: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(target.parent, 0o700)
    except PermissionError:
        pass
    if target.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    temporary = _atomic_path(target)
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists() and not replace:
            raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
        os.replace(temporary, target)
        os.chmod(target, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path | str, value: object, *, replace: bool = False) -> None:
    write_bytes_atomic(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        replace=replace,
    )


def read_json(path: Path | str) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join((canonical_json(dict(row)) + "\n").encode("utf-8") for row in rows)


def read_jsonl(path: Path | str) -> list[Dict[str, object]]:
    output: list[Dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            output.append(value)
    return output


def _find_secret_like_fields(value: object, found: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ("api_key", "apikey", "authorization", "access_key")):
                found.add(str(key))
            _find_secret_like_fields(child, found)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _find_secret_like_fields(child, found)


def validate_batch_requests(path: Path | str) -> Dict[str, object]:
    request_path = Path(path)
    size = request_path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(f"Batch request file exceeds {MAX_FILE_BYTES} bytes: {request_path}")
    rows = read_jsonl(request_path)
    if not rows:
        raise ValueError(f"Batch request file is empty: {request_path}")
    if len(rows) > MAX_REQUESTS_PER_FILE:
        raise ValueError(f"Batch request file exceeds {MAX_REQUESTS_PER_FILE} requests")
    custom_ids: set[str] = set()
    models: set[str] = set()
    for index, row in enumerate(rows, start=1):
        line_size = len((canonical_json(row) + "\n").encode("utf-8"))
        if line_size > MAX_LINE_BYTES:
            raise ValueError(f"Batch request row {index} exceeds the conservative 1 MiB limit")
        if set(row) != {"custom_id", "method", "url", "body"}:
            raise ValueError(f"Batch request row {index} has missing or unexpected top-level fields")
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id or len(custom_id) > MAX_CUSTOM_ID_CHARS:
            raise ValueError(f"Batch request row {index} has an invalid custom_id")
        if custom_id in custom_ids:
            raise ValueError(f"duplicate custom_id in Batch request file: {custom_id}")
        custom_ids.add(custom_id)
        if row.get("method") != "POST" or row.get("url") != "/v1/chat/completions":
            raise ValueError(f"Batch request row {index} has an unsupported method or endpoint")
        body = row.get("body")
        if not isinstance(body, dict):
            raise ValueError(f"Batch request row {index} body must be an object")
        required = {"model", "messages", "response_format", "enable_thinking", "temperature"}
        if not required.issubset(body):
            raise ValueError(f"Batch request row {index} body is missing protocol fields")
        if "max_tokens" in body or "stream" in body:
            raise ValueError("Batch structured-output requests must omit max_tokens and stream")
        if body.get("enable_thinking") is not False or float(body.get("temperature", -1)) != 0.0:
            raise ValueError(f"Batch request row {index} must disable thinking and use temperature 0")
        if body.get("response_format") != {"type": "json_object"}:
            raise ValueError(f"Batch request row {index} must request json_object output")
        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"Batch request row {index} has an invalid model")
        models.add(model)
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Batch request row {index} messages must be a non-empty list")
        message_text = " ".join(
            str(item.get("content", "")) for item in messages if isinstance(item, dict)
        )
        if "json" not in message_text.casefold():
            raise ValueError(f"Batch request row {index} prompt must explicitly mention JSON")
        secret_fields: set[str] = set()
        _find_secret_like_fields(row, secret_fields)
        if secret_fields:
            raise ValueError(f"Batch request contains credential-like fields: {sorted(secret_fields)}")
    if len(models) != 1:
        raise ValueError("one Batch input file must use exactly one model")
    return {
        "request_count": len(rows),
        "file_bytes": size,
        "sha256": sha256_file(request_path),
        "model": next(iter(models)),
        "custom_ids": custom_ids,
    }


def parse_candidate_content(content: str, *, max_candidates: int = 10) -> Mapping[str, object]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Batch response content is empty")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Batch response content is not valid JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"candidates"}:
        raise ValueError("Batch response must contain exactly the top-level candidates field")
    candidates = value["candidates"]
    if not isinstance(candidates, list) or len(candidates) > max_candidates:
        raise ValueError(f"Batch response candidates must be a list of at most {max_candidates}")
    required = {"entity_name", "confidence", "temporal_rationale", "temporal_consistency"}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or set(candidate) != required:
            raise ValueError(f"Batch response candidate {index} has missing or unexpected fields")
        if not isinstance(candidate["entity_name"], str) or not candidate["entity_name"].strip():
            raise ValueError(f"Batch response candidate {index} has an invalid entity_name")
        if not isinstance(candidate["temporal_rationale"], str):
            raise ValueError(f"Batch response candidate {index} has an invalid rationale")
        for field in ("confidence", "temporal_consistency"):
            number = candidate[field]
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ValueError(f"Batch response candidate {index} has non-numeric {field}")
            if not 0.0 <= float(number) <= 1.0:
                raise ValueError(f"Batch response candidate {index} has {field} outside [0,1]")
    return value


class AliyunQwenBatchClient:
    """Small standard-library client restricted to the official Beijing host."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout_seconds: float = 120.0,
        opener=None,
    ) -> None:
        self.api_key = validate_api_key(api_key)
        self.base_url = official_base_url(base_url)
        self.timeout_seconds = float(timeout_seconds)
        self._opener = opener or urllib.request.build_opener()

    @classmethod
    def from_environment(cls, *, timeout_seconds: float = 120.0) -> "AliyunQwenBatchClient":
        return cls(
            os.environ.get("DASHSCOPE_API_KEY", ""),
            os.environ.get(
                "ALIYUN_QWEN_BATCH_BASE_URL",
                os.environ.get("ALIYUN_QWEN_BASE_URL", DEFAULT_BASE_URL),
            ),
            timeout_seconds=timeout_seconds,
        )

    def _open(self, request: urllib.request.Request) -> bytes:
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(f"Alibaba Model Studio HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Alibaba Model Studio request failed: {exc.reason}") from exc

    def _json_request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None = None,
    ) -> Dict[str, object]:
        data = canonical_json(payload).encode("utf-8") if payload is not None else None
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}/{path.lstrip('/')}", data=data, headers=headers, method=method
        )
        raw = self._open(request)
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Alibaba Model Studio returned a non-JSON control response") from exc
        if not isinstance(result, dict):
            raise RuntimeError("Alibaba Model Studio control response is not an object")
        return result

    def upload_batch_file(self, path: Path | str) -> Dict[str, object]:
        input_path = Path(path)
        validate_batch_requests(input_path)
        file_bytes = input_path.read_bytes()
        boundary = f"codex-qwen-batch-{uuid.uuid4().hex}"
        chunks = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n".encode(),
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                f"filename=\"{input_path.name}\"\r\nContent-Type: application/jsonl\r\n\r\n"
            ).encode(),
            file_bytes,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
        request = urllib.request.Request(
            f"{self.base_url}/files",
            data=b"".join(chunks),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        raw = self._open(request)
        result = json.loads(raw.decode("utf-8"))
        if not isinstance(result, dict) or not result.get("id"):
            raise RuntimeError("Alibaba Model Studio file upload returned no file id")
        return result

    def get_file(self, file_id: str) -> Dict[str, object]:
        return self._json_request("GET", f"files/{urllib.parse.quote(file_id, safe='')}")

    def wait_until_file_processed(
        self, file_id: str, *, timeout_seconds: float = 300.0, poll_seconds: float = 2.0
    ) -> Dict[str, object]:
        deadline = time.monotonic() + float(timeout_seconds)
        while True:
            result = self.get_file(file_id)
            status = str(result.get("status", ""))
            if status == "processed":
                return result
            if status in {"error", "failed", "cancelled"}:
                raise RuntimeError(f"Alibaba Model Studio rejected the uploaded Batch file: {status}")
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for Alibaba Model Studio to process the Batch file")
            time.sleep(max(0.1, float(poll_seconds)))

    def create_batch(
        self,
        input_file_id: str,
        *,
        completion_window: str = "24h",
        metadata: Mapping[str, object] | None = None,
    ) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "input_file_id": input_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": completion_window,
        }
        if metadata:
            payload["metadata"] = dict(metadata)
        result = self._json_request("POST", "batches", payload)
        if not result.get("id"):
            raise RuntimeError("Alibaba Model Studio Batch creation returned no batch id")
        return result

    def get_batch(self, batch_id: str) -> Dict[str, object]:
        return self._json_request("GET", f"batches/{urllib.parse.quote(batch_id, safe='')}")

    def cancel_batch(self, batch_id: str) -> Dict[str, object]:
        return self._json_request(
            "POST", f"batches/{urllib.parse.quote(batch_id, safe='')}/cancel"
        )

    def download_file(self, file_id: str) -> bytes:
        request = urllib.request.Request(
            f"{self.base_url}/files/{urllib.parse.quote(file_id, safe='')}/content",
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        return self._open(request)

    def delete_file(self, file_id: str) -> Dict[str, object]:
        return self._json_request("DELETE", f"files/{urllib.parse.quote(file_id, safe='')}")
