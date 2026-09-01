"""Fail-closed I/O and response validation for Alibaba Cloud Qwen.

These helpers support the completed realtime ``qwen-flash`` cache pipeline.
They contain no graph-model or training code and open no network connection.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
import uuid
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_REALTIME_MODEL = "qwen-flash"
PROVIDER_NAME = "aliyun_qwen_realtime"
OFFICIAL_HOST = "dashscope.aliyuncs.com"
MAX_REQUESTS_PER_FILE = 50_000
MAX_FILE_BYTES = 500 * 1024 * 1024
# Two official pages document different per-line ceilings.  Use the stricter
# one so a locally accepted job is accepted by both interpretations.
MAX_LINE_BYTES = 1 * 1024 * 1024
MAX_CUSTOM_ID_CHARS = 256


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
            f"{DEFAULT_BASE_URL}; realtime Qwen is not routed through an arbitrary host"
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


def validate_qwen_requests(path: Path | str) -> Dict[str, object]:
    request_path = Path(path)
    size = request_path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(f"Qwen request plan exceeds {MAX_FILE_BYTES} bytes: {request_path}")
    rows = read_jsonl(request_path)
    if not rows:
        raise ValueError(f"Qwen request plan is empty: {request_path}")
    if len(rows) > MAX_REQUESTS_PER_FILE:
        raise ValueError(f"Qwen request plan exceeds {MAX_REQUESTS_PER_FILE} requests")
    custom_ids: set[str] = set()
    models: set[str] = set()
    for index, row in enumerate(rows, start=1):
        line_size = len((canonical_json(row) + "\n").encode("utf-8"))
        if line_size > MAX_LINE_BYTES:
            raise ValueError(f"Qwen request row {index} exceeds the conservative 1 MiB limit")
        if set(row) != {"custom_id", "method", "url", "body"}:
            raise ValueError(f"Qwen request row {index} has missing or unexpected top-level fields")
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id or len(custom_id) > MAX_CUSTOM_ID_CHARS:
            raise ValueError(f"Qwen request row {index} has an invalid custom_id")
        if custom_id in custom_ids:
            raise ValueError(f"duplicate custom_id in Qwen request plan: {custom_id}")
        custom_ids.add(custom_id)
        if row.get("method") != "POST" or row.get("url") != "/v1/chat/completions":
            raise ValueError(f"Qwen request row {index} has an unsupported method or endpoint")
        body = row.get("body")
        if not isinstance(body, dict):
            raise ValueError(f"Qwen request row {index} body must be an object")
        required = {"model", "messages", "response_format", "enable_thinking", "temperature"}
        if not required.issubset(body):
            raise ValueError(f"Qwen request row {index} body is missing protocol fields")
        if "max_tokens" in body or "stream" in body:
            raise ValueError("Qwen structured-output requests must omit max_tokens and stream")
        if body.get("enable_thinking") is not False or float(body.get("temperature", -1)) != 0.0:
            raise ValueError(f"Qwen request row {index} must disable thinking and use temperature 0")
        if body.get("response_format") != {"type": "json_object"}:
            raise ValueError(f"Qwen request row {index} must request json_object output")
        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"Qwen request row {index} has an invalid model")
        models.add(model)
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Qwen request row {index} messages must be a non-empty list")
        message_text = " ".join(
            str(item.get("content", "")) for item in messages if isinstance(item, dict)
        )
        if "json" not in message_text.casefold():
            raise ValueError(f"Qwen request row {index} prompt must explicitly mention JSON")
        secret_fields: set[str] = set()
        _find_secret_like_fields(row, secret_fields)
        if secret_fields:
            raise ValueError(f"Qwen request contains credential-like fields: {sorted(secret_fields)}")
    if len(models) != 1:
        raise ValueError("one Qwen request plan must use exactly one model")
    return {
        "request_count": len(rows),
        "file_bytes": size,
        "sha256": sha256_file(request_path),
        "model": next(iter(models)),
        "custom_ids": custom_ids,
    }


def parse_candidate_content(content: str, *, max_candidates: int = 10) -> Mapping[str, object]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Qwen response content is empty")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Qwen response content is not valid JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {"candidates"}:
        raise ValueError("Qwen response must contain exactly the top-level candidates field")
    candidates = value["candidates"]
    if not isinstance(candidates, list) or len(candidates) > max_candidates:
        raise ValueError(f"Qwen response candidates must be a list of at most {max_candidates}")
    required = {"entity_name", "confidence", "temporal_rationale", "temporal_consistency"}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or set(candidate) != required:
            raise ValueError(f"Qwen response candidate {index} has missing or unexpected fields")
        if not isinstance(candidate["entity_name"], str) or not candidate["entity_name"].strip():
            raise ValueError(f"Qwen response candidate {index} has an invalid entity_name")
        if not isinstance(candidate["temporal_rationale"], str):
            raise ValueError(f"Qwen response candidate {index} has an invalid rationale")
        for field in ("confidence", "temporal_consistency"):
            number = candidate[field]
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ValueError(f"Qwen response candidate {index} has non-numeric {field}")
            if not 0.0 <= float(number) <= 1.0:
                raise ValueError(f"Qwen response candidate {index} has {field} outside [0,1]")
    return value
