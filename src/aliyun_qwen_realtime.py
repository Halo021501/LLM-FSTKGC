"""Rate-limited Alibaba Qwen realtime helpers for offline cache generation.

Only the explicit realtime runner constructs this client.  Training and
evaluation consume completed local caches and never call the provider.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from email.message import Message
from typing import Callable, Deque, Dict, Mapping, Tuple

from src.aliyun_qwen_batch import (
    DEFAULT_BASE_URL,
    canonical_json,
    official_base_url,
    parse_candidate_content,
    validate_api_key,
)


PROVIDER_NAME = "aliyun_qwen_realtime"
DEFAULT_REALTIME_MODEL = "qwen-flash"
OFFICIAL_FLOATING_MODEL_RPM = 30_000
OFFICIAL_FLOATING_MODEL_TPM = 10_000_000


class RealtimeAPIError(RuntimeError):
    """Sanitized provider/transport failure safe for local logs."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str = "unknown",
        retry_after_seconds: float | None = None,
        retriable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = str(code)
        self.retry_after_seconds = retry_after_seconds
        self.retriable = bool(retriable)


def _retry_after(headers: Message | Mapping[str, str] | None) -> float | None:
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return max(0.0, value)


def _provider_error_code(raw: bytes) -> str:
    try:
        value = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return "unparseable_provider_error"
    if not isinstance(value, dict):
        return "unparseable_provider_error"
    error = value.get("error")
    if isinstance(error, dict):
        code = error.get("code") or error.get("type")
    else:
        code = value.get("code")
    return str(code or "provider_error")[:128]


def validate_realtime_body(body: Mapping[str, object], *, expected_model: str) -> Dict[str, object]:
    value = dict(body)
    required = {"model", "messages", "response_format", "enable_thinking", "temperature"}
    if not required.issubset(value):
        raise ValueError("realtime body is missing the target-blind generation protocol")
    if value.get("model") != expected_model:
        raise ValueError("realtime request model differs from the reviewed plan")
    if value.get("enable_thinking") is not False:
        raise ValueError("realtime generation must set enable_thinking=false")
    if float(value.get("temperature", -1)) != 0.0:
        raise ValueError("realtime generation must use temperature=0")
    if value.get("response_format") != {"type": "json_object"}:
        raise ValueError("realtime generation must request json_object output")
    if value.get("stream") not in (None, False):
        raise ValueError("realtime cache generation does not support streaming")
    messages = value.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("realtime request messages must be non-empty")
    prompt_text = " ".join(
        str(message.get("content", "")) for message in messages if isinstance(message, dict)
    )
    if "json" not in prompt_text.casefold():
        raise ValueError("structured-output prompt must explicitly mention JSON")
    return value


class SlidingWindowRateLimiter:
    """Thread-safe RPM/TPM limiter with a shared provider cooldown."""

    def __init__(
        self,
        *,
        max_rpm: int,
        max_tpm: int,
        token_reservation: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_rpm < 1 or max_tpm < 1 or token_reservation < 1:
            raise ValueError("rate-limit values must be positive")
        if token_reservation > max_tpm:
            raise ValueError("one token reservation exceeds the configured TPM budget")
        self.max_rpm = int(max_rpm)
        self.max_tpm = int(max_tpm)
        self.token_reservation = int(token_reservation)
        self.window_seconds = float(window_seconds)
        self._clock = clock
        self._requests: Deque[float] = deque()
        self._tokens: Deque[Tuple[float, int]] = deque()
        self._token_total = 0
        self._cooldown_until = 0.0
        self._condition = threading.Condition()

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._requests and self._requests[0] <= cutoff:
            self._requests.popleft()
        while self._tokens and self._tokens[0][0] <= cutoff:
            _, tokens = self._tokens.popleft()
            self._token_total -= tokens

    def defer(self, seconds: float) -> None:
        with self._condition:
            self._cooldown_until = max(
                self._cooldown_until, self._clock() + max(0.0, float(seconds))
            )
            self._condition.notify_all()

    def acquire(self) -> None:
        with self._condition:
            while True:
                now = self._clock()
                self._evict(now)
                waits = [max(0.0, self._cooldown_until - now)]
                if len(self._requests) >= self.max_rpm:
                    waits.append(max(0.001, self._requests[0] + self.window_seconds - now))
                if self._token_total + self.token_reservation > self.max_tpm and self._tokens:
                    waits.append(max(0.001, self._tokens[0][0] + self.window_seconds - now))
                wait_seconds = max(waits)
                if wait_seconds <= 0.0:
                    self._requests.append(now)
                    self._tokens.append((now, self.token_reservation))
                    self._token_total += self.token_reservation
                    return
                self._condition.wait(timeout=wait_seconds)


class AliyunQwenRealtimeClient:
    """Minimal OpenAI-compatible client restricted to Alibaba's Beijing host."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        *,
        model: str = DEFAULT_REALTIME_MODEL,
        timeout_seconds: float = 120.0,
        opener=None,
    ) -> None:
        self.api_key = validate_api_key(api_key)
        self.base_url = official_base_url(base_url)
        self.model = str(model)
        if self.model != DEFAULT_REALTIME_MODEL:
            raise ValueError("the reviewed realtime path currently permits only qwen-flash")
        self.timeout_seconds = float(timeout_seconds)
        self._opener = opener or urllib.request.build_opener()

    @classmethod
    def from_environment(
        cls, *, model: str = DEFAULT_REALTIME_MODEL, timeout_seconds: float = 120.0
    ) -> "AliyunQwenRealtimeClient":
        import os

        return cls(
            os.environ.get("DASHSCOPE_API_KEY", ""),
            os.environ.get(
                "ALIYUN_QWEN_REALTIME_BASE_URL",
                os.environ.get("ALIYUN_QWEN_BATCH_BASE_URL", DEFAULT_BASE_URL),
            ),
            model=model,
            timeout_seconds=timeout_seconds,
        )

    def complete(self, body: Mapping[str, object]) -> Dict[str, object]:
        payload = validate_realtime_body(body, expected_model=self.model)
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=canonical_json(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:4096]
            code = _provider_error_code(detail)
            retriable = exc.code in {408, 409, 425, 429} or 500 <= exc.code <= 599
            raise RealtimeAPIError(
                f"Alibaba realtime HTTP {exc.code} ({code})",
                status_code=exc.code,
                code=code,
                retry_after_seconds=_retry_after(exc.headers),
                retriable=retriable,
            ) from exc
        except urllib.error.URLError as exc:
            raise RealtimeAPIError(
                "Alibaba realtime transport failure",
                code="transport_error",
                retriable=True,
            ) from exc
        except TimeoutError as exc:
            # urllib wraps many connection failures as URLError, but a timeout
            # raised while consuming the response body can escape directly.
            # Normalize it here so one transient read timeout cannot tear down
            # the result-consumer loop in the concurrent runner.
            raise RealtimeAPIError(
                "Alibaba realtime transport timeout",
                code="transport_timeout",
                retriable=True,
            ) from exc
        except (OSError, http.client.HTTPException) as exc:
            # response.read() may also expose connection resets, incomplete
            # reads, TLS failures, or malformed HTTP responses directly.
            raise RealtimeAPIError(
                "Alibaba realtime transport failure",
                code="transport_error",
                retriable=True,
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RealtimeAPIError(
                "Alibaba realtime returned invalid JSON",
                code="invalid_envelope_json",
                retriable=True,
            ) from exc
        if not isinstance(result, dict):
            raise RealtimeAPIError(
                "Alibaba realtime response is not an object",
                code="invalid_envelope_type",
                retriable=True,
            )
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RealtimeAPIError(
                "Alibaba realtime response has no first choice",
                code="missing_choice",
                retriable=True,
            )
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            raise RealtimeAPIError(
                "Alibaba realtime response was truncated",
                code="truncated_output",
                retriable=True,
            )
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        try:
            parse_candidate_content(content)
        except ValueError as exc:
            raise RealtimeAPIError(
                "Alibaba realtime candidate JSON failed the strict schema",
                code="invalid_candidate_json",
                retriable=True,
            ) from exc
        safe_body = {
            "id": result.get("id"),
            "model": result.get("model", self.model),
            "choices": [choice],
            "usage": result.get("usage", {}),
        }
        return {"body": safe_body, "latency_ms": latency_ms}
