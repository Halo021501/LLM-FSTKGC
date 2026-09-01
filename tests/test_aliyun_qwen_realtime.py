import contextlib
import io
import json
import os
import socket
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from pathlib import Path
from unittest import mock

from scripts import stlp_aliyun_qwen_request_plan as plan_cli
from scripts import stlp_aliyun_qwen_realtime as realtime_cli
from src.aliyun_qwen_io import read_jsonl
from src.aliyun_qwen_realtime import (
    DEFAULT_REALTIME_MODEL,
    AliyunQwenRealtimeClient,
    RealtimeAPIError,
)
from src.llm_cache import LLMEvidenceCache


def _candidate_content(name="Past_Candidate"):
    return json.dumps(
        {
            "candidates": [
                {
                    "entity_name": name,
                    "confidence": 0.8,
                    "temporal_rationale": "Earlier causal support",
                    "temporal_consistency": 0.7,
                }
            ]
        }
    )


def _provider_body(ordinal=0):
    return {
        "id": f"chat-{ordinal}",
        "model": DEFAULT_REALTIME_MODEL,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": _candidate_content()},
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


class _FakeClient:
    def __init__(self):
        self.calls = 0

    def complete(self, _body):
        self.calls += 1
        return {"body": _provider_body(self.calls), "latency_ms": 12.5}


class _ImmediateLimiter:
    def __init__(self):
        self.acquires = 0
        self.deferred = []

    def acquire(self):
        self.acquires += 1

    def defer(self, seconds):
        self.deferred.append(seconds)


class _InspectionBlockedThenSuccessClient(_FakeClient):
    def complete(self, body):
        self.calls += 1
        if self.calls <= 3:
            raise RealtimeAPIError(
                "sanitized provider inspection rejection",
                status_code=400,
                code=realtime_cli.INSPECTION_ERROR_CODE,
                retriable=False,
            )
        return {"body": _provider_body(self.calls), "latency_ms": 8.0}


class AliyunQwenRealtimeOfflineTests(unittest.TestCase):
    def setUp(self):
        self._guards = [
            mock.patch(
                "urllib.request.urlopen",
                side_effect=AssertionError("realtime offline test attempted network access"),
            ),
            mock.patch.object(
                socket,
                "create_connection",
                side_effect=AssertionError("realtime offline test attempted socket access"),
            ),
        ]
        for guard in self._guards:
            guard.start()
            self.addCleanup(guard.stop)

    @staticmethod
    def _dataset(root):
        data_dir = root / "tiny_tkg"
        data_dir.mkdir()
        (data_dir / "stat.txt").write_text("4\t1\t0\n", encoding="utf-8")
        (data_dir / "entity2id.txt").write_text(
            "Public_Known\t0\n"
            "Hidden_Target_SENTINEL\t1\n"
            "Past_Candidate\t2\n"
            "Other_Candidate\t3\n",
            encoding="utf-8",
        )
        (data_dir / "relation2id.txt").write_text("Public_Relation\t0\n", encoding="utf-8")
        (data_dir / "train.txt").write_text("0\t0\t2\t1\n", encoding="utf-8")
        (data_dir / "valid.txt").write_text("3\t0\t2\t2\n", encoding="utf-8")
        (data_dir / "test.txt").write_text("0\t0\t1\t3\n", encoding="utf-8")
        return data_dir

    @staticmethod
    def _prepare(data_dir, source_dir, *, limit=2):
        args = plan_cli.build_parser().parse_args(
            [
                "--job-dir",
                str(source_dir),
                "--data-dir",
                str(data_dir),
                "--split",
                "test",
                "--shot",
                "5",
                "--model",
                DEFAULT_REALTIME_MODEL,
                "--limit",
                str(limit),
            ]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            plan_cli.prepare_job(args)

    @classmethod
    def _many_request_dataset(cls, root):
        data_dir = cls._dataset(root)
        (data_dir / "test.txt").write_text(
            "0\t0\t1\t3\n"
            "1\t0\t2\t4\n"
            "2\t0\t3\t5\n"
            "3\t0\t0\t6\n"
            "0\t0\t2\t7\n"
            "1\t0\t3\t8\n"
            "2\t0\t0\t9\n"
            "3\t0\t1\t10\n",
            encoding="utf-8",
        )
        return data_dir

    @staticmethod
    def _run_args(source_dir, run_dir):
        return realtime_cli.build_parser().parse_args(
            [
                "run",
                "--source-job-dir",
                str(source_dir),
                "--run-dir",
                str(run_dir),
                "--workers",
                "2",
                "--max-rpm",
                "20",
                "--max-tpm",
                "50000",
                "--resume",
                "--execute-api",
            ]
        )

    def test_client_uses_official_endpoint_and_never_serializes_key(self):
        secret = "sk-test-secret-never-write"
        captured = {}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(_provider_body()).encode()

        class Opener:
            def open(self, request, timeout):
                captured["url"] = request.full_url
                captured["authorization"] = request.get_header("Authorization")
                captured["body"] = request.data
                captured["timeout"] = timeout
                return Response()

        client = AliyunQwenRealtimeClient(secret, opener=Opener())
        body = {
            "model": DEFAULT_REALTIME_MODEL,
            "enable_thinking": False,
            "messages": [{"role": "user", "content": "Return JSON only."}],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }
        result = client.complete(body)
        self.assertTrue(captured["url"].endswith("/chat/completions"))
        self.assertEqual(captured["authorization"], f"Bearer {secret}")
        self.assertNotIn(secret.encode(), captured["body"])
        self.assertEqual(result["body"]["model"], DEFAULT_REALTIME_MODEL)

    def test_response_read_timeout_is_wrapped_and_retried(self):
        body = {
            "model": DEFAULT_REALTIME_MODEL,
            "enable_thinking": False,
            "messages": [{"role": "user", "content": "Return JSON only."}],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }

        class TimeoutResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                raise TimeoutError("raw response read timeout")

        class TimeoutOpener:
            def open(self, _request, timeout):
                self.timeout = timeout
                return TimeoutResponse()

        timeout_client = AliyunQwenRealtimeClient(
            "sk-timeout-never-persist", opener=TimeoutOpener()
        )
        with self.assertRaises(RealtimeAPIError) as raised:
            timeout_client.complete(body)
        self.assertTrue(raised.exception.retriable)
        self.assertIsInstance(raised.exception.__cause__, TimeoutError)
        self.assertNotIn("sk-timeout-never-persist", str(raised.exception))

        class RetryResponse:
            def __init__(self, opener):
                self.opener = opener

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                if self.opener.reads == 0:
                    self.opener.reads += 1
                    raise TimeoutError("first response read timed out")
                self.opener.reads += 1
                return json.dumps(_provider_body()).encode("utf-8")

        class RetryOpener:
            def __init__(self):
                self.opens = 0
                self.reads = 0

            def open(self, _request, timeout):
                self.opens += 1
                self.timeout = timeout
                return RetryResponse(self)

        opener = RetryOpener()
        client = AliyunQwenRealtimeClient("sk-test", opener=opener)
        result, failure = realtime_cli._call_one(
            {"custom_id": "d" * 64, "body": body},
            client=client,
            limiter=_ImmediateLimiter(),
            max_attempts=2,
            inspection_max_attempts=2,
            backoff_base=0.0,
            backoff_max=0.0,
            stop_event=threading.Event(),
            sleep_fn=lambda _seconds: None,
        )
        self.assertIsNone(failure)
        self.assertEqual(result["realtime_audit"]["attempts"], 2)
        self.assertEqual(len(result["realtime_audit"]["retry_codes"]), 1)
        self.assertEqual(opener.opens, 2)

    def test_unexpected_future_exception_stops_pending_calls_and_finalizes_state(self):
        class UnexpectedOnceClient(_FakeClient):
            def __init__(self):
                super().__init__()
                self._lock = threading.Lock()

            def complete(self, body):
                with self._lock:
                    self.calls += 1
                    ordinal = self.calls
                if ordinal == 1:
                    raise LookupError("unexpected worker implementation defect")
                time.sleep(0.02)
                return {"body": _provider_body(ordinal), "latency_ms": 2.0}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = self._many_request_dataset(root)
            source_dir = root / "source"
            run_dir = root / "run"
            self._prepare(data_dir, source_dir, limit=12)
            total = len(read_jsonl(source_dir / plan_cli.REQUEST_FILENAME))
            client = UnexpectedOnceClient()
            args = self._run_args(source_dir, run_dir)
            args.workers = 2
            environment = {
                "DASHSCOPE_API_KEY": "sk-test",
                "CONFIRM_ALIYUN_QWEN_DATA_UPLOAD": "YES",
                "CONFIRM_ALIYUN_QWEN_PAID_REALTIME": "YES",
            }
            with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
                realtime_cli.AliyunQwenRealtimeClient,
                "from_environment",
                return_value=client,
            ), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    realtime_cli.run_job(args)

            state = json.loads(
                (run_dir / realtime_cli.STATE_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(state["status"], "failed")
            self.assertLess(client.calls, total)
            self.assertLessEqual(client.calls, args.workers)
            failures = read_jsonl(run_dir / realtime_cli.FAILURE_FILENAME)
            self.assertEqual(len(failures), 1)
            self.assertTrue(failures[0]["fatal"])
            self.assertIn(failures[0]["custom_id"], {
                row["custom_id"]
                for row in read_jsonl(source_dir / plan_cli.REQUEST_FILENAME)
            })

    def test_run_job_bounds_submitted_in_flight_futures(self):
        class BlockingClient(_FakeClient):
            def __init__(self, release, started, workers):
                super().__init__()
                self.release = release
                self.started = started
                self.workers = workers
                self._lock = threading.Lock()

            def complete(self, body):
                with self._lock:
                    self.calls += 1
                    ordinal = self.calls
                    if self.calls >= self.workers:
                        self.started.set()
                if not self.release.wait(timeout=2.0):
                    raise AssertionError("test did not release blocked fake API call")
                return {"body": _provider_body(ordinal), "latency_ms": 1.0}

        class RecordingExecutor:
            instances = []

            def __init__(self, *args, **kwargs):
                self._inner = RealThreadPoolExecutor(*args, **kwargs)
                self._lock = threading.Lock()
                self.outstanding = 0
                self.max_outstanding = 0
                self.instances.append(self)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self._inner.shutdown(wait=True)
                return False

            def submit(self, *args, **kwargs):
                with self._lock:
                    self.outstanding += 1
                    self.max_outstanding = max(self.max_outstanding, self.outstanding)
                try:
                    future = self._inner.submit(*args, **kwargs)
                except BaseException:
                    with self._lock:
                        self.outstanding -= 1
                    raise

                def finished(_future):
                    with self._lock:
                        self.outstanding -= 1

                future.add_done_callback(finished)
                return future

            def shutdown(self, *args, **kwargs):
                return self._inner.shutdown(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = self._many_request_dataset(root)
            source_dir = root / "source"
            run_dir = root / "run"
            self._prepare(data_dir, source_dir, limit=12)
            release = threading.Event()
            started = threading.Event()
            args = self._run_args(source_dir, run_dir)
            args.workers = 2
            client = BlockingClient(release, started, args.workers)
            errors = []

            def invoke():
                try:
                    realtime_cli.run_job(args)
                except BaseException as exc:  # recorded and asserted in the test thread
                    errors.append(exc)

            environment = {
                "DASHSCOPE_API_KEY": "sk-test",
                "CONFIRM_ALIYUN_QWEN_DATA_UPLOAD": "YES",
                "CONFIRM_ALIYUN_QWEN_PAID_REALTIME": "YES",
            }
            with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
                realtime_cli.AliyunQwenRealtimeClient,
                "from_environment",
                return_value=client,
            ), mock.patch.object(
                realtime_cli, "ThreadPoolExecutor", RecordingExecutor
            ), contextlib.redirect_stdout(io.StringIO()):
                thread = threading.Thread(target=invoke, daemon=True)
                thread.start()
                try:
                    self.assertTrue(started.wait(timeout=1.0))
                    time.sleep(0.05)
                finally:
                    release.set()
                    thread.join(timeout=3.0)

            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(len(RecordingExecutor.instances), 1)
            self.assertLessEqual(
                RecordingExecutor.instances[0].max_outstanding,
                args.workers,
            )

    def test_running_state_heartbeats_while_no_result_is_available(self):
        class BlockingClient(_FakeClient):
            def __init__(self, release, entered):
                super().__init__()
                self.release = release
                self.entered = entered

            def complete(self, body):
                self.calls += 1
                ordinal = self.calls
                self.entered.set()
                if not self.release.wait(timeout=2.0):
                    raise AssertionError("test did not release blocked fake API call")
                return {"body": _provider_body(ordinal), "latency_ms": 1.0}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = self._dataset(root)
            source_dir = root / "source"
            run_dir = root / "run"
            self._prepare(data_dir, source_dir)
            release = threading.Event()
            entered = threading.Event()
            client = BlockingClient(release, entered)
            args = self._run_args(source_dir, run_dir)
            args.workers = 1
            args.heartbeat_seconds = 0.05
            errors = []

            def invoke():
                try:
                    realtime_cli.run_job(args)
                except BaseException as exc:  # recorded and asserted in the test thread
                    errors.append(exc)

            environment = {
                "DASHSCOPE_API_KEY": "sk-test",
                "CONFIRM_ALIYUN_QWEN_DATA_UPLOAD": "YES",
                "CONFIRM_ALIYUN_QWEN_PAID_REALTIME": "YES",
            }
            heartbeat_seen = False
            heartbeat_state = None
            with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
                realtime_cli.AliyunQwenRealtimeClient,
                "from_environment",
                return_value=client,
            ), contextlib.redirect_stdout(io.StringIO()):
                thread = threading.Thread(target=invoke, daemon=True)
                thread.start()
                try:
                    self.assertTrue(entered.wait(timeout=1.0))
                    state_path = run_dir / realtime_cli.STATE_FILENAME
                    initial = json.loads(state_path.read_text(encoding="utf-8"))
                    deadline = time.monotonic() + 0.6
                    while time.monotonic() < deadline:
                        candidate = json.loads(state_path.read_text(encoding="utf-8"))
                        if candidate["updated_at_utc"] != initial["updated_at_utc"]:
                            heartbeat_seen = True
                            heartbeat_state = candidate
                            break
                        time.sleep(0.01)
                    self.assertFalse(
                        (run_dir / realtime_cli.RESPONSE_FILENAME).exists()
                    )
                finally:
                    release.set()
                    thread.join(timeout=3.0)

            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue(heartbeat_seen)
            self.assertEqual(heartbeat_state["status"], "running")
            self.assertEqual(heartbeat_state["request_counts"]["completed"], 0)

    def test_429_is_retried_with_shared_cooldown(self):
        class OnceRateLimited(_FakeClient):
            def complete(self, body):
                self.calls += 1
                if self.calls == 1:
                    raise RealtimeAPIError(
                        "rate limited",
                        status_code=429,
                        code="Throttling",
                        retry_after_seconds=0.01,
                        retriable=True,
                    )
                return {"body": _provider_body(self.calls), "latency_ms": 4.0}

        limiter = _ImmediateLimiter()
        sleeps = []
        result, failure = realtime_cli._call_one(
            {
                "custom_id": "a" * 64,
                "body": {"unused": True},
            },
            client=OnceRateLimited(),
            limiter=limiter,
            max_attempts=3,
            inspection_max_attempts=3,
            backoff_base=1.0,
            backoff_max=30.0,
            stop_event=threading.Event(),
            sleep_fn=sleeps.append,
        )
        self.assertIsNone(failure)
        self.assertEqual(result["realtime_audit"]["attempts"], 2)
        self.assertEqual(result["realtime_audit"]["retry_codes"], ["Throttling"])
        self.assertEqual(limiter.acquires, 2)
        self.assertEqual(limiter.deferred, [0.01])
        self.assertEqual(sleeps, [0.01])

    def test_data_inspection_failure_becomes_bounded_nonfatal_abstention(self):
        class AlwaysInspectionBlocked(_FakeClient):
            def complete(self, body):
                self.calls += 1
                raise RealtimeAPIError(
                    "sanitized provider inspection rejection",
                    status_code=400,
                    code=realtime_cli.INSPECTION_ERROR_CODE,
                    retriable=False,
                )

        limiter = _ImmediateLimiter()
        sleeps = []
        client = AlwaysInspectionBlocked()
        result, failure = realtime_cli._call_one(
            {"custom_id": "b" * 64, "body": {"unused": True}},
            client=client,
            limiter=limiter,
            max_attempts=5,
            inspection_max_attempts=3,
            backoff_base=1.0,
            backoff_max=30.0,
            stop_event=threading.Event(),
            sleep_fn=sleeps.append,
        )
        self.assertIsNone(result)
        self.assertTrue(failure["provider_abstention"])
        self.assertFalse(failure["fatal"])
        self.assertEqual(failure["attempts"], 3)
        self.assertEqual(failure["policy"], realtime_cli.ABSTENTION_POLICY)
        self.assertEqual(client.calls, 3)
        self.assertEqual(limiter.acquires, 3)
        self.assertEqual(len(sleeps), 2)

    def test_unrelated_http_400_remains_fatal(self):
        class InvalidRequest(_FakeClient):
            def complete(self, body):
                self.calls += 1
                raise RealtimeAPIError(
                    "sanitized invalid request",
                    status_code=400,
                    code="invalid_parameter",
                    retriable=False,
                )

        result, failure = realtime_cli._call_one(
            {"custom_id": "c" * 64, "body": {"unused": True}},
            client=InvalidRequest(),
            limiter=_ImmediateLimiter(),
            max_attempts=5,
            inspection_max_attempts=3,
            backoff_base=1.0,
            backoff_max=30.0,
            stop_event=threading.Event(),
            sleep_fn=lambda _seconds: None,
        )
        self.assertIsNone(result)
        self.assertTrue(failure["fatal"])
        self.assertNotIn("provider_abstention", failure)
        self.assertEqual(failure["attempts"], 1)

    def test_concurrent_run_is_complete_resume_safe_and_secret_free(self):
        secret = "sk-runtime-secret-never-persist"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = self._dataset(root)
            source_dir = root / "source"
            run_dir = root / "run"
            self._prepare(data_dir, source_dir)
            client = _FakeClient()
            environment = {
                "DASHSCOPE_API_KEY": secret,
                "CONFIRM_ALIYUN_QWEN_DATA_UPLOAD": "YES",
                "CONFIRM_ALIYUN_QWEN_PAID_REALTIME": "YES",
            }
            with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
                realtime_cli.AliyunQwenRealtimeClient,
                "from_environment",
                return_value=client,
            ), contextlib.redirect_stdout(io.StringIO()):
                first = realtime_cli.run_job(self._run_args(source_dir, run_dir))
                second = realtime_cli.run_job(self._run_args(source_dir, run_dir))

            self.assertEqual(first["status"], "completed")
            self.assertEqual(second["status"], "completed")
            self.assertEqual(client.calls, 2)
            rows = read_jsonl(run_dir / realtime_cli.RESPONSE_FILENAME)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({row["custom_id"] for row in rows}), 2)
            for path in run_dir.iterdir():
                if path.is_file():
                    self.assertNotIn(secret, path.read_text(encoding="utf-8"))
                    self.assertNotIn("Hidden_Target_SENTINEL", path.read_text(encoding="utf-8"))

    def test_offline_collect_produces_loadable_realtime_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = self._dataset(root)
            source_dir = root / "source"
            run_dir = root / "run"
            output = root / "cache.jsonl"
            self._prepare(data_dir, source_dir)
            client = _FakeClient()
            with mock.patch.dict(
                os.environ,
                {
                    "DASHSCOPE_API_KEY": "sk-test",
                    "CONFIRM_ALIYUN_QWEN_DATA_UPLOAD": "YES",
                    "CONFIRM_ALIYUN_QWEN_PAID_REALTIME": "YES",
                },
                clear=False,
            ), mock.patch.object(
                realtime_cli.AliyunQwenRealtimeClient,
                "from_environment",
                return_value=client,
            ), contextlib.redirect_stdout(io.StringIO()):
                realtime_cli.run_job(self._run_args(source_dir, run_dir))
                args = realtime_cli.build_parser().parse_args(
                    [
                        "collect",
                        "--source-job-dir",
                        str(source_dir),
                        "--run-dir",
                        str(run_dir),
                        "--data-dir",
                        str(data_dir),
                        "--output",
                        str(output),
                        "--allow-incomplete-cache",
                    ]
                )
                result = realtime_cli.collect_job(args)
            self.assertEqual(result["records"], 2)
            cache = LLMEvidenceCache(str(output), require_generation_metadata=True)
            metadata = cache.metadata()["generation_metadata"]
            self.assertEqual(metadata["provider"], "aliyun_qwen_realtime")
            self.assertTrue(metadata["generation_audit"]["per_request_latency_available"])
            self.assertEqual(metadata["generation_audit"]["retry_count"], 0)
            self.assertGreater(metadata["generation_audit"]["estimated_list_price_cny"], 0)

    def test_abstention_run_resumes_and_collects_empty_side_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = self._dataset(root)
            source_dir = root / "source"
            run_dir = root / "run"
            output = root / "cache.jsonl"
            self._prepare(data_dir, source_dir)
            client = _InspectionBlockedThenSuccessClient()
            run_args = self._run_args(source_dir, run_dir)
            run_args.workers = 1
            run_args.backoff_base = 0.0
            run_args.backoff_max = 0.0
            environment = {
                "DASHSCOPE_API_KEY": "sk-test",
                "CONFIRM_ALIYUN_QWEN_DATA_UPLOAD": "YES",
                "CONFIRM_ALIYUN_QWEN_PAID_REALTIME": "YES",
            }
            with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
                realtime_cli.AliyunQwenRealtimeClient,
                "from_environment",
                return_value=client,
            ), contextlib.redirect_stdout(io.StringIO()):
                first = realtime_cli.run_job(run_args)
                second = realtime_cli.run_job(run_args)

            self.assertEqual(first["status"], "completed")
            self.assertEqual(first["request_counts"]["successful_responses"], 1)
            self.assertEqual(first["request_counts"]["provider_abstentions"], 1)
            self.assertEqual(second["request_counts"], first["request_counts"])
            self.assertEqual(client.calls, 4)
            self.assertEqual(
                len(read_jsonl(run_dir / realtime_cli.ABSTENTION_FILENAME)), 1
            )
            self.assertEqual(len(read_jsonl(run_dir / realtime_cli.RESPONSE_FILENAME)), 1)

            collect_args = realtime_cli.build_parser().parse_args(
                [
                    "collect",
                    "--source-job-dir",
                    str(source_dir),
                    "--run-dir",
                    str(run_dir),
                    "--data-dir",
                    str(data_dir),
                    "--output",
                    str(output),
                    "--allow-incomplete-cache",
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = realtime_cli.collect_job(collect_args)
            self.assertEqual(result["provider_abstentions"], 1)
            records = read_jsonl(output)
            abstained = [
                row
                for row in records
                if row["diagnostics"].get("provider_abstention") == 1.0
            ]
            self.assertEqual(len(abstained), 1)
            self.assertEqual(abstained[0]["candidates"], [])
            self.assertIsNone(abstained[0]["response_id"])
            cache = LLMEvidenceCache(str(output), require_generation_metadata=True)
            audit = cache.metadata()["generation_metadata"]["generation_audit"]
            self.assertEqual(audit["provider_abstention_count"], 1)
            self.assertEqual(
                audit["provider_abstention_policy"], realtime_cli.ABSTENTION_POLICY
            )
            self.assertFalse(audit["per_request_latency_available"])

    def test_offline_estimate_uses_reviewed_plan_without_client(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = self._dataset(root)
            source_dir = root / "source"
            self._prepare(data_dir, source_dir)
            args = realtime_cli.build_parser().parse_args(
                ["estimate", "--source-job-dir", str(source_dir)]
            )
            with mock.patch.object(
                realtime_cli.AliyunQwenRealtimeClient,
                "from_environment",
                side_effect=AssertionError("offline estimate constructed a client"),
            ), contextlib.redirect_stdout(io.StringIO()):
                result = realtime_cli.estimate_job(args)
            self.assertFalse(result["network_called"])
            self.assertEqual(result["request_count"], 2)
            self.assertGreater(result["buffered_estimated_cost_cny"], 0)


if __name__ == "__main__":
    unittest.main()
