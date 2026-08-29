#!/usr/bin/env python3
"""Run pinned vLLM 0.6.3 without externally reachable service sockets.

The upstream 0.6.3 API entrypoint pre-binds ``("", port)`` before passing the
socket to uvicorn, which widens an explicitly requested 127.0.0.1 bind to all
interfaces. This small pinned-version wrapper keeps the normal vLLM engine and
OpenAI-compatible app, but lets uvicorn bind the requested loopback address
directly. Its single-GPU executor also defaults to a wildcard-bound PyTorch
TCPStore even at world size one, so the wrapper replaces only that private,
pinned-version rendezvous with a process-local FileStore.
"""

from __future__ import annotations

import atexit
import importlib.metadata
import os
from pathlib import Path
import uuid


def _refuse_pre_reboot_startup() -> None:
    state_text = os.environ.get("LOCAL_QWEN_STATE_DIR", "")
    if not state_text:
        return
    state_dir = Path(state_text).resolve()
    if state_dir.parent.name != "servers":
        return
    inhibit = state_dir.parent.parent / "PRE_REBOOT_CHECKPOINT.lock"
    if inhibit.exists():
        raise SystemExit(f"pre-reboot checkpoint inhibits local-Qwen startup: {inhibit}")


_refuse_pre_reboot_startup()

import pynvml
import uvloop


def _visible_gpu_fallback_handle(device_id: int):
    """Ignore an unrelated broken GPU during vLLM's import-time warning.

    vLLM 0.6.3 enumerates every physical GPU while importing its CUDA
    platform, even though this wrapper deliberately exposes one GPU through
    ``CUDA_VISIBLE_DEVICES``.  A single NVML-broken card would therefore stop
    healthy single-GPU servers from starting.  During the import only, map an
    unqueryable *unrelated* device to the selected healthy card.  A failure on
    the selected card is never hidden.
    """

    try:
        return _ORIGINAL_NVML_GET_HANDLE(device_id)
    except pynvml.NVMLError:
        if _VISIBLE_PHYSICAL_GPU is None or device_id == _VISIBLE_PHYSICAL_GPU:
            raise
        return _ORIGINAL_NVML_GET_HANDLE(_VISIBLE_PHYSICAL_GPU)


_ORIGINAL_NVML_GET_HANDLE = pynvml.nvmlDeviceGetHandleByIndex
_visible_token = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
_VISIBLE_PHYSICAL_GPU = int(_visible_token) if _visible_token.isdigit() else None
pynvml.nvmlDeviceGetHandleByIndex = _visible_gpu_fallback_handle
try:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.entrypoints.launcher import serve_http
    from vllm.entrypoints.openai.api_server import (
        TIMEOUT_KEEP_ALIVE,
        build_app,
        build_async_engine_client,
        init_app_state,
    )
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils import FlexibleArgumentParser
finally:
    pynvml.nvmlDeviceGetHandleByIndex = _ORIGINAL_NVML_GET_HANDLE


SUPPORTED_VLLM_VERSION = "0.6.3.post1"
_RENDEZVOUS_PATH: Path | None = None


def _single_gpu_filestore_method() -> str:
    global _RENDEZVOUS_PATH
    if _RENDEZVOUS_PATH is None:
        directory = Path(
            os.environ.get("LOCAL_QWEN_RDZV_DIR", "logs/local_qwen/rendezvous")
        ).resolve()
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)
        _RENDEZVOUS_PATH = directory / (
            f"vllm-{os.getpid()}-{uuid.uuid4().hex}.store"
        )

        def cleanup() -> None:
            try:
                _RENDEZVOUS_PATH.unlink(missing_ok=True)
            except OSError:
                pass

        atexit.register(cleanup)
    return _RENDEZVOUS_PATH.as_uri()


def install_single_gpu_filestore() -> None:
    """Patch the pinned single-GPU executor before an engine is constructed."""

    installed_version = importlib.metadata.version("vllm")
    if installed_version != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            "The loopback/FileStore wrapper is audited only for vLLM "
            f"{SUPPORTED_VLLM_VERSION}; found {installed_version}"
        )
    from vllm.executor.gpu_executor import GPUExecutor

    original = GPUExecutor._get_worker_kwargs
    if getattr(original, "_ninefuse_filestore_patch", False):
        return

    def get_worker_kwargs(
        executor,
        local_rank: int = 0,
        rank: int = 0,
        distributed_init_method=None,
    ):
        if distributed_init_method is None:
            if executor.parallel_config.world_size != 1:
                raise RuntimeError("FileStore override is restricted to a single-GPU executor")
            distributed_init_method = _single_gpu_filestore_method()
        return original(
            executor,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
        )

    get_worker_kwargs._ninefuse_filestore_patch = True
    GPUExecutor._get_worker_kwargs = get_worker_kwargs


async def run_loopback_server(args) -> None:
    if args.host != "127.0.0.1":
        raise ValueError("The local Qwen service must bind exactly to 127.0.0.1")

    async with build_async_engine_client(args) as engine_client:
        app = build_app(args)
        model_config = await engine_client.get_model_config()
        init_app_state(engine_client, model_config, app.state, args)
        shutdown_task = await serve_http(
            app,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
        )
    await shutdown_task


def main() -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    install_single_gpu_filestore()
    parser = FlexibleArgumentParser(
        description="Loopback-only vLLM OpenAI-compatible server"
    )
    parser.add_argument("model_tag", help="Local model directory")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    if args.model != EngineArgs.model:
        raise ValueError("Pass the model directory as the positional argument")
    args.model = args.model_tag
    uvloop.run(run_loopback_server(args))


if __name__ == "__main__":
    main()
