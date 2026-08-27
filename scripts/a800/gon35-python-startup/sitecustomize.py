"""Give every GON-35 Python process private compiler caches.

The A800 artifact filesystem is shared by all workers. TorchInductor and
Triton can concurrently replace identical cache keys, which is unsafe on the
mounted filesystem. Python imports ``sitecustomize`` before application code,
so this module isolates both newly spawned interpreters and forked children
without changing VERL or the scientific entry script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _isolate_compiler_caches() -> None:
    root = os.environ.get("GON35_COMPILER_CACHE_ROOT")
    if not root:
        return

    process_root = Path(root) / f"pid-{os.getpid()}"
    locations = {
        "TRITON_CACHE_DIR": process_root / "triton",
        "TORCHINDUCTOR_CACHE_DIR": process_root / "torchinductor",
    }
    for variable, path in locations.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)

    # A fork may inherit TorchInductor's memoized cache root from its parent.
    cache_module = sys.modules.get("torch._inductor.runtime.cache_dir")
    cache_dir = getattr(cache_module, "cache_dir", None)
    cache_clear = getattr(cache_dir, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


def _isolate_or_exit() -> None:
    try:
        _isolate_compiler_caches()
    except BaseException as error:
        message = f"FATAL: GON-35 compiler-cache isolation failed: {error}\n"
        try:
            os.write(2, message.encode(errors="backslashreplace"))
        finally:
            # Exceptions from sitecustomize and at-fork callbacks are otherwise
            # non-fatal. Never continue with an inherited shared cache.
            os._exit(70)


_isolate_or_exit()
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_isolate_or_exit)
