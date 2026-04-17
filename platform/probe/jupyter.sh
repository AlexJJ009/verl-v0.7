#!/usr/bin/env bash
# Smoke probe for the verl image (MLP custom image based on
# ai-search_training_ubuntu22_cuda12.8_python3.12_torch2.8_verl_megatron_1.0.2_534ef92c
# or its derivative with verl deltas baked in).
#
# Two run modes, same script:
#   1. Against the UNMODIFIED base image — discovery mode. Report what is
#      present vs missing so we know the minimal Dockerfile delta to layer on.
#   2. Against the FINAL verl image — validation mode. All verl-required
#      packages should be importable with working CUDA ABI.
#
# Design:
#   * Items promised by the base image tags (torch/vllm/sglang/transformers/ray/
#     python versions, cuda runtime) are HARD-ASSERTED — broken == FAIL.
#   * Items that may or may not be in the base (flash-attn, apex, TE, megatron,
#     mbridge, deepspeed, trl, verl deltas) are REPORTED — missing is not fatal,
#     but broken (present-but-unimportable or CUDA ABI broken) IS fatal.
#   * Every CUDA ABI extension present gets a kernel-roundtrip smoke.
#   * Full `pip list` + `pip check` dumped to disk for offline inspection.
#
# Exit code:
#   0  -> all required checks passed
#   1  -> one or more required checks failed (see "SUMMARY")

set -uo pipefail
set -x

# --- Resolve LGX_DIR (dolphinfs anchor) from script location. Mirrors the
# pattern in dpo-experiment/platform/probe/jupyter.sh so the file is relocatable.
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" 2>/dev/null && pwd)"
if [ -z "${SCRIPT_DIR}" ]; then SCRIPT_DIR="$(pwd)"; fi
LGX_DIR=""
candidate="${SCRIPT_DIR}"
for _ in 1 2 3 4; do
  candidate="$(cd "${candidate}/.." 2>/dev/null && pwd)" || break
  # New anchor: verl-exp (verl repo on dolphinfs) OR hope_dir (submission staging)
  if [ -d "${candidate}/verl-exp" ] || [ -d "${candidate}/dpo-exp" ] || [ -d "${candidate}/hope_dir" ]; then
    LGX_DIR="${candidate}"; break
  fi
done
if [ -z "${LGX_DIR}" ]; then LGX_DIR="$(cd "${SCRIPT_DIR}/.." 2>/dev/null && pwd)"; fi
REPO_DIR_GUESS="${LGX_DIR}/verl-exp"

LOG_ROOT=""
if mkdir -p "${LGX_DIR}/logs" 2>/dev/null && [ -w "${LGX_DIR}/logs" ]; then
  LOG_ROOT="${LGX_DIR}/logs"
fi
[ -z "${LOG_ROOT}" ] && { LOG_ROOT="/tmp/probe_logs"; mkdir -p "${LOG_ROOT}"; }
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_ROOT}/verl_probe_${TS}_$$.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "================================================================"
echo "=== VERL PROBE START $(date -Is) on $(hostname)"
echo "=== LOG_FILE=${LOG_FILE}"
echo "================================================================"

FAILS=()
record_fail() { FAILS+=("$1"); echo "  [FAIL] $1"; }
record_ok()   { echo "  [OK]   $1"; }
record_info() { echo "  [info] $1"; }

section() { echo; echo "---------- $1 ----------"; }

# ==================== A. host & hardware ====================
section "A. identity / host"
whoami || true
id || true
uname -a || true
cat /etc/os-release 2>/dev/null | head -5 || true
nproc || true
echo "--- /proc/meminfo (top) ---"
head -5 /proc/meminfo || true

section "A. GPU"
if which nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || record_fail "nvidia-smi exited non-zero"
else
  record_fail "nvidia-smi not on PATH"
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
ls /dev/nvidia* 2>/dev/null || record_fail "no /dev/nvidia* device nodes"

section "A. shm / disk"
df -h /dev/shm || true
df -h / /tmp 2>&1 | head -10 || true
echo "--- NIC (for distributed training) ---"
(ip -br addr show 2>/dev/null || ifconfig -a 2>/dev/null) | head -20 || true
echo "--- IB / RDMA devices ---"
ls /dev/infiniband 2>/dev/null || echo "(no /dev/infiniband)"
which ibv_devinfo >/dev/null 2>&1 && ibv_devinfo -l 2>&1 | head -10 || echo "(no ibv_devinfo)"

# ==================== B. runtime ENV ====================
section "B. runtime ENV (report only — no assertions yet, use output to decide what to bake)"
for name in PYTHONPATH HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE \
            VLLM_USE_V1 VLLM_NO_USAGE_STATS VLLM_DO_NOT_TRACK VLLM_DOWNLOAD_DIR \
            RAY_ADDRESS RAY_DISABLE_IMPORT_WARNING \
            NCCL_DEBUG NCCL_IB_DISABLE NCCL_SOCKET_IFNAME \
            CUDA_HOME CUDA_VISIBLE_DEVICES LD_LIBRARY_PATH \
            NVSHMEM_DIR GDRCOPY_HOME \
            OMP_NUM_THREADS TOKENIZERS_PARALLELISM \
            VIRTUAL_ENV PATH; do
  v="${!name:-<unset>}"
  record_info "${name}=${v}"
done

# ==================== C/D/E/F. Python matrix + smokes ====================
section "C-N. Python dependency matrix + functional smokes"
PY_REPORT="${LOG_ROOT}/verl_probe_${TS}_py.txt"
PIP_LIST="${LOG_ROOT}/verl_probe_${TS}_piplist.txt"
PIP_CHECK="${LOG_ROOT}/verl_probe_${TS}_pipcheck.txt"

echo "--- dumping pip list to ${PIP_LIST} ---"
pip list --format=columns > "${PIP_LIST}" 2>&1 || echo "pip list failed"
echo "--- dumping pip check to ${PIP_CHECK} ---"
pip check > "${PIP_CHECK}" 2>&1 || echo "pip check reported issues (see file)"
echo "--- pip list ---"
cat "${PIP_LIST}" | head -200
echo "--- (truncated — full dump at ${PIP_LIST}) ---"
wc -l "${PIP_LIST}" "${PIP_CHECK}" 2>/dev/null || true

python3 - <<'PY' | tee "${PY_REPORT}"
import sys, os, importlib, traceback

def line_ok(tag):      print(f"  [OK]   {tag}")
def line_fail(tag):    print(f"  [FAIL] {tag}")
def line_info(tag):    print(f"  [info] {tag}")
def line_missing(tag): print(f"  [miss] {tag}")

def probe(mod, required=False, want=None, op="=="):
    """Import mod, report version. required=True means FAIL if missing or broken."""
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        if want is None:
            line_ok(f"{mod:<28} == {ver}")
            return m, ver
        if op == "==":
            ok = (ver == want)
        elif op == ">=":
            def t(s):
                s = s.split('+')[0].split('.')
                return tuple(int(x) for x in s if x.isdigit())
            ok = t(ver) >= t(want)
        else:
            ok = False
        (line_ok if ok else line_fail)(f"{mod} {op} {want} (got {ver})")
        return m, ver
    except Exception as e:
        if required:
            line_fail(f"{mod} import error: {e!r}")
        else:
            line_missing(f"{mod} not present ({type(e).__name__})")
        return None, None

# ==================== C. Python interpreter ====================
print("---- C. python interpreter ----")
line_info(f"python: {sys.version.split()[0]}  exec: {sys.executable}")
line_info(f"prefix: {sys.prefix}")
line_info(f"sys.path[0:3]: {sys.path[:3]}")

# ==================== D. Versions promised by the base image tags ====================
# Tag line from user's paste:
#   python3.12.11 cuda12.8 torch2.8.0+cu128 ubuntu22.04 glibc2.35
#   java11.0.28 ray2.50.0 sglang0.5.4 transformers4.57.1 vllm0.10.2
print("---- D. tag-promised packages (required by base image advertisement) ----")
probe("torch",        required=True, want="2.8",    op=">=")
probe("vllm",         required=True, want="0.10",   op=">=")
probe("sglang",       required=True)
probe("transformers", required=True, want="4.57",   op=">=")
probe("ray",          required=True, want="2.50",   op=">=")

# ==================== E. verl delta deps (REPORT — may or may not be in base) ====================
# Sourced from /data-1/verl07/verl/docker/Dockerfile.joint-training.cu126
# and requirements.txt / requirements-cuda.txt.
print("---- E. verl delta: CUDA-compiled extensions (critical for training) ----")
probe("flash_attn")              # flash-attn
probe("apex")                    # NVIDIA apex
probe("transformer_engine")      # TE
probe("transformer_engine.pytorch")
probe("megatron.core")           # Megatron-LM
probe("mbridge")                 # ISEEKYAN/mbridge
probe("deep_ep")                 # DeepEP (NVIDIA deepseek-ai)
probe("nvidia.nvshmem", required=False)

print("---- E. verl delta: pure-Python deps ----")
for mod in [
    "accelerate", "peft", "datasets", "tensordict", "liger_kernel",
    "deepspeed", "trl", "hydra", "codetiming", "dill",
    "pylatexenc", "latex2sympy2_extended", "math_verify", "mathruler",
    "qwen_vl_utils", "nvtx", "matplotlib", "fastapi", "uvicorn",
    "pybind11", "pyarrow", "pandas", "numpy", "tensorboard", "wandb",
    "torchdata", "torchvision", "torchaudio",
]:
    probe(mod)

# ==================== F. verl itself (may be preinstalled from the verl_megatron base) ====================
print("---- F. verl package ----")
verl_mod, verl_ver = probe("verl")
if verl_mod is not None:
    line_info(f"verl.__file__: {verl_mod.__file__}")
    # Try a few well-known submodule imports.
    for sub in ["verl.trainer", "verl.workers", "verl.protocol",
                "verl.single_controller", "verl.utils"]:
        try:
            importlib.import_module(sub)
            line_ok(f"submodule {sub} imports")
        except Exception as e:
            line_fail(f"submodule {sub} failed: {e!r}")

# ==================== G. CUDA kernel roundtrip ====================
print("---- G. CUDA runtime smoke ----")
try:
    import torch
    if not torch.cuda.is_available():
        line_fail("torch.cuda.is_available() is False")
    else:
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        cap  = torch.cuda.get_device_capability(0)
        line_ok(f"torch.cuda: {n} dev(s), dev0={name} sm={cap[0]}.{cap[1]}")
        line_info(f"torch.version.cuda={torch.version.cuda}  cudnn={torch.backends.cudnn.version()}")
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x).sum().item()
        line_ok(f"cuda matmul smoke: sum={y:.3e}")
        torch.cuda.synchronize()
        line_ok("torch.cuda.synchronize() ok")
        # bf16 smoke (verl training uses bf16)
        xb = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        yb = (xb @ xb).float().sum().item()
        line_ok(f"cuda bf16 matmul smoke: sum={yb:.3e}")
except Exception as e:
    line_fail(f"cuda smoke failed: {e!r}")
    traceback.print_exc()

# ==================== H. flash_attn functional ====================
print("---- H. flash_attn functional smoke ----")
try:
    import torch
    import flash_attn
    from flash_attn import flash_attn_func
    # (batch, seqlen, nheads, headdim); bf16/fp16 only
    B, S, H, D = 2, 64, 4, 64
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    out = flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    line_ok(f"flash_attn_func ok, out.shape={tuple(out.shape)}, ver={flash_attn.__version__}")
except ModuleNotFoundError:
    line_missing("flash_attn not installed")
except Exception as e:
    line_fail(f"flash_attn functional failed: {e!r}")
    traceback.print_exc()

# ==================== I. TransformerEngine functional ====================
print("---- I. TransformerEngine functional smoke ----")
try:
    import torch
    import transformer_engine.pytorch as te
    # fp8 requires SM>=8.9; bf16 path always works on Ampere+
    with torch.cuda.device(0):
        linear = te.Linear(512, 512, bias=True).cuda().to(torch.bfloat16)
        x = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
        y = linear(x)
        torch.cuda.synchronize()
        line_ok(f"te.Linear fwd ok, out.shape={tuple(y.shape)}")
except ModuleNotFoundError:
    line_missing("transformer_engine not installed")
except Exception as e:
    line_fail(f"transformer_engine functional failed: {e!r}")
    traceback.print_exc()

# ==================== J. apex FusedAdam functional ====================
print("---- J. apex FusedAdam functional smoke ----")
try:
    import torch
    from apex.optimizers import FusedAdam
    w = torch.randn(128, 128, device="cuda", requires_grad=True)
    opt = FusedAdam([w], lr=1e-3)
    loss = (w ** 2).sum()
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    line_ok(f"apex FusedAdam step ok, loss={loss.item():.3e}")
except ModuleNotFoundError:
    line_missing("apex not installed")
except Exception as e:
    line_fail(f"apex FusedAdam failed: {e!r}")
    traceback.print_exc()

# ==================== K. megatron.core ====================
print("---- K. megatron.core smoke ----")
try:
    from megatron.core import parallel_state, tensor_parallel  # noqa: F401
    import megatron.core as mc
    line_ok(f"megatron.core import ok (version={getattr(mc, '__version__', '?')})")
except ModuleNotFoundError:
    line_missing("megatron.core not installed")
except Exception as e:
    line_fail(f"megatron.core import failed: {e!r}")
    traceback.print_exc()

# ==================== L. DeepEP ====================
print("---- L. DeepEP smoke ----")
try:
    import deep_ep
    line_ok(f"deep_ep import ok (version={getattr(deep_ep, '__version__', '?')})")
except ModuleNotFoundError:
    line_missing("deep_ep not installed")
except Exception as e:
    line_fail(f"deep_ep import failed: {e!r}")
    traceback.print_exc()

# ==================== M. vLLM import-only smoke (no engine) ====================
print("---- M. vLLM import smoke (no engine spin-up) ----")
try:
    from vllm import LLM, SamplingParams  # noqa: F401
    import vllm
    sp = SamplingParams(temperature=0.0, max_tokens=16)
    line_ok(f"vllm.LLM / SamplingParams importable (vllm {vllm.__version__}, sp.max_tokens={sp.max_tokens})")
except Exception as e:
    line_fail(f"vllm import failed: {e!r}")
    traceback.print_exc()

# ==================== N. sglang import smoke ====================
print("---- N. sglang import smoke ----")
try:
    import sglang
    line_ok(f"sglang import ok (version={sglang.__version__})")
except Exception as e:
    line_fail(f"sglang import failed: {e!r}")
PY
PY_RC=$?
if [ "${PY_RC}" -ne 0 ]; then
  record_fail "python probe block exited rc=${PY_RC}"
fi

# ==================== O. nvshmem / gdrcopy filesystem ====================
section "O. nvshmem / gdrcopy filesystem presence (for DeepEP)"
for p in \
  /usr/local/lib/libgdrapi.so \
  /usr/local/lib/libgdrapi.so.2 \
  /opt/venv/lib/python3.12/site-packages/nvidia/nvshmem/lib/libnvshmem_host.so \
  /opt/venv/lib/python3.12/site-packages/nvidia/nvshmem/lib/libnvshmem_host.so.3 \
  /opt/venv/lib/python3.12/site-packages/nvidia/nvshmem; do
  if [ -e "$p" ]; then record_info "present: $p"; else record_info "absent:  $p"; fi
done
# find the actual venv site-packages if /opt/venv is not the path
SITEPKG=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
if [ -n "${SITEPKG}" ] && [ "${SITEPKG}" != "/opt/venv/lib/python3.12/site-packages" ]; then
  record_info "actual site-packages: ${SITEPKG}"
  ls -d "${SITEPKG}/nvidia/nvshmem" 2>/dev/null && record_info "nvshmem dir found under actual sitepkg"
fi

# Count [FAIL] lines from the python block
PY_FAILS=$(grep -c "^  \[FAIL\] " "${PY_REPORT}" 2>/dev/null || echo 0)
if [ "${PY_FAILS}" -gt 0 ]; then
  record_fail "python checks: ${PY_FAILS} failure(s) — see ${PY_REPORT}"
fi

# ==================== P. verl source mount (dolphinfs) ====================
section "P. verl source on dolphinfs (informational — runtime mount target)"
if [ -d "${REPO_DIR_GUESS}" ]; then
  record_info "verl-exp/ found at ${REPO_DIR_GUESS}"
  (cd "${REPO_DIR_GUESS}" && git log -1 --oneline 2>&1) || true
  ls "${REPO_DIR_GUESS}" 2>&1 | head -20 || true
  # Try PYTHONPATH import of verl from the mounted source
  PYTHONPATH="${REPO_DIR_GUESS}:${PYTHONPATH:-}" python3 -c "
import sys
sys.path.insert(0, '${REPO_DIR_GUESS}')
try:
    import verl
    print(f'  [OK]   verl from mounted source: __version__={getattr(verl, \"__version__\", \"?\")}, file={verl.__file__}')
except Exception as e:
    print(f'  [FAIL] verl from mounted source failed: {e!r}')
" || record_fail "verl mounted-source import failed"
else
  record_info "verl-exp/ NOT on dolphinfs yet — upload before real runs"
  record_info "(expected at ${REPO_DIR_GUESS})"
fi

# ==================== Q. beacon ====================
section "Q. beacon"
if [ -d "${LGX_DIR}" ]; then
  mkdir -p "${LGX_DIR}/beacons" 2>/dev/null || true
  BEACON="${LGX_DIR}/beacons/verl_probe_$(hostname)_$(date +%s).txt"
  {
    echo "verl probe alive at $(date -Is)"
    echo "host: $(hostname)"
    echo "LGX_DIR: ${LGX_DIR}"
    echo "fails: ${#FAILS[@]}"
    echo "log:   ${LOG_FILE}"
    echo "py_report: ${PY_REPORT}"
    echo "pip_list:  ${PIP_LIST}"
    echo "pip_check: ${PIP_CHECK}"
  } > "${BEACON}" 2>&1 && record_ok "beacon written: ${BEACON}"
fi

# ==================== SUMMARY ====================
section "SUMMARY"
echo "Artifacts:"
echo "  log:       ${LOG_FILE}"
echo "  py report: ${PY_REPORT}"
echo "  pip list:  ${PIP_LIST}"
echo "  pip check: ${PIP_CHECK}"
echo
if [ "${#FAILS[@]}" -eq 0 ]; then
  echo "ALL PROBE CHECKS PASSED"
  PROBE_RC=0
else
  echo "PROBE FAILURES (${#FAILS[@]}):"
  for f in "${FAILS[@]}"; do echo "  - $f"; done
  PROBE_RC=1
fi

echo "=== sleeping 300s for UI inspection ==="
sleep 300
echo "=== VERL PROBE END $(date -Is) rc=${PROBE_RC} ==="
exit ${PROBE_RC}
