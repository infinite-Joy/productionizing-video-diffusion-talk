#!/usr/bin/env bash
# run_backend_comparison.sh
# Run compare_wan_backends.py with --backend inductor, then --backend tensorrt,
# then call --compare and diff the two captured log files.
#
# Usage:
#   ./run_backend_comparison.sh [extra args passed to both backend runs]
#
# Common overrides:
#   ./run_backend_comparison.sh --steps 1 --mode max-autotune
#   ./run_backend_comparison.sh --no-dryrun   # triggers real TRT engine build
#
# By default the TensorRT run uses --dryrun (partition report, no engine build).
# Pass --no-dryrun to force the full engine build.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SCRIPT_DIR}/compare_wan_backends.py"
OUTDIR="${BACKEND_OUTDIR:-./backend_debug}"

# ---------------------------------------------------------------------------
# uv setup
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "uv not found. Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

VENV_DIR="${SCRIPT_DIR}/.venv"

echo "[uv] Syncing dependencies from pyproject.toml..."
uv sync --project "${SCRIPT_DIR}"

PY="${VENV_DIR}/bin/python"

# Suppress noisy but harmless deprecation warnings from torchao / torch internals.
# export PYTHONWARNINGS="ignore::UserWarning:torchao,ignore::DeprecationWarning:torch"

# Split out --no-dryrun from the extra args; everything else is forwarded.
DRYRUN_FLAG="--dryrun"
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--no-dryrun" ]]; then
        DRYRUN_FLAG=""
    else
        EXTRA_ARGS+=("$arg")
    fi
done

LOG_IND="${OUTDIR}/run_inductor.log"
LOG_TRT="${OUTDIR}/run_tensorrt.log"
LOG_CMP="${OUTDIR}/compare.log"

mkdir -p "${OUTDIR}"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
banner() { printf '\n%s\n%s\n%s\n' "$(printf '=%.0s' {1..78})" "$1" "$(printf '=%.0s' {1..78})"; }

# ---------------------------------------------------------------------------
# 1) Inductor
# ---------------------------------------------------------------------------
banner "STEP 1 — Inductor backend"
echo "Logging to: ${LOG_IND}"
"${PY}" "${TARGET}" \
    --backend inductor \
    --outdir "${OUTDIR}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    2>&1 | tee "${LOG_IND}"

# ---------------------------------------------------------------------------
# 2) TensorRT  (non-fatal: torch-tensorrt may not be installed)
# ---------------------------------------------------------------------------
banner "STEP 2 — TensorRT backend${DRYRUN_FLAG:+ (dryrun)}"
echo "Logging to: ${LOG_TRT}"
TRT_OK=true
{ "${PY}" "${TARGET}" \
    --backend tensorrt \
    --outdir "${OUTDIR}" \
    ${DRYRUN_FLAG} \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    2>&1 | tee "${LOG_TRT}"; } || TRT_OK=false

if ! $TRT_OK; then
    echo ""
    echo "WARNING: TensorRT step failed (see ${LOG_TRT} for details)."
    echo "  If torch-tensorrt is not installed, run:"
    echo "    uv sync --extra tensorrt"
    echo "  or manually: pip install torch-tensorrt  (must match torch/CUDA/TRT versions)"
fi

# ---------------------------------------------------------------------------
# 3) Structured comparison (built-in JSON diff) — only if both summaries exist
# ---------------------------------------------------------------------------
IND_JSON="${OUTDIR}/summary_inductor.json"
TRT_JSON="${OUTDIR}/summary_tensorrt.json"
if [[ -f "${IND_JSON}" && -f "${TRT_JSON}" ]]; then
    banner "STEP 3 — Structured comparison (--compare)"
    echo "Logging to: ${LOG_CMP}"
    "${PY}" "${TARGET}" \
        --compare \
        --outdir "${OUTDIR}" \
        2>&1 | tee "${LOG_CMP}"
else
    banner "STEP 3 — Skipped (missing inductor or tensorrt summary JSON)"
    [[ -f "${IND_JSON}" ]] || echo "  missing: ${IND_JSON}"
    [[ -f "${TRT_JSON}" ]] || echo "  missing: ${TRT_JSON}"
fi

# ---------------------------------------------------------------------------
# 4) Raw log diff  (only when both logs exist)
# ---------------------------------------------------------------------------
if [[ -f "${LOG_IND}" && -f "${LOG_TRT}" ]]; then
    banner "STEP 4 — Raw log diff (inductor vs tensorrt stdout)"
    DIFF_OUT="${OUTDIR}/log_diff.txt"
    diff --unified=3 "${LOG_IND}" "${LOG_TRT}" > "${DIFF_OUT}" || true
    echo "Diff written to: ${DIFF_OUT}"
    echo ""
    if [[ -s "${DIFF_OUT}" ]]; then
        diff --unified=0 "${LOG_IND}" "${LOG_TRT}" || true
    else
        echo "(logs are identical)"
    fi
else
    banner "STEP 4 — Skipped (one or both run logs missing)"
fi

banner "All done"
echo "Artifacts in: ${OUTDIR}/"
echo "  summary_inductor.json  — inductor kernel listing"
echo "  summary_tensorrt.json  — tensorrt partition report"
echo "  run_inductor.log       — full inductor stdout/stderr"
echo "  run_tensorrt.log       — full tensorrt stdout/stderr"
echo "  compare.log            — structured JSON diff output"
echo "  log_diff.txt           — raw unified diff of the two logs"
