#!/usr/bin/env bash
# benchmark_parallel.sh — wan2_text_to_video_parallel.py timing matrix
#
# Fixed: --dtype bfloat16, --monitor-memory, --seed 42, --num-frames 81, --fps 16
#
# Matrix dimensions:
#   parallelism : none (single GPU)
#               | cfg-parallel      (2 GPUs, parallel CFG; needs guidance_scale > 1.0)
#               | seq-ulysses       (2 GPUs, Ulysses attention; torchrun)
#               | seq-ring          (2 GPUs, Ring attention;    torchrun)
#   compile     : eager | jit-default | jit-reduce-overhead | jit-max-autotune
#
# Sections:
#   1. Vanilla   50 steps, guidance_scale=5.0, scheduler=default
#                none × 4 compile = 4 runs
#                cfg-parallel × 2 compile (no CUDA-graph modes) = 2 runs
#                seq-ulysses/seq-ring × 2 compile (no CUDA-graph modes) = 4 runs
#
# Grand total: 10 runs
#
# Usage:
#   ./benchmark_parallel.sh                          # run all, skip completed
#   ./benchmark_parallel.sh 2>&1 | tee run.log
#   rm -f benchmark_parallel.done.log && ./benchmark_parallel.sh   # start fresh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PROMPT="A cat walking through a sunlit garden, cinematic lighting, high quality, detailed"
NUM_FRAMES=81
FPS=16
DTYPE="bfloat16"
SEED=42

DONE_LOG="$SCRIPT_DIR/benchmark_parallel.done.log"

# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------
PY=(uv run python wan2_text_to_video_parallel.py)
TR=(uv run torchrun --nproc-per-node 2 wan2_text_to_video_parallel.py)

# Set LAUNCHER before each call to run(); reset to PY between sections.
LAUNCHER=("${PY[@]}")

# ---------------------------------------------------------------------------
# run <label> <output.mp4> [script flags…]
# Uses the current LAUNCHER array. Writes label to DONE_LOG on success;
# prints a warning and continues on failure (benchmark is not aborted).
# ---------------------------------------------------------------------------
run() {
    local label="$1"; shift
    local out="$1"; shift

    if [[ -f "$DONE_LOG" ]] && grep -qF "=== $label ===" "$DONE_LOG"; then
        echo "=== $label === [SKIPPED]"
        return 0
    fi

    echo ""
    echo "=== $label ==="
    if time "${LAUNCHER[@]}" \
        "$PROMPT" \
        --num-frames "$NUM_FRAMES" \
        --fps        "$FPS" \
        --dtype      "$DTYPE" \
        --seed       "$SEED" \
        --monitor-memory \
        --output     "$out" \
        "$@"; then
        echo "=== $label ===" >> "$DONE_LOG"
    else
        echo "[WARN] $label failed — continuing." >&2
    fi
}

# ---------------------------------------------------------------------------
# compile_flags <tag>  →  extra CLI flags (empty string for eager)
# ---------------------------------------------------------------------------
compile_flags() {
    case "$1" in
        eager)               echo "" ;;
        teacache)            echo "--teacache" ;;
        jit-default)         echo "--compile --compile-mode default" ;;
        jit-reduce-overhead) echo "--compile --compile-mode reduce-overhead" ;;
        jit-max-autotune)    echo "--compile --compile-mode max-autotune" ;;
    esac
}

# ===========================================================================
# Section 1 — Vanilla × Compile  (50 steps, guidance_scale=5.0, default sched)
# ===========================================================================
echo ""
echo "==========================================================================="
echo "Section 1 — Vanilla × Compile (50 steps, cfg=5.0, scheduler=default)"
echo "==========================================================================="

VANILLA_COMPILE_TAGS=(eager jit-default jit-reduce-overhead jit-max-autotune)

# torchrun-based parallelism (seq-ulysses, seq-ring): CUDA graphs (reduce-overhead,
# max-autotune) deadlock during NCCL communicator teardown after generation completes,
# so those modes are excluded here too.
TORCHRUN_COMPILE_TAGS=(eager jit-default)

# none (single GPU, standard Python launch)
LAUNCHER=("${PY[@]}")
for ctag in "${VANILLA_COMPILE_TAGS[@]}"; do
    cflags=$(compile_flags "$ctag")
    # shellcheck disable=SC2086
    run "Vanilla par=none compile=${ctag}" \
        "par_none_vanilla_${ctag}.mp4" \
        --steps 50 --guidance-scale 5.0 $cflags
done

# cfg-parallel (single Python process, 2 GPUs via threading)
# reduce-overhead and max-autotune use CUDA graphs whose C++ TLS is not
# inherited by spawned threads, so those modes are excluded here.
CFG_COMPILE_TAGS=(eager jit-default)
LAUNCHER=("${PY[@]}")
for ctag in "${CFG_COMPILE_TAGS[@]}"; do
    cflags=$(compile_flags "$ctag")
    # shellcheck disable=SC2086
    run "Vanilla par=cfg-parallel compile=${ctag}" \
        "par_cfg_vanilla_${ctag}.mp4" \
        --cfg-parallel --steps 50 --guidance-scale 5.0 $cflags
done

# seq-ulysses (torchrun, Ulysses all-to-all, 2 GPUs)
LAUNCHER=("${TR[@]}")
for ctag in "${TORCHRUN_COMPILE_TAGS[@]}"; do
    cflags=$(compile_flags "$ctag")
    # shellcheck disable=SC2086
    run "Vanilla par=seq-ulysses compile=${ctag}" \
        "par_ulysses_vanilla_${ctag}.mp4" \
        --seq-parallel --ulysses-degree 2 --ring-degree 1 \
        --steps 50 --guidance-scale 5.0 $cflags
done

# seq-ring (torchrun, Ring attention, 2 GPUs)
LAUNCHER=("${TR[@]}")
for ctag in "${TORCHRUN_COMPILE_TAGS[@]}"; do
    cflags=$(compile_flags "$ctag")
    # shellcheck disable=SC2086
    run "Vanilla par=seq-ring compile=${ctag}" \
        "par_ring_vanilla_${ctag}.mp4" \
        --seq-parallel --ulysses-degree 1 --ring-degree 2 \
        --steps 50 --guidance-scale 5.0 $cflags
done

echo ""
echo "==========================================================================="
echo "All benchmark runs complete."
echo "Results in: $DONE_LOG"
echo "==========================================================================="
