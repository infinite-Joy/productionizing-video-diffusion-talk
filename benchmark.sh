#!/usr/bin/env bash
# Full combination benchmark for wan2_text_to_video.py
#
# Fixed settings: --dtype bfloat16, --monitor-memory (always on)
#
# Matrix dimensions:
#   config     : vanilla (no-lora, 50 steps, cfg 5.0)
#              | fast    (lora,    8 steps,  cfg 1.0)
#   scheduler  : vanilla → default / unipc / dpm
#              : fast    → unipc / dpm
#   quant/attn : base / fp8 / sage / fp8+sage
#   compile    : eager / teacache /
#                jit-default / jit-reduce-overhead / jit-max-autotune /
#                aot-default / aot-reduce-overhead / aot-max-autotune
#
# Total runs: vanilla 3×4×8 = 96  |  fast 2×4×7 = 56  |  grand total 152
#
# Note: TeaCache + compile/aot prints a warning and continues (the code
# disables TeaCache internally when AOT is active; with JIT you get graph
# breaks/recompiles). Results for those combos are informational.

set -euo pipefail

PROMPT="A cat walking through a sunlit garden, cinematic lighting"
NUM_FRAMES=81
FPS=16

# run <label> <output.mp4> [extra flags…]
run() {
    local label="$1"; shift
    local out="$1"; shift
    echo ""
    echo "=== $label ==="
    time uv run python wan2_text_to_video.py \
        "$PROMPT" \
        --num-frames "$NUM_FRAMES" \
        --fps "$FPS" \
        --dtype bfloat16 \
        --monitor-memory \
        --output "$out" \
        "$@"
}

# ---------------------------------------------------------------------------
# Shared dimension arrays
# ---------------------------------------------------------------------------

QUANT_TAGS=("base"  "fp8"    "sage"             "fp8sage")
QUANT_FLAGS=(""     "--fp8"  "--sage-attention" "--fp8 --sage-attention")

# Vanilla: 8 strategies including teacache (50 steps = enough skippable steps)
VANILLA_COMP_TAGS=(
    "eager"
    "teacache"
    "jit_default"
    "jit_reduce_overhead"
    "jit_max_autotune"
    "aot_default"
    "aot_reduce_overhead"
    "aot_max_autotune"
)

# Fast: 7 strategies — teacache excluded (8 steps too few to benefit)
FAST_COMP_TAGS=(
    "eager"
    "jit_default"
    "jit_reduce_overhead"
    "jit_max_autotune"
    "aot_default"
    "aot_reduce_overhead"
    "aot_max_autotune"
)

# Resolve compile flags; caller must export $aot_key before calling.
compile_flags() {
    local ctag="$1"
    case "$ctag" in
        eager)               echo "" ;;
        teacache)            echo "--teacache" ;;
        jit_default)         echo "--compile --compile-mode default" ;;
        jit_reduce_overhead) echo "--compile --compile-mode reduce-overhead" ;;
        jit_max_autotune)    echo "--compile --compile-mode max-autotune" ;;
        aot_default)         echo "--aot-compile --compile-mode default         --aot-cache aot_${aot_key}_default.pt" ;;
        aot_reduce_overhead) echo "--aot-compile --compile-mode reduce-overhead --aot-cache aot_${aot_key}_ro.pt" ;;
        aot_max_autotune)    echo "--aot-compile --compile-mode max-autotune    --aot-cache aot_${aot_key}_ma.pt" ;;
    esac
}

# ===========================================================================
# Section 1 — Vanilla  (no LoRA, 50 steps, guidance_scale=5.0)
# Axes: scheduler × quant × compile
# ===========================================================================

VANILLA_SCHED_TAGS=("default"   "unipc"             "dpm")
VANILLA_SCHED_FLAGS=(""         "--scheduler unipc" "--scheduler dpm")

for si in "${!VANILLA_SCHED_TAGS[@]}"; do
    stag="${VANILLA_SCHED_TAGS[$si]}"
    sflags="${VANILLA_SCHED_FLAGS[$si]}"
    for qi in "${!QUANT_TAGS[@]}"; do
        qtag="${QUANT_TAGS[$qi]}"
        qflags="${QUANT_FLAGS[$qi]}"
        for ctag in "${VANILLA_COMP_TAGS[@]}"; do
            aot_key="vanilla_${stag}_${qtag}"
            cflags="$(compile_flags "$ctag")"
            run "Vanilla sched=${stag} quant=${qtag} compile=${ctag}" \
                "cat_vanilla_${stag}_${qtag}_${ctag}.mp4" \
                --no-lora --steps 50 --guidance-scale 5.0 \
                $sflags $qflags $cflags
        done
    done
done

# ===========================================================================
# Section 2 — Fast  (CausVid LoRA, 8 steps, guidance_scale=1.0)
# Axes: scheduler × quant × compile
# Default scheduler is omitted — not recommended with the distillation LoRA.
# ===========================================================================

FAST_SCHED_TAGS=("unipc"                               "dpm")
FAST_SCHED_FLAGS=("--scheduler unipc --flow-shift 3.0" "--scheduler dpm")

for si in "${!FAST_SCHED_TAGS[@]}"; do
    stag="${FAST_SCHED_TAGS[$si]}"
    sflags="${FAST_SCHED_FLAGS[$si]}"
    for qi in "${!QUANT_TAGS[@]}"; do
        qtag="${QUANT_TAGS[$qi]}"
        qflags="${QUANT_FLAGS[$qi]}"
        for ctag in "${FAST_COMP_TAGS[@]}"; do
            aot_key="fast_${stag}_${qtag}"
            cflags="$(compile_flags "$ctag")"
            run "Fast sched=${stag} quant=${qtag} compile=${ctag}" \
                "cat_fast_${stag}_${qtag}_${ctag}.mp4" \
                --lora --steps 8 --guidance-scale 1.0 \
                $sflags $qflags $cflags
        done
    done
done
