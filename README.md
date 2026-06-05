# Production Video Diffusion

Text-to-video generation using **Wan-AI/Wan2.1-T2V-1.3B-Diffusers** via HuggingFace Diffusers, with profiling, `torch.compile`, CFG parallelism, and sequence parallelism support.

## Install

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create an isolated environment

```bash
uv venv bangpyper-env
source bangpyper-env/bin/activate
```

### 3. Install Rust (required by `outlines-core`)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

### 4. Install SGLang with diffusion extras

```bash
uv pip install 'sglang[diffusion]' --prerelease=allow
```

### 5. Install HuggingFace CLI and download models

```bash
pip install huggingface_hub[cli]
hf auth login                                          # needed for gated models
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers          # ~10 GB
```

### 6. Sync the project environment

Dependencies are declared in `pyproject.toml`:

```bash
uv sync
```

A CUDA-capable GPU is required. ~12 GB VRAM is sufficient at the default 480×832 resolution; use `--cpu-offload` if you have less.

---

## Quick start

Generate a video with the default prompt:

```bash
uv run python wan2_text_to_video.py
```

Custom prompt, 5-second clip at 16 fps:

```bash
uv run python wan2_text_to_video.py "A cat walking through a sunlit garden, cinematic lighting" \
    --num-frames 81 --fps 16 --output output.mp4
```

> `num_frames` must be of the form `4k + 1` (e.g. 33, 49, 65, 81). 81 frames @ 16 fps ≈ 5.06 s.

---

## CLI options

### Core

| Flag | Default | Description |
|---|---|---|
| `prompt` (positional) | cat in garden | Text prompt. |
| `--negative-prompt` | `""` | Things to avoid. |
| `--output` | `output.mp4` | Output file path. |
| `--num-frames` | `33` | Frame count — must be `4k + 1`. |
| `--height` | `480` | Frame height in pixels. |
| `--width` | `832` | Frame width in pixels. |
| `--steps` | `50` | Denoising steps. |
| `--guidance-scale` | `5.0` | CFG strength. Set to `1.0` to disable CFG (~2× faster/step). |
| `--seed` | `None` | Random seed for reproducibility. |
| `--fps` | `16` | Output frame rate. |
| `--dtype` | `float16` | Transformer dtype (`float16` or `bfloat16`). VAE always stays in fp32. |

### Scheduler

| Flag | Default | Description |
|---|---|---|
| `--scheduler` | `default` | `unipc` (fastest), `dpm`, or `default` (model's own). |
| `--flow-shift` | `3.0` | UniPC flow shift. ~3.0 @480p, ~5.0 @720p. |

### Parallelism

| Flag | Default | Description |
|---|---|---|
| `--cfg-parallel` | off | Run cond and uncond CFG passes in parallel across `cuda:0` and `cuda:1`. Requires two GPUs and `guidance_scale > 1.0`. |
| `--seq-parallel` | off | Sequence parallelism (Ulysses / Ring / Unified) across N GPUs. Must be launched with `torchrun`. Incompatible with `--cfg-parallel` and `--cpu-offload`. |
| `--ulysses-degree` | `1` | Number of GPUs for Ulysses head parallelism. `1` = off. |
| `--ring-degree` | `2` | Number of GPUs for Ring sequence parallelism. Set both degrees > 1 for Unified SP (≥4 GPUs). |
| `--cpu-offload` | off | `enable_model_cpu_offload` for <12 GB VRAM (slower). |

### Acceleration

| Flag | Default | Description |
|---|---|---|
| `--teacache` | off | Skip redundant denoising steps when the time-modulation signal barely changes (experimental). |
| `--teacache-threshold` | `0.08` | Accumulation threshold. Higher = more skips, lower quality. Tune in 0.05–0.15 range. |

### Compile

| Flag | Default | Description |
|---|---|---|
| `--compile` | off | JIT torch.compile the DiT via regional block compilation. |
| `--compile-mode` | `default` | `default`, `reduce-overhead`, or `max-autotune`. |
| `--compile-vae` | off | Also compile the VAE decoder. |
| `--warmup` | auto | Warmup runs before the timed generation. Defaults to 1 when `--compile` is set, else 0. |

### Observability

| Flag | Default | Description |
|---|---|---|
| `--monitor-memory` | off | Report peak GPU memory (allocated / reserved) after the run. |
| `--profile` | off | Wrap inference in `torch.profiler`. |
| `--profile-dir` | `./profiler_logs` | Directory for profiler traces. |

---

## Parallelism examples

**CFG parallel** — two GPUs, cond/uncond in parallel:

```bash
python wan2_text_to_video.py "A cat in a garden" \
    --num-frames 81 --cfg-parallel --fps 16 --output output.mp4
```

**Ring attention** — two GPUs, sequence split across them:

```bash
torchrun --nproc-per-node 2 wan2_text_to_video.py "A cat in a garden" \
    --num-frames 81 --seq-parallel --ring-degree 2
```

**Unified sequence parallelism** — four GPUs (Ulysses × Ring):

```bash
torchrun --nproc-per-node 4 wan2_text_to_video.py "A cat in a garden" \
    --num-frames 81 --seq-parallel --ulysses-degree 2 --ring-degree 2
```

---

## torch.compile

JIT-compile the transformer for faster inference. The first step pays a compilation cost; subsequent steps are faster.

```bash
python wan2_text_to_video.py "A cat in a garden" \
    --num-frames 81 --compile --compile-mode max-autotune
```

Modes:
- `default` — safe, modest speedup, no CUDA graphs.
- `reduce-overhead` — CUDA graphs, larger speedup.
- `max-autotune` — most aggressive; longest first-step compile.

`reduce-overhead` and `max-autotune` use CUDA graphs, which alias output buffers across calls. The script handles this automatically via a forward hook that clones outputs after each compiled call.

---

## Profiling (torch.profiler)

```bash
python wan2_text_to_video.py "A cat in a garden" \
    --num-frames 81 --profile --profile-dir ./profiler_logs
```

Writes a Chrome trace and TensorBoard logs:

```bash
# Chrome: open chrome://tracing and load profiler_logs/trace.json
tensorboard --logdir ./profiler_logs
```

---

## Benchmarking

`benchmark.sh` runs the full combination matrix (scheduler × quant/attention × compile strategy) for both the vanilla 50-step pipeline and the fast 8-step distilled pipeline:

```bash
bash benchmark.sh
```

Individual commands for the vanilla + torch.compile matrix are listed in `commands.txt`.

---

## Files

| File | Description |
|---|---|
| `wan2_text_to_video.py` | Main generation script — vanilla 50-step pipeline with parallelism and compile support. |
| `benchmark.sh` | Full combination benchmark (152 runs). |
| `commands.txt` | Expanded vanilla + torch.compile commands (72 runs). |
