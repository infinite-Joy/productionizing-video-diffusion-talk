# Demo Plan — *"74,000 Tokens Per Step: Running Video Diffusion at Production Scale"*

**Audience:** Python developers, mostly comfortable with PyTorch and HuggingFace, mixed familiarity with CUDA / GPU internals.
**Format:** ~45-minute talk with live demo blocks.
**Goal:** Convince the audience that with SGLang Diffusion, *any* Python developer can stand up a production-grade, OpenAI-compatible video diffusion API — and along the way teach the systems concepts (profiling → torch.compile → FlashAttention → custom kernels → serving) that make it 1.2×–5.9× faster than vanilla `diffusers`.

---

## Environment Setup — Lessons Learned

### What we discovered on the demo box


| Issue                                                    | Root cause                                                      | Resolution                                                                                                  |
| -------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `docker pull` fails: "Cannot connect to Docker daemon"   | Docker daemon not running                                       | `sudo dockerd &` attempted                                                                                  |
| `sudo dockerd` fails: iptables permission denied         | Box is a managed container/VM without full root iptables access | **Skip Docker entirely** — use native pip install                                                           |
| `systemctl` not found                                    | No systemd in this environment                                  | Confirmed: managed container, not fixable                                                                   |
| `uv pip install` fails: "No virtual environment found"   | uv defaults to requiring a venv                                 | Use `uv venv bangpyper-env` then activate                                                                   |
| `outlines-core` build fails: "can't find Rust compiler"  | outlines-core 0.1.26 needs Rust to build from source            | Install Rust via `rustup` first                                                                             |
| `huggingface-cli` deprecated                             | Newer `huggingface_hub` ships `hf` CLI instead                  | Use `hf download`, `hf auth login` etc.                                                                     |
| `from sglang.multimodal_gen import VideoGenerator` fails | Import path changed in current SGLang version                   | **Use diffusers for baseline profiling** (better talk narrative anyway); find correct SGLang API separately |


### Confirmed working setup recipe

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh


# 1. Create isolated environment
uv venv bangpyper-env
source bangpyper-env/bin/activate

# 2. Install Rust (needed by outlines-core dependency)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# 3. Install SGLang with diffusion extras
uv pip install 'sglang[diffusion]' --prerelease=allow

# 4. Install HF CLI and download models
pip install huggingface_hub[cli]

hf auth login  # needed for gated models like FLUX.1-dev

hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers       # ~10 GB
hf download black-forest-labs/FLUX.1-dev             # ~24 GB, gated
hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers        # ~28 GB, optional
```

### Pre-talk checklist (night before)

- Verify `which sglang` returns a path
- Verify `python -c "import sglang; print(sglang.__version__)"` works
- Verify `hf scan-cache` shows all three models downloaded
- Verify `df -h ~` shows 60+ GB free
- Run `sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers --prompt "test" --num-inference-steps 2 --save-output` once to confirm end-to-end
- Run the profiling script once and verify traces land in `./profiles/`
- Pre-load `profiles/wan21_trace.json` in Perfetto and bookmark the tab
- Have a pre-recorded MP4 fallback for every video generation
- Two tmux panes ready: one for `sglang serve`, one for `curl`

---

## Talk Structure — Timing Breakdown (45 min)


| Block | Topic                                             | Duration | Live demo?                             |
| ----- | ------------------------------------------------- | -------- | -------------------------------------- |
| 0     | Hook: generate a 5-second video from one CLI line | 3 min    | ✅                                      |
| 1     | Diffusion vs Autoregressive LLM inference         | 5 min    | —                                      |
| 2     | Torch profiling on Wan2.1 (diffusers baseline)    | 6 min    | ✅ (pre-baked trace)                    |
| 3     | `torch.compile` for diffusion                     | 6 min    | ✅ (1-line speedup)                     |
| 4     | FlashAttention for DiT / video                    | 5 min    | —                                      |
| 5     | Custom Triton / CUDA kernels on the hot path      | 6 min    | ✅ (mini Triton kernel)                 |
| 6     | The full inference stack: GPU + server-level arch | 7 min    | —                                      |
| 7     | SGLang as the serving layer (the punchline)       | 8 min    | ✅ (OpenAI-compatible curl + multi-GPU) |
| 8     | Wrap-up, roadmap, Q&A handoff                     | 4 min    | —                                      |


---
## Title slide

The title of this talk is 74000 tokens every tokens. Running diffusions models in production scale.

An alternate title could have been what is the current gap in the market in terms of what is needed.

and that is AI inference.

My friends here at frontiersmind have trained the best model for the indian space. which is great. we absolutely need it.

but the money gets made when there are real users in a real application and when the models are running in production. only then are the efforts spent during training are also realised. only then are we able to earn money.

if you are a seasoned dev or a junior dev in software or a researcher i urge you to start with inference.

and within inference LLM inference has its own challenges and diffusion has its own challenges. still there are many libraries out there which are trying to solve this. the most famous being vLLM. within the diffusion space there is not much options. sglang is there but the diffusion part just opened towards the start of this year. would be happy to be corrected here, but very few companies are working on the diffusion inference front which is also telling about the complexity involved.


---

## Block 0 — The Hook (3 min)

Open with the terminal already showing the **Eplain diffusers baseline running live** (or play the pre-recorded output if time is tight):

```bash
uv run python wan2_text_to_video.py \
  "A cat walking through a sunlit garden, cinematic lighting, high quality, detailed" \
  --num-frames 81 --fps 16 --output output.mp4
```

Let the progress bar tick — 50 denoising steps, ~1:11 total — while you narrate:

Question to audience: I am assuming that most of you have run some form of LLMs. how many have not run an LLM ever maybe using huggingface or something. You know the general autoregressive pipeline right? this is the autoregressive pipeline.

<img src="Firefly_Flux_The%20simplest%20autoregressive%20pipeline%20in%20LLMs%20240169.jpg" width="500"/>

Now a show of hands how many of you have never run a diffusion pipeline such as this with any diffusion model. I am running a video pipeline here but any pipeline is fine, image, audio, or others.

> *"Five seconds of video. 81 frames at 832×480. Stock HuggingFace Diffusers, H100, nothing fancy. Over a minute. That's the baseline we're starting from today."*

Show the output video. Then open the pre-baked Perfetto trace (`profiles/wan21_trace.json`) and **annotate the three zones live**:

---

## Block 2 — Torch Profiling (6 min)

### Narrative pivot — profile diffusers first, then show SGLang's answer

This is the key insight from our setup work: **profile the vanilla diffusers baseline**. This is actually a better talk narrative — "here's what's slow, and here's what SGLang does about it."

### Profiling script (confirmed working approach)

```python
# profile_wan21.py — saves traces into ./profiles/

import os
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from diffusers import AutoPipelineForVideo

TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
os.makedirs(TRACE_DIR, exist_ok=True)

pipe = AutoPipelineForVideo.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Warmup — critical! First run triggers torch.compile / Triton autotune
print("Warmup run...")
_ = pipe("warmup", num_inference_steps=2, num_frames=9).frames

# Profiled run
print("Profiling...")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    profile_memory=True,
) as prof:
    with record_function("e2e_video_generation"):
        _ = pipe(
            "A curious raccoon peers through sunflowers",
            num_inference_steps=10,
            num_frames=33,
        ).frames

# 1. Print top CUDA ops
table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
print(table)

# 2. Save table as text
table_path = os.path.join(TRACE_DIR, "wan21_top_ops.txt")
with open(table_path, "w") as f:
    f.write(table)
print(f"Saved op table  → {table_path}")

# 3. Chrome/Perfetto trace
trace_path = os.path.join(TRACE_DIR, "wan21_trace.json")
prof.export_chrome_trace(trace_path)
print(f"Saved trace     → {trace_path}")

# 4. CUDA stacks for flame graph
stacks_path = os.path.join(TRACE_DIR, "wan21_cuda_stacks.txt")
prof.export_stacks(stacks_path, "self_cuda_time_total")
print(f"Saved stacks    → {stacks_path}")

print(f"\nAll traces saved in {TRACE_DIR}/")
print("Open wan21_trace.json in https://ui.perfetto.dev")
```

### What to show on stage (pre-baked trace)

1. Open `profiles/wan21_trace.json` in Perfetto (pre-loaded the night before).
2. Zoom into one denoising step. Point out:
  - Repeating "WanAttentionBlock" pattern × N layers.
  - Fat `flash_attn_func` / `_scaled_dot_product_efficient_attention` bar inside each block.
  - Small `linear`/`silu`/`add_norm` bars between attention.
  - Single big `vae_decode` block at the very end.
3. Show `profiles/wan21_top_ops.txt` — attention kernels dominate by 5–10×.

```
|← ~1s text encoder →|←————————— 50 × DiT denoising steps (~69s) ————————————→|← ~1s VAE decode →|
       (tiny)                       same model, 50 identical forward passes                (tiny)
```

Point out:
- The text encoder at the far left — runs once, small.
- The repeating zigzag band in the middle — that is the **exact same DiT transformer called 50 times**, one per denoising step. Each zigzag is one step; each tooth inside it is one transformer block.
- The single block at the far right — the VAE decoder, also runs once.

> *"Over 95% of the wall-clock time is here, in the middle. The same model. 50 times. That is the target. Everything we do for the rest of this talk is about making those 50 calls faster — or fewer."*

This pipeline shape — **Encoder → N × DiT → Decoder** — is universal: FLUX, HunyuanVideo, CogVideoX, Wan2.1. Fix the DiT loop once and the win applies everywhere.

Then flip to a single slide listing the optimization layers:

1. **Profile** — find the hot path
2. **`torch.compile`** — fuse the kernel graph
3. **FlashAttention** — tame the O(L²) attention bottleneck
4. **Triton / CUDA kernels** — fuse the residual ops
5. **SGLang** — production serving with all of the above + multi-GPU

> *"By the end of this talk, we're going to peel back each layer and understand why it works — and then hand the whole stack to a single Python package. Let's profile first."*

**Baseline numbers to anchor the room:**
- 50 steps × 1.42 s/step = ~71 s wall-clock for a 5-second clip
- Hardware: H100 80 GB (or equivalent)
- Framework: `diffusers` 0.33+, `torch` 2.5+

### Three profiling takeaways for the audience

- Always insert `record_function("...")` around the denoising loop.
- Sort by `self_cuda_time_total`, not `cpu_time_total`. Diffusion is GPU-bound.
- The warmup run matters — skip it and your trace is full of JIT noise.

---

### The diffusion gap (India and World)

Now the obvious question from here is based on this what are the ways to make a diffusion pipeline faster. I would like from someone who has primarily worked with LLM pipelines or has more knowledge of LLMs and has never worked at a diffusion pipeline.

Intelligence is supported by all these massive list of companies out there. But dont be intimidated by this slide. its not enough. There are huge gaps.

<img src="Firefly_Gemini%20Flash_the%207%20layers%20of%20an%20AI%20factory,%20you%20need%20all%20of%20these%20to%20work%20in%20tandem%20to%20bring%20you%20i%20240169.png" width="500"/>

We need exponential more innovation to democratise intelligence and to make intelligence truly useful. For this we need lots of companies.

---

## Block 1 — Diffusion vs Autoregressive LLM Inference (5 min)

Frame the key insight: most Python devs think "LLM inference = predict next token from a KV-cache." Video diffusion is fundamentally different, and that difference dictates every optimization that follows.

### Side-by-side comparison (one slide)

![AR and diffusion](Firefly_Gemini%20Flash_Split-screen%20infographic%20comparing%20two%20AI%20generation%20methods,%20left%20side%20warm%20amber%20ba%20240169.png)


| Aspect                | Autoregressive LLM (Llama, Qwen)  | Diffusion DiT (Wan2.1, FLUX, HunyuanVideo)                              |
| --------------------- | --------------------------------- | ----------------------------------------------------------------------- |
| Generation primitive  | One token at a time               | One denoising step over the **whole** latent                            |
| Iterations per output | `output_len` tokens (100s–1000s)  | 25–50 denoising steps                                                   |
| Per-step compute      | Tiny (1 token × hidden) at decode | **Huge** — full sequence forward each step                              |
| Sequence length       | Grows monotonically with KV cache | **Fixed and very long from step 0** (Wan2.1 720p 5s ≈ 74K tokens)       |
| Memory bottleneck     | KV cache (O(L) growth)            | Activations + attention scores (O(L²) inside attention)                 |
| Cache                 | RadixAttention / PagedAttention   | **No KV cache**; timestep embedding + optional feature-cache (TeaCache) |
| Compute pattern       | Memory-bound at decode            | **Compute-bound** at every step                                         |
| Bottleneck op         | attention + MoE all-to-all        | self-attention (>76% of DiT FLOPs in 1.3B; ~91% in 14B)                 |


### What this means for serving

So now based on the profile what are the ways in which we can improve the latency of creation of these video models.

- Speed-ups come from: (a) fewer denoising steps, (b) faster *each* step (compile, FlashAttn, fused kernels, quantization), (c) sharding the long sequence across GPUs (USP, CFG-parallel, TP).

Now I want to spend couple of minutes on the gap between AR generation and diffusion

The fundamental split is compute-bound vs memory-bound. LLM decode is memory-bandwidth-bound — you're streaming weights and the KV cache regardless, so adding more sequences to a batch is nearly free throughput. That's why continuous batching is such a windfall. Diffusion backbones (UNet/DiT) are compute-bound: each denoising step is a big dense FLOP load over a large spatial latent, and a single high-res sample can already saturate the GPU's matmul units. Batching doesn't give you the free lunch it gives LLMs — you're trading latency for throughput much more directly, and GPU-seconds-per-image stays high no matter how clever you are.
On top of that structural fact, several things compound:

Activation memory scales with resolution, and explodes for video. Spatial attention is quadratic in the number of latent tokens. A high-res image is already a lot; a multi-second clip with spatiotemporal attention across frames is brutal — you frequently can't fit a clip in memory and have to tile / chunk / sliding-window, which adds complexity and seam artifacts. Text is 1D and modest by comparison.
No dominant serving stack. There's no vLLM-for-diffusion. People run diffusers (a reference/research library, not a throughput engine), ComfyUI for workflows, some TensorRT. Nothing consolidates paged memory + continuous batching + production scheduling the way the LLM stack does. The engineering investment is an order of magnitude smaller, and it's fragmented.
The pipeline is heterogeneous. An LLM is basically one homogeneous transformer. A diffusion request is text encoder(s) → diffusion backbone (×N steps) → VAE decode, often plus a refiner, upscaler, safety checker. Batching and pipelining efficiently across those uneven stages is genuinely messy.
Batching is hard even when you want it. Continuous batching works for LLMs because requests join/leave at token granularity and share prefixes. Diffusion requests march in lockstep through N steps, and they differ in resolution, aspect ratio, step count, and conditioning — variable tensor shapes break uniform packing. You can't interleave a half-finished request the way you can with token streams.
Quantization is less plug-and-play. INT4/FP8 for LLMs is mature and mostly lossless-enough. Diffusion is more sensitive — quality degrades and errors compound across the iterative trajectory. Approaches like SVDQuant exist but it's not the press-a-button win it is for text.
Cross-step caching is only an approximation. DeepCache and friends exploit that high-level features change slowly between adjacent steps, but that's a quality/speed tradeoff, not the exact, free reuse a KV cache gives autoregressive decode.

Video is where all of this turns from "gap" into "frontier." Temporal consistency demands attention across frames, memory and compute scale with clip length, and long-form generation is still an open systems problem, not a solved serving problem. That's the sharpest end of the gap right now.
So the honest framing: the math of fast sampling is well-studied, but the systems economics — compute-bound backbones, no batching free-lunch, immature serving infra, heterogeneous pipelines, and video's memory wall — are where diffusion inference is genuinely years behind where LLM serving is. If you want, I can pull current numbers on where realized throughput/latency lands for a specific model (say, a current DiT image model or a video model) to make the gap concrete.



## Block 3 — `torch.compile` Benefits (6 min)

### The 1-line story

```python
pipe.transformer.compile(fullgraph=True)
```

On an H100, FLUX-1-Dev: 6.7s → 4.5s, ~1.5× speedup, no quality change. Compile *only* the DiT — text encoders and VAE are <5% of runtime.

### What torch.compile actually does for diffusion

1. **Graph capture (Dynamo).** Traces the Python forward pass into an FX graph.
2. **Operator fusion (Inductor + Triton).** Fuses `silu → mul → add_norm` into one kernel. Kernel-launch overhead disappears.
3. **CUDA graph capture.** `mode="reduce-overhead"` removes Python overhead for the entire denoising step.
4. **Specialization.** Compiles for an exact (B, H, W) shape.

### Possible issues to look out for

The compiler is complicated. One of the things we’ve slowly been coming to terms with is that, uh, maybe promising you could just slap torch.compile on a model and have it run faster was overselling the feature a teensy bit? There seems to be some irreducible complexity with compilers that any user bringing their own model to torch.compile has to grapple with. So yes, you are going to spend some of your complexity budget on torch.compile, in hopes that the payoff is worth it (we think it is!) One ameliorating factor is that the design of torch.compile (graph breaks) means it is very easy to incrementally introduce torch.compile into a codebase, without having to do a ton of upfront investment.
Compile time can be long. The compiler is not a straightforward unconditional win. Even if the compiler doesn’t slow down your code (which it can, in pathological cases), you have to spend some amount of time compiling your model (investment), which you then have to make back by training the model more quickly (return). For very small experimentation jobs, or jobs that are simply crashing, the time spent compiling is just dead weight, increasing the overall time your job takes to run. (teaser: async compilation aims to solve this.) To make matters worse, if you are scheduling your job on systems that have preemption, you might end up repeatedly compiling over and over again every time your job gets rescheduled (teaser: caching aims to solve this.) But even when you do spend some time training, it is not obvious without an A/B test whether or not you are actually getting a good ROI. In an ideal world, everyone using torch.compile would actually verify this ROI calculation, but it doesn’t happen automatically (teaser: automatic ROI calculation) and in large organizations we see people running training runs without even realizing torch.compile is enabled.
Numerics divergence from eager. Unfortunately, the compiler does not guarantee exact bitwise equivalence with eager code; we reserve the right to do things like select different matrix multiply algorithms with different numerics or eliminate unnecessary downcast/upcasts when fusing half precision compute together. The compiler is also complicated and can have bugs that can cause loss not to converge. Expect to also have to evaluate whether or not application of torch.compile affects accuracy. Fortunately, for most uses of compiler for training efficiency, the baseline is the eager model, so you can just run an ablation to figure out who is actually causing the accuracy problem. (This won’t be true in a later use case when the compiler is load bearing, see below!)

source: https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/

In production we take the middle path for engineering complexity and effort vs production latency. we perform ahead of time compilation and actually save the model and run inference using the model and not using the pytorch code. 

source: https://dev.to/minwook/pytorch-compile-vs-export-omc

### Speedups reported (slide)

| Setting                                                 | Speedup vs eager      | Source                      |
| ------------------------------------------------------- | --------------------- | --------------------------- |
| FLUX-1-Dev, H100, `transformer.compile(fullgraph=True)` | **1.5×** (6.7→4.5s)   | PyTorch blog                |
| FLUX-1-Dev, H100, compile + FP8 (torchao)               | **+53.88%** over bf16 | sayakpaul/diffusers-torchao |
| CogVideoX-5B, A100, compile + INT8                      | +27.33%               | same                        |
| FLUX-1-dev, 4×H100, USP + compile                       | **2.63×** vs 1×H100   | xDiT                        |
| FLUX-2, B200, compile + CUDA Graphs + NVFP4 + TeaCache  | up to **10.2×**       | NVIDIA blog                 |


### Caveats — dynamic shapes (the gotcha section)

Resolution/aspect ratio changes trigger recompiles. Three workarounds:

```python
# A: relax shapes
pipe.transformer.compile(fullgraph=True, dynamic=True)

# B: explicitly mark dimensions dynamic
torch._dynamo.mark_dynamic(latents, 2)  # H
torch._dynamo.mark_dynamic(latents, 3)  # W

# C: regional compilation — compile individual blocks, reuse across all N
for block in pipe.transformer.transformer_blocks:
    block.compile(fullgraph=True)
```

### Live demo

- Run `pipe(...)` once → time it (e.g., 6.7s).
- Add `pipe.transformer.compile(fullgraph=True)`, run twice (first pays compile cost, second is steady state) → time it (e.g., 4.5s).
- Show how SGLang Diffusion bakes this in: `--enable-torch-compile` flag.

---

## Block 4 — Flash Attention for Diffusion Transformers (5 min)

### Why FA matters more for video DiTs than for LLMs

- An LLM decode step: 1 query × few-K KV. Easy.
- A Wan2.1 720p 5s denoising step: **74,000-token sequence × 50 steps = 3.7M attention positions per generation**.
- Vanilla attention's O(L²) memory: score matrix alone ~22 GB at fp16. Won't even fit.
- FlashAttention-2: tiles QKV through SRAM, never materializes L×L. Memory becomes O(L).
- FlashAttention-3: Hopper-specific async TMA + warp specialization + FP8 — **740 TFLOPs/s FP16 (75% H100 peak), ~1.2 PFLOPs/s FP8**.

### Concrete numbers for the slide

- HunyuanVideo e2e: 945s → 685s with Sliding-Tile-Attention (training-free).
- Wan2.1 1.3B: 31s → 18s with Video Sparse Attention.
- FA3 vs FA2 in DiTs: 1.5–2× forward pass speedup.

### How SGLang Diffusion uses it

- `sgl-kernel` ships pre-compiled attention kernels (same ones from the LLM stack).
- Backend selection: `--attention-backend flash_attn_3` CLI flag.
- Roadmap: FA4 integration for Blackwell in Q1 2026.

### Slide content (no live demo — too low-level)

> "In Wan2.1 14B, attention is 91% of the DiT runtime. Halve attention with FA3 → halve total runtime. Period."

---

## The hardware angle

Now here's the full updated overview with both additions integrated:

---

## The Big Picture

The AI hardware market is at a major inflection point. The defining shifts are the transition from training-dominated to inference-dominated workloads, the rise of custom ASICs over general-purpose GPUs, and the emergence of HBM4 memory as a critical enabler for next-generation models. Combined hyperscaler capital expenditure has reached $660–690 billion in 2026, with about 75% directed at AI-specific infrastructure.

---

## NVIDIA: Still Dominant, But the Moat is Narrowing

NVIDIA remains the clear leader with an estimated 80–90% share of the AI accelerator market. Its Blackwell platform, including the B200 and Ultra variants, continues ramping rapidly across hyperscaler deployments at Meta, Microsoft, and others. The next-generation **Vera Rubin** architecture, slated for late 2026, is a major leap: it offers 3.3x the FP4 compute of Blackwell Ultra, pairs HBM4 memory with NVLink 6, and delivers 1.2 ExaFLOPS of FP8 training per rack.

That said, NVIDIA's dominance is no longer uncontested. Analysts project that NVIDIA's share of the inference market specifically could fall from over 90% to between 20–30% by 2028 as custom silicon matures.

---

## AMD: The Most Credible GPU Challenger

AMD is mounting a serious challenge. The Instinct MI400 series, based on the new CDNA 5 architecture, features 432GB of HBM4 memory and 19.6TB/s bandwidth, with the flagship MI450 delivering up to 40 PFLOPS of FP4 performance. The MI455X packs 320 billion transistors across 12 TSMC N2 compute chiplets.

AMD posted full-year 2025 data center revenue of $16.64 billion (up 32%) and expects continued strong growth fueled by the MI400 ramp. AMD also secured a reported $60 billion deal with Meta, signaling growing hyperscaler willingness to diversify away from NVIDIA.

---

## Cerebras: The Wafer-Scale Wildcard

Cerebras has gone from scrappy startup to Wall Street sensation. The company raised $5.5 billion in its IPO on May 14, pricing shares at $185 — well above its raised range — then opened to public trading at $385, more than doubling on day one. The deal valued the company at about $56.4 billion fully diluted, making it the largest US tech IPO in years.

What sets Cerebras apart is its architecture: its Wafer Scale Engine is a processor the size of a dinner plate, packing more than 4 trillion transistors onto a single piece of silicon. Instead of slicing a wafer into many small chips, Cerebras uses the entire wafer as one massive processor, giving it advantages in memory bandwidth and on-chip communication. The company specializes in AI inference — running trained models — rather than training, positioning it in the fastest-growing compute segment.

Cerebras reported $510 million in 2025 revenue (up 76% year-over-year) and swung to $237.8 million in net income from a loss of nearly half a billion the prior year. An AWS partnership and OpenAI's $20 billion-plus compute deal largely resolved the customer concentration issue that had stalled the company's original 2024 IPO attempt. The order book closed roughly 20 times oversubscribed — a clear signal of investor appetite for pure-play AI infrastructure bets beyond NVIDIA.

---

## The Rise of Custom Hyperscaler Silicon

Perhaps the most significant structural shift is the explosion of in-house chips from the cloud giants:

**Google TPU v7 (Ironwood)** — Google projects 4.3 million TPU shipments in 2026, rising to 10 million in 2027. Anthropic disclosed deploying more than one million Ironwood chips for Claude inference workloads, making it the first custom ASIC to reach seven-figure deployment at a single customer. Google is already planning 8th-gen TPUs split into dedicated training and inference variants, designed by multiple partners including Broadcom and MediaTek.

**Amazon Trainium** — Amazon values its custom chip business at $50 billion and has hinted at selling Trainium externally. AWS has collaborated with NVIDIA on NVLink Fusion integration for future Trainium generations.

**Microsoft Maia 200** — Announced in January 2026, Maia 200 claims three times the FP4 performance of Amazon's Trainium 3, fabricated on TSMC 3nm.

**Meta MTIA v3** — Meta continues developing its own accelerator, primarily targeting inference for its recommendation and generative AI workloads.

The custom ASIC market for AI is growing at 44.6% annually, compared with 16.1% for GPUs.

---

## Apple M5: On-Device AI Goes Mainstream

Apple is playing a different game from the data center players, but its impact on the AI hardware landscape is enormous because of sheer scale — hundreds of millions of devices.

The M5, announced in late 2025 and shipping across MacBook Pro, iPad Pro, and Apple Vision Pro, features a next-generation GPU architecture where every compute block is optimized for AI. The standout addition is a **Neural Accelerator built into every GPU core** — a first for Apple silicon — meaning AI workloads run on the GPU itself rather than being offloaded solely to a separate Neural Engine.

The numbers are significant: Apple claims up to 4x faster AI performance versus M4 Pro and M4 Max, including 4x faster large language model prompt processing and up to 3.8x faster AI image generation. Unified memory bandwidth reaches 153GB/s, a nearly 30% increase over M4 and more than 2x over M1, enabling larger AI models to run entirely on device. The M5 Pro and M5 Max use what Apple calls "Fusion Architecture," combining two 3nm dies into a single system on a chip.

Why this matters for the broader landscape: there's been a major shift toward on-device AI, as developers increasingly prefer running models locally for faster response times, better privacy, lower cloud costs, and offline capability. Apple's approach runs AI tasks locally by default, reducing latency and protecting user data, while more complex requests scale securely using Apple's cloud infrastructure and external models like Google's Gemini. With the M5 now in MacBook Airs starting at $1,099, Apple is putting serious AI inference capability into consumer-grade machines at scale.

---

## Intel: A Surprising Comeback Story

Intel's stock surged roughly 240% year-to-date in 2026, driven largely by its foundry business. Reports emerged that Apple is finalizing a chip foundry deal that could deliver 25% of Apple's chip orders to Intel by 2030, potentially generating $10 billion in annual foundry revenue. Intel's 18A process node debuted at CES 2026, and the company is re-entering the data center AI market with its Gaudi accelerators.

---

## Key Themes to Watch

**Training → inference**: Inference economics are reshaping silicon design — training gets the headlines, but deploying models at scale drives recurring revenue. About two-thirds of all AI compute is now inference, which is why Cerebras, Google Ironwood, and Apple's M5 are all optimized for this workload.

**Manufacturing bottlenecks**: TSMC's 3nm node runs at 100% capacity utilization with demand roughly three times exceeding supply. TSMC's 2nm (N2) node is entering production and will underpin the next wave of chips.

**Software ecosystem unlocking**: OpenAI's Triton has emerged as the industry's primary "off-ramp," allowing developers to write hardware-agnostic high-performance kernels in Python, reducing NVIDIA's CUDA lock-in.

**Edge and consumer AI**: Apple's M5 and competitors from Qualcomm are making local LLM inference a mainstream reality, shifting the cost and privacy calculus away from cloud-only approaches.

**Geopolitics**: The era of the "borderless" tech company has been replaced by technological protectionism and the race for sovereign AI, with governments transitioning from regulators to primary customers and protectors of the sector.

**IPO wave**: Should OpenAI and SpaceX follow Cerebras to market, they're expected to be even larger, and the broader semiconductor space has been on fire, with the VanEck Semiconductor ETF up 58% in 2026.

---

In short, the landscape is diversifying fast. NVIDIA still holds the high ground for training, but 2026 is the year that custom silicon, wafer-scale architecture, and on-device AI all became genuine competitive forces — especially for the inference workloads that now dominate AI compute.

---

## Block 5 — Custom Triton / CUDA Kernels on the Hot Path (6 min)

### What's left after compile + FlashAttention?

Look back at the profile — the residual hot path:

1. **RoPE** (rotary embedding) — applied to Q and K every block, every step.
2. **RMSNorm** — twice per block.
3. **Quantization** (FP8/INT8/NVFP4) cast + scale.
4. **AdaLN modulation** — the timestep-conditioning pattern: `x = (1 + scale) * norm(x) + shift`.

All memory-bound, pointwise/reduce ops — perfect for fusion.

### Reference speedups (Liger Kernel benchmarks)


| Op                        | Speedup vs unfused | Memory reduction |
| ------------------------- | ------------------ | ---------------- |
| Fused RoPE (Triton)       | **8×**             | 3×               |
| Fused RMSNorm             | **7×**             | 3×               |
| Fused SwiGLU / GeGLU      | ~2×                | ~2×              |
| Fused Linear+CrossEntropy | ~2×                | up to 5×         |


### Mini Triton kernel demo — fused RMSNorm

```python
import triton
import triton.language as tl
import torch

@triton.jit
def rmsnorm_fwd(
    x_ptr, w_ptr, y_ptr,
    stride, n_cols,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask)
    y = (x * rstd) * w
    tl.store(y_ptr + row * stride + cols, y, mask=mask)

def rmsnorm(x, w, eps=1e-6):
    y = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK = triton.next_power_of_2(n_cols)
    rmsnorm_fwd[(n_rows,)](
        x, w, y, x.stride(0), n_cols,
        eps=eps, BLOCK=BLOCK, num_warps=4,
    )
    return y
```

Bench it live:

```python
x = torch.randn(74000, 1536, device="cuda", dtype=torch.bfloat16)
w = torch.ones(1536, device="cuda", dtype=torch.bfloat16)
%timeit torch.nn.functional.rms_norm(x, [1536], w)   # ~0.5 ms
%timeit rmsnorm(x, w)                                # ~0.07 ms
```

colab notebook related to this:
https://colab.research.google.com/drive/1LsrhwqFaCu-iVUm4BxWHOYqOkUey7BOM#scrollTo=ABMI6grU5iMU


There are some points here. Why are we doing this?

1. Because we can now. writing cuda code is hard, but there has been an explosion of DSLs which are trying to make the computations fast and will be productive from a coding POV. Before 2026 this was not possible.
2. cudnn although quite good will still rely on some heuristics. Its always worth it to go through your production usage, and then create the kernel that is exactly inline with that. That will be the best.
3. even if you fail, you would have proven that better than this is not possible and the exact reasons for that.
4. in the proces you will learn a lot of things about the hardware that you are using. not caring about the hardware where you are running is pre 2025. Now you want to run the software that best gels with the hardware that you have. its like you may not be usain bolt, but there are latent capabilties that you have that will make you the most efficient and you need to discover the conditions to do that.

### How SGLang / FastVideo use custom kernels

- `sgl-kernel` provides fused RoPE, RMSNorm, quantization kernels.
- FastVideo contributes Video Sparse Attention (VSA), Sliding Tile Attention (STA).
- Jan 2026 update: `--dit-layerwise-offload true` — peak VRAM ↓ 30GB, perf ↑ up to 58%.

### Punchline

> "You don't have to write any of these. SGLang ships them. But understanding what is being fused tells you where to look when your serving latency regresses."

---

## Block 6 — The Full Inference Stack: GPU-level and Server-level Architecture (7 min)

### Narrative pivot — zoom out from the kernel to the system

> *"We've spent the last 20 minutes inside a single GPU — fusing kernels, tiling attention matrices. Before we hand this all to SGLang, let's zoom out one level and ask: how does a production AI inference system actually look, end to end?"*

### The three-layer inference stack (one slide)

Every production AI inference system has the same three-layer shape, whether you're serving an LLM or a video diffusion model. Source: [AWS Prescriptive Guidance — Inference Stack Components](https://docs.aws.amazon.com/prescriptive-guidance/latest/gen-ai-inference-architecture-and-best-practices-on-aws/inference-stack-components.html)

A good starting point with a lot of the things that i have talked about today is there in SGlang, but its good to understand the library in depth. I was trying to find a library that best encapsulates the concerns that i have talked about in this talk and i found SGLang to be the closest. Maybe you can start from here, but definitely go through all the points and then find ways on improving on them.

```
┌──────────────────────────────────────────────────────────┐
│  Layer 1 — End User Application                          │
│  Auth · Rate limiting · Queueing · Multi-region LB       │
│  Tools: LiteLLM, Portkey, Kong AI Gateway                │
├──────────────────────────────────────────────────────────┤
│  Layer 2 — Inference API Server (Frontend)               │
│  Request batching · Routing · Queue mgmt · Metrics       │
│  Tools: NVIDIA Triton, NVIDIA Dynamo, Ray Serve, TGI     │
├──────────────────────────────────────────────────────────┤
│  Layer 3 — Inference Backend                             │
│  Memory mgmt · Model loading · Compute batching          │
│  Model formats: Safetensors, GGUF, ONNX, NeMo, HF       │
│  Tools: vLLM, TensorRT-LLM, ONNX Runtime, SGLang        │
└──────────────────────────────────────────────────────────┘
```

These layers communicate via gRPC internally and expose an OpenAI-compatible REST API externally. Key point for this talk: **SGLang spans Layers 2 and 3** — it is both the inference backend *and* a production-grade API server frontend.

### GPU-level parallelism: scaling one model across many GPUs

Video diffusion's core problem: a 74K-token sequence is present at step 0. No KV cache. No autoregressive decomposition. The sequence is huge, all at once, every step. Three strategies handle this:

**Tensor Parallelism (TP)** — split weight matrices column/row-wise across N GPUs. Each GPU computes 1/N of every matmul, then all-reduces to merge. Universal — applies to any layer.

```
Q × W[d × 4d] split across 4 GPUs:
  GPU0: W[:, 0:d]   GPU1: W[:, d:2d]   GPU2: W[:, 2d:3d]   GPU3: W[:, 3d:4d]
                           ↓ all-reduce
```

SGLang flag: `--num-gpus N`

**Sequence Parallelism — USP (Ulysses-SP × Ring-Attention)** — split the *sequence dimension* across GPUs. Each GPU holds 74K/N tokens. For attention:
- *Ulysses*: all-to-all redistributes Q, K, V so each GPU computes full attention for its own heads
- *Ring-Attention*: each GPU passes its K, V chunk around a ring, accumulating partial softmax scores

This is the critical win for long-sequence diffusion — the communication cost does not grow beyond what attention already requires.

```
74K tokens → GPU0: 0–18K | GPU1: 18–37K | GPU2: 37–55K | GPU3: 55–74K
                              ↕ ring K,V pass ↕
                          all-reduce after softmax
```

**CFG-Parallel** — classifier-free guidance requires two independent forward passes per denoising step: one conditioned on the text prompt, one unconditional. Run each on a separate GPU. 2× throughput, zero communication overhead between the pair.

```
Step t:  GPU0 runs forward(x_t, text_embed)
         GPU1 runs forward(x_t, null_embed)
         → merge: x_{t-1} = uncond + scale × (cond − uncond)
```

SGLang flag: `--enable-cfg-parallel`

These stack: `--num-gpus 4 --enable-cfg-parallel` gives TP=2 on two independent CFG halves.

### Server-level parallelism: scaling across many requests

Once the model is fast per-request, serving at scale requires the frontend layer to handle:

| Concern | The problem | The solution |
| --- | --- | --- |
| **Continuous batching** | Video generation takes 15–60s; can't block on a static batch | Dynamic request queue — fill GPU gaps with new requests |
| **Request routing** | Multiple replicas or GPU pools | Route by queue depth, not round-robin |
| **Throttling** | One user floods the GPU | Per-API-key rate limits at the gateway |
| **Failover** | A GPU node crashes mid-generation | Multi-region replica with health checks |
| **Observability** | Did step latency regress after a deploy? | Per-step Prometheus metrics, p99 alerting |

**Diffusion-specific note:** unlike LLM decode (fast, bursty, milliseconds per token), video generation holds a GPU for 15–60s per request. This changes the economics:
- Batch size matters less — each request saturates the GPU compute budget alone
- Queue depth and request admission control matter more
- Mid-generation pre-emption is expensive — you cannot cheaply pause a denoising run at step 23 of 50

### Where SGLang sits in this picture (slide to photograph)

```
┌──────────────────────────────────────────────────┐
│  Your app / OpenAI SDK client                    │  ← Layer 1: you bring this
├──────────────────────────────────────────────────┤
│  sglang serve --port 3000                        │  ← Layer 2: built-in
│  OpenAI REST · gRPC · Prometheus metrics         │
├──────────────────────────────────────────────────┤
│  SGLang Inference Engine                         │  ← Layer 3: built-in
│  Scheduler · sgl-kernel · USP · CFG-parallel    │
│  torch.compile · FA3 · Fused Triton kernels      │
├──────────────────────────────────────────────────┤
│  GPU(s): H100 / A100 / B200 — CUDA + NVLink     │
└──────────────────────────────────────────────────┘
```

> *"Everything in blocks 3–5 — compile, FlashAttention, fused kernels — lives in that bottom box. Layers 2 and 3 of the AWS stack — SGLang gives you out of the box. The only layer you write is your own application on top."*

---

## Block 7 — SGLang as the Serving Layer (8 min) — The Live Demo Close

### Architecture recap (1 slide)

- `**ComposedPipelineBase**` orchestrates `PipelineStage`s: `EncodingStage` → `DenoisingStage` → `DecodingStage`.
- Reuses SGLang's scheduler and sgl-kernel from the LLM stack.
- Built on a FastVideo fork (collaboration is explicit and ongoing).
- Three parallelism axes: USP (Ulysses-SP × Ring-Attention), CFG-parallel, Tensor Parallel.
- Three entry points: CLI, Python API, OpenAI-compatible HTTP server.

### Close the loop on the hook

Call back to Block 0 explicitly:

> *"We opened with a 71-second baseline. Every block since then removed one layer of that cost. Now let's hand all of it to SGLang and run the same prompt."*

### Benchmark headlines

- SGLang Diffusion: **1.2× to 5.9× speedup** vs HuggingFace Diffusers (Nov 2025).
- Jan 2026 update: **up to 2.5× faster** than the initial release, **up to 5×** vs other vendors.
- On our box: same 5-second Wan2.1 clip goes from **~71 s → ~12–18 s**.

### Demo 6a: One-line CLI — same prompt, now with SGLang

```bash
sglang generate \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --prompt "A cat walking through a sunlit garden, cinematic lighting, high quality, detailed" \
  --save-output
```

Play both videos side by side (pre-recorded fallback fine). Let the audience see the quality is identical, the time is not.

### Demo 6b: OpenAI-compatible API (the killer feature)

```bash
# Terminal 1 — start server
sglang serve --model-path black-forest-labs/FLUX.1-dev --port 3000

# Terminal 2 — standard OpenAI image API
curl http://127.0.0.1:3000/v1/images/generations \
  -o >(jq -r '.data[0].b64_json' | base64 --decode > out.png) \
  -H "Content-Type: application/json" \
  -d '{
    "model": "black-forest-labs/FLUX.1-dev",
    "prompt": "A cute baby sea otter",
    "n": 1,
    "size": "1024x1024",
    "response_format": "b64_json"
  }'
```

Then show the Python OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:3000/v1", api_key="EMPTY")
img = client.images.generate(
    model="black-forest-labs/FLUX.1-dev",
    prompt="A robot reading Python docs in a coffee shop, ghibli style",
    size="1024x1024",
)
```

> *"That's it. You just took a 12-billion-parameter diffusion transformer and put it behind the same API your existing OpenAI client already speaks."*

### Demo 6c: Multi-GPU with CFG-parallel (if 2 GPUs available)

```bash
sglang generate \
  --model-path Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
  --prompt "Summer beach vacation, white cat in sunglasses on a surfboard" \
  --image-path ./cat.jpg \
  --num-gpus 2 --enable-cfg-parallel \
  --save-output
```

### Closing the loop — what SGLang does under the hood

- The denoising loop from the profiler → `DenoisingStage`
- `torch.compile` you ran by hand → enabled by default for supported models
- FlashAttention → `sgl-kernel` calling FA2/FA3 automatically
- Fused kernels → also in `sgl-kernel` + FastVideo contributions
- 74K-token attention bottleneck → split across GPUs with USP

### Production checklist (one slide for the room to photograph)

```bash
# Install
uv pip install 'sglang[diffusion]' --prerelease=allow

# Key flags
sglang serve --model-path <MODEL> \
  --enable-torch-compile \           # on by default for many models
  --attention-backend flash_attn_3 \ # Hopper GPUs
  --num-gpus N \                     # multi-GPU
  --enable-cfg-parallel \            # free 2× for CFG models
  --dit-layerwise-offload true \     # save 30GB VRAM, +58% speed
  --port 3000

# Then hit it with any OpenAI SDK
```

---

## Block 8 — Wrap-up + Q&A (4 min)

### One-paragraph summary slide

> Video diffusion is **compute-bound**, not KV-cache-bound. The optimization stack: profile → reduce step time with `torch.compile` → reduce attention time with FlashAttention → fuse the residual hot path with custom Triton kernels → shard the long sequence with USP/CFG-parallel/TP. SGLang Diffusion bundles every layer behind an OpenAI-compatible API. Result: 1.2×–5.9× over Diffusers out of the box.

### Calls to action

- **Try tonight:** `uv pip install 'sglang[diffusion]' --prerelease=allow`
- **Read:** lmsys.org SGLang Diffusion blog (Nov 7 2025) + 2-month update (Jan 16 2026)
- **Contribute:** `#diffusion` on `slack.sglang.io`

---

## Live Demo Risk Log


| Risk                                                | Mitigation                                                                                                                              |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Docker won't work on managed boxes                  | **Confirmed.** Use native pip install. Skip Docker entirely.                                                                            |
| `huggingface-cli` deprecated                        | **Confirmed.** Use `hf download` instead.                                                                                               |
| `sglang.multimodal_gen.VideoGenerator` import fails | **Confirmed.** Use diffusers for baseline profiling; use `sglang generate` CLI for optimized demo. Investigate correct Python API path. |
| HuggingFace download stall                          | Models pre-fetched with `hf download` the night before                                                                                  |
| `torch.compile` compile-time eats 60s               | Run hidden warmup before stage time; show **second** generation                                                                         |
| OOM on smaller GPU                                  | Default to Wan2.1-1.3B; have `--dit-layerwise-offload true` ready                                                                       |
| Multi-GPU demo fails                                | Pre-record output video as fallback                                                                                                     |
| Network flakiness in venue                          | All demos local; offline wheel cache on USB                                                                                             |
| Profiler trace too large for Perfetto               | Pre-export and pre-load in Perfetto tab                                                                                                 |
| Triton kernel won't compile on venue GPU            | Run bench on remote H100 over SSH                                                                                                       |
| Rust compiler missing for install                   | `rustup` install needed before `sglang[diffusion]`                                                                                      |


---

## Open Investigation Items


| Item                                             | Status                                          | Next step                                                                                     |
| ------------------------------------------------ | ----------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Correct SGLang Python API import path            | `VideoGenerator` not in `sglang.multimodal_gen` | Run `python -c "import sglang.multimodal_gen; print(dir(sglang.multimodal_gen))"` to discover |
| SGLang-native profiling (not diffusers baseline) | Blocked on import path                          | Once API found, write a paired profiling script                                               |
| FLUX.1-dev gated access                          | Need to accept license on HF + `hf auth login`  | Do before model download                                                                      |
| `--enable-torch-compile` flag verification       | Untested on this box                            | Test after imports are resolved                                                               |
| Wan2.2-14B multi-GPU test                        | Untested                                        | Needs 2+ GPU box; verify `--num-gpus 2 --enable-cfg-parallel`                                 |


---

## Numbers Cheat Sheet (back pocket on stage)

- Wan2.1 720p / 5s ≈ 74,000 attention tokens per denoising step
- 76% of Wan2.1-1.3B DiT compute, ~91% of 14B, is attention
- FlashAttention-3: 740 TFLOPs/s FP16 (75% H100 peak), ~1.2 PFLOPs/s FP8
- `torch.compile` on FLUX-1-Dev: 6.7 → 4.5s on H100 (1.5×)
- Liger fused Triton kernels: RoPE 8×, RMSNorm 7× vs unfused
- SGLang Diffusion vs Diffusers: **1.2× – 5.9×** (Nov 2025); Jan 2026: further **2.5×** improvement
- `--dit-layerwise-offload`: peak VRAM ↓ 30GB, perf ↑ up to 58%

---

## Caveats to Flag Honestly

- The 1.2×–5.9× range varies per model, GPU, and resolution; consumer GPUs (4090) more mixed.
- Sparse / sliding-tile attention deliver big wins but are not always quality-neutral.
- `torch.compile` recompilation on shape changes is a real production gotcha.
- SGLang Diffusion is young (launched Nov 2025) — APIs may shift. Pin Docker/pip versions.
- FP4 / FA4 are Blackwell-only; on Ampere/Hopper the stack is FA2/FA3 + bf16/fp8 + compile + USP.

---

# references

* https://diffstudy.com/gpu-vs-tpu-vs-npu-ai-workloads/