"""
Text-to-Video Generation with Wan2.1-T2V-1.3B-Diffusers
=========================================================
Generates a short video clip from a text prompt using the
Wan-AI/Wan2.1-T2V-1.3B-Diffusers model via HuggingFace Diffusers.

50-step full model, no distillation.

Acceleration knobs (opt-in):

  1. --guidance-scale  Set to 1.0 to disable CFG (~2x per step).
  2. --scheduler       unipc | dpm | default. UniPC + flow_shift is fastest.
  3. --cfg-parallel    Split CFG batch across 2 GPUs (cuda:0 + cuda:1). Each GPU
                       runs one of the two forward passes (uncond / cond) in
                       parallel. Requires guidance_scale > 1.0.
  4. --teacache        Skip redundant denoising compute (experimental).
  5. --compile         JIT torch.compile the DiT (regional; compiles on 1st run).
  6. --seq-parallel    Unified Sequence Parallelism: splits the attention sequence
                       across N GPUs using a combination of Ulysses all-to-all and
                       Ring attention. Requires torchrun. Mutually exclusive with
                       --cfg-parallel and --cpu-offload.

                       Control the 2-D parallelism grid with:
                         --ulysses-degree U   (default 1 = off)
                         --ring-degree R      (default 2)
                       Total GPUs used = U * R. When both > 1, unified attention
                       is used (requires ≥4 GPUs). When only one is > 1, falls
                       back to pure Ulysses or pure Ring Attention.

Requirements:
    pip install "diffusers>=0.34" torch transformers accelerate imageio[ffmpeg]

Vanilla baseline (50-step, high quality):
    python wan2_text_to_video_parallel.py "A cat in a garden" \
        --num-frames 81 --fps 16 --output output.mp4

CFG parallel on two GPUs:
    python wan2_text_to_video_parallel.py "A cat in a garden" \
        --num-frames 81 --cfg-parallel --fps 16 --output output.mp4

Unified sequence parallelism on 4 GPUs (ulysses=2, ring=2):
    torchrun --nproc-per-node 4 wan2_text_to_video_parallel.py "A cat in a garden" \
        --num-frames 81 --seq-parallel --ulysses-degree 2 --ring-degree 2

Ring-only sequence parallelism on 2 GPUs:
    torchrun --nproc-per-node 2 wan2_text_to_video_parallel.py "A cat in a garden" \
        --num-frames 81 --seq-parallel --ring-degree 2

Profiling (torch.profiler):
    python wan2_text_to_video_parallel.py "A cat in a garden" \
        --num-frames 81 --profile --profile-dir ./profiler_logs

    View results:
        - Chrome:      open chrome://tracing and load profiler_logs/trace.json
        - TensorBoard: tensorboard --logdir ./profiler_logs
"""

import argparse

import torch
from diffusers import (AutoencoderKLWan, WanPipeline, UniPCMultistepScheduler,
                       ContextParallelConfig)
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import export_to_video


# ---------------------------------------------------------------------------
# TeaCache  (experimental)
# ---------------------------------------------------------------------------
def enable_teacache(pipe, threshold: float = 0.08) -> bool:
    """Patch WanTransformer3DModel.forward with a TeaCache-style residual cache.

    Skip signal: relative L1 change of the time-modulation embedding between
    steps. `threshold` accumulates that distance; higher -> more skips -> faster
    but lower fidelity. ~0.05-0.15 is a sane range.
    """
    transformer = pipe.transformer
    required = ["rope", "patch_embedding", "condition_embedder", "blocks",
                "scale_shift_table", "norm_out", "proj_out"]
    missing = [a for a in required if not hasattr(transformer, a)]
    if missing:
        print(f"[warn] TeaCache disabled: transformer is missing {missing}.")
        return False

    try:
        from diffusers.models.modeling_outputs import Transformer2DModelOutput
    except Exception:  # noqa: BLE001
        Transformer2DModelOutput = None

    state = {"prev_mod": None, "acc": 0.0, "residual": None}
    transformer._tc_state = state
    transformer._tc_threshold = float(threshold)

    def reset():
        state["prev_mod"] = None
        state["acc"] = 0.0
        state["residual"] = None

    transformer.reset_teacache = reset

    def teacache_forward(self, hidden_states, timestep, encoder_hidden_states,
                         encoder_hidden_states_image=None, return_dict=True,
                         attention_kwargs=None):
        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        rotary_emb = self.rope(hidden_states)
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(timestep, encoder_hidden_states,
                                    encoder_hidden_states_image)
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1)

        mod = timestep_proj.detach()
        st = self._tc_state
        if st["prev_mod"] is None or st["residual"] is None:
            should_calc = True
        else:
            denom = st["prev_mod"].abs().mean()
            rel = ((mod - st["prev_mod"]).abs().mean()
                   / (denom + 1e-8)).item()
            st["acc"] += rel
            should_calc = st["acc"] >= self._tc_threshold
            if should_calc:
                st["acc"] = 0.0
        st["prev_mod"] = mod

        if should_calc:
            ori = hidden_states
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states,
                                      timestep_proj, rotary_emb)
            st["residual"] = (hidden_states - ori).detach()
        else:
            hidden_states = hidden_states + st["residual"]

        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift
                         ).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, ppf, pph, ppw, p_t, p_h, p_w, -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)
        if Transformer2DModelOutput is not None:
            return Transformer2DModelOutput(sample=output)
        return (output,)

    import types
    transformer._tc_orig_forward = transformer.forward
    transformer.forward = types.MethodType(teacache_forward, transformer)
    print(f"TeaCache enabled (threshold={threshold}). Experimental — tune the "
          f"threshold and verify quality.")
    return True


# ---------------------------------------------------------------------------
# CFG parallelism across two GPUs
# ---------------------------------------------------------------------------
def setup_cfg_parallel(pipe, model_id: str,
                       device0: str = "cuda:0", device1: str = "cuda:1") -> None:
    """Run the cond and uncond CFG passes in parallel across two GPUs.

    WanPipeline makes two separate sequential transformer calls per denoising
    step: first cond (positive prompt), then uncond (negative prompt), using
    the same latent/timestep but different encoder_hidden_states. This wrapper
    intercepts those calls using a two-call state machine:

      Step 1 (learning):   both calls run sequentially on device0 to capture
                           the negative prompt embeds and extra kwargs.
      Steps 2..N (parallel): at the cond call, launch cond on device0 AND uncond
                           on device1 simultaneously via two threads; cache the
                           device1 result. At the uncond call, return the cached
                           result (already computed) moved to device0.

    All other pipeline components (VAE, text encoder, scheduler) stay on device0.
    Pipe must already be on device0 before calling this function.
    """
    import threading
    from diffusers import WanTransformer3DModel

    dtype = pipe.transformer.dtype
    print(f"CFG parallel: loading second transformer on {device1} …")
    transformer1 = WanTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=dtype
    ).to(device1)
    transformer1.eval()

    transformer0 = pipe.transformer

    # call_idx 0 = next call is cond, 1 = next call is uncond
    # neg_embeds_d1 / neg_kwargs_d1: GPU-1 copies, created once and reused.
    state = {
        "call_idx": 0,
        "neg_embeds": None, "neg_kwargs": None,
        "neg_embeds_d1": None, "neg_kwargs_d1": None,
        "pending": None,
    }

    def _to(v, dev):
        return v.to(dev) if isinstance(v, torch.Tensor) else v

    def _run(transformer, hs, ts, ehs, kw, out, errors, idx):
        # torch.no_grad() is thread-local and not inherited by new threads,
        # so we must re-enter it here to avoid building a computation graph
        # across all transformer blocks (which accumulates ~78 GiB and OOMs).
        try:
            with torch.inference_mode():
                out[idx] = transformer(
                    hidden_states=hs,
                    timestep=ts,
                    encoder_hidden_states=ehs,
                    **kw,
                )
        except Exception as e:
            errors[idx] = e

    def _repack(raw, return_dict, target_device):
        """Extract the sample tensor, move to target_device, and repack."""
        if hasattr(raw, "sample"):
            sample = raw.sample.to(target_device)
        elif isinstance(raw, (tuple, list)):
            sample = raw[0].to(target_device)
        else:
            sample = raw.to(target_device)
        if not return_dict:
            return (sample,)
        try:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput
            return Transformer2DModelOutput(sample=sample)
        except ImportError:
            return (sample,)

    def _split_call(hidden_states, timestep, encoder_hidden_states, **kwargs):
        call_idx = state["call_idx"]
        return_dict = kwargs.get("return_dict", True)

        if call_idx == 0:
            # ---- cond call ----
            state["call_idx"] = 1

            if state["neg_embeds"] is not None:
                # Parallel: cond on GPU 0, uncond on GPU 1 simultaneously.
                # Pre-copy all GPU-1 inputs before launching threads so that
                # no cross-device transfer competes with t0's forward pass on
                # GPU 0 (concurrent P2P staging can exhaust GPU 0's allocator).
                if state["neg_embeds_d1"] is None:
                    state["neg_embeds_d1"] = state["neg_embeds"].to(device1)
                    state["neg_kwargs_d1"] = {k: _to(v, device1)
                                              for k, v in state["neg_kwargs"].items()}
                hs1 = hidden_states.to(device1)
                ts1 = timestep.to(device1)
                torch.cuda.synchronize(device1)  # copies done before threads start

                results = [None, None]
                errors = [None, None]
                t0 = threading.Thread(target=_run, args=(
                    transformer0,
                    hidden_states, timestep, encoder_hidden_states, kwargs,
                    results, errors, 0))
                t1 = threading.Thread(target=_run, args=(
                    transformer1,
                    hs1, ts1, state["neg_embeds_d1"], state["neg_kwargs_d1"],
                    results, errors, 1))
                t0.start(); t1.start()
                t0.join(); t1.join()
                for i, e in enumerate(errors):
                    if e is not None:
                        raise RuntimeError(f"CFG parallel: GPU {i} error") from e
                state["pending"] = results[1]   # uncond result; returned on next call
                return results[0]               # cond result; already on device0
            else:
                # Learning step: run cond sequentially on GPU 0
                return transformer0(
                    hidden_states=hidden_states, timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states, **kwargs)

        else:
            # ---- uncond call ----
            state["call_idx"] = 0

            if state["neg_embeds"] is None:
                # Learning step: capture neg embeds, run uncond sequentially on GPU 0
                state["neg_embeds"] = encoder_hidden_states
                state["neg_kwargs"] = kwargs
                return transformer0(
                    hidden_states=hidden_states, timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states, **kwargs)
            else:
                # Return cached uncond result (computed in parallel with previous cond)
                raw = state["pending"]
                state["pending"] = None
                return _repack(raw, return_dict, device0)

    # Proxy: routes __call__ through _split_call; delegates all attr access to
    # transformer0 so compile, teacache, etc. still work transparently.
    # _t1 exposes transformer1 so the caller can also compile it.
    class _CFGParallelWrapper:
        def __init__(self, t0, t1):
            object.__setattr__(self, "_t0", t0)
            object.__setattr__(self, "_t1", t1)

        def __call__(self, *args, **kwargs):
            return _split_call(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_t0"), name)

        def __setattr__(self, name, value):
            if name in ("_t0", "_t1"):
                object.__setattr__(self, name, value)
            else:
                setattr(object.__getattribute__(self, "_t0"), name, value)

    pipe.transformer = _CFGParallelWrapper(transformer0, transformer1)
    print(f"CFG parallel ready: cond on {device0}, uncond on {device1} (in parallel).")


# ---------------------------------------------------------------------------
# Unified Sequence Parallelism
# ---------------------------------------------------------------------------
def setup_seq_parallel(pipe, ulysses_degree: int, ring_degree: int) -> int:
    """Enable context parallelism on the transformer across torchrun ranks.

    With both degrees > 1, Unified Sequence Parallelism (USP) is activated:
    Ulysses all-to-all first redistributes heads/tokens, Ring attention handles
    the sequence split, then a second all-to-all restores the layout. With only
    one degree > 1 it falls back to pure Ulysses or pure Ring attention.

    Total GPUs required = ulysses_degree * ring_degree.

    Must be called *after* the pipeline has been moved to the local device.
    Returns the local rank so callers can gate per-rank operations.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    local_device = f"cuda:{rank}"
    torch.cuda.set_device(local_device)
    pipe.to(local_device)

    if ring_degree > 1:
        # Ring attention requires return_lse=True for cross-rank softmax normalisation.
        # The default "native" backend (torch SDPA) does not expose log-sum-exp output,
        # so we switch to cuDNN which supports it via compute_log_sumexp.
        from diffusers.models.attention_dispatch import _AttentionBackendRegistry, AttentionBackendName
        _AttentionBackendRegistry.set_active_backend(AttentionBackendName._NATIVE_CUDNN)

    cp_config = ContextParallelConfig(
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
    )
    pipe.transformer.enable_parallelism(config=cp_config)

    if rank == 0:
        total = ulysses_degree * ring_degree
        if ulysses_degree > 1 and ring_degree > 1:
            mode = f"unified (ulysses={ulysses_degree} × ring={ring_degree})"
        elif ulysses_degree > 1:
            mode = f"ulysses-only (degree={ulysses_degree})"
        else:
            mode = f"ring-only (degree={ring_degree})"
        print(f"Sequence parallelism: {mode} across {total} GPUs.")

    return rank


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_video(
    prompt: str,
    negative_prompt: str = "",
    output_path: str = "output.mp4",
    num_frames: int = 33,
    height: int = 480,
    width: int = 832,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    seed: int | None = None,
    fps: int = 16,
    dtype: str = "float16",
    # --- speed knobs ---
    scheduler: str = "default",
    flow_shift: float = 3.0,
    cfg_parallel: bool = False,
    teacache: bool = False,
    teacache_threshold: float = 0.08,
    cpu_offload: bool = False,
    # --- sequence parallelism ---
    seq_parallel: bool = False,
    ulysses_degree: int = 1,
    ring_degree: int = 2,
    # --- compile / profile ---
    compile: bool = False,
    compile_mode: str = "default",
    compile_vae: bool = False,
    warmup: int | None = None,
    monitor_memory: bool = False,
    profile: bool = False,
    profile_dir: str = "./profiler_logs",
):
    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]

    # --- Incompatibility checks ---------------------------------------------
    if seq_parallel:
        if cfg_parallel:
            print("[warn] --seq-parallel is incompatible with --cfg-parallel; "
                  "disabling cfg-parallel.")
            cfg_parallel = False
        if cpu_offload:
            print("[warn] --seq-parallel is incompatible with --cpu-offload; "
                  "disabling cpu-offload.")
            cpu_offload = False

    # --- Load VAE & Pipeline ------------------------------------------------
    print("Loading VAE …")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32,
    )

    print("Loading pipeline …")
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch_dtype)

    # --- Scheduler ----------------------------------------------------------
    if scheduler == "unipc":
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=flow_shift)
        print(f"Scheduler: UniPC (flow_shift={flow_shift}).")
    elif scheduler == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        print("Scheduler: DPMSolverMultistep.")
    else:
        print("Scheduler: pipeline default.")

    # --- Device / offload ---------------------------------------------------
    _seq_rank = 0  # local rank; stays 0 for non-distributed modes
    if seq_parallel:
        # setup_seq_parallel handles dist init, device placement, and enable_parallelism
        _seq_rank = setup_seq_parallel(pipe, ulysses_degree, ring_degree)
        _main_device = f"cuda:{_seq_rank}"
    elif cpu_offload:
        if cfg_parallel:
            print("[warn] --cfg-parallel is incompatible with --cpu-offload; "
                  "disabling cfg-parallel.")
            cfg_parallel = False
        pipe.enable_model_cpu_offload()
        print("Model CPU offload enabled (slower, low VRAM).")
    else:
        pipe.to("cuda:0" if cfg_parallel else "cuda")

    # --- CFG parallelism (two GPUs) -----------------------------------------
    if cfg_parallel:
        if guidance_scale <= 1.0:
            print("[warn] --cfg-parallel has no effect when guidance_scale<=1.0 "
                  "(CFG disabled). Running on cuda:0 only.")
            cfg_parallel = False
        else:
            setup_cfg_parallel(pipe, model_id, "cuda:0", "cuda:1")

    # --- Optional knobs -----------------------------------------------------
    if teacache:
        if compile:
            print("[warn] TeaCache + torch.compile: dynamic skipping causes graph "
                  "breaks/recompiles. Prefer one or the other.")
        enable_teacache(pipe, teacache_threshold)

    # --- torch.compile (JIT, regional) --------------------------------------
    if compile:
        torch.set_float32_matmul_precision("high")
        print(f"Compiling DiT repeated blocks (mode={compile_mode!r}) — first step is slow.")
        pipe.transformer.compile_repeated_blocks(
            mode=compile_mode, fullgraph=False, dynamic=False)
        if compile_vae:
            pipe.vae.decoder = torch.compile(
                pipe.vae.decoder, mode=compile_mode, fullgraph=False, dynamic=False)

        if compile_mode in ("reduce-overhead", "max-autotune"):
            def _mark_step_begin(_mod, _args, _kwargs):
                torch.compiler.cudagraph_mark_step_begin()

            def _clone_output_hook(_mod, _args, _kwargs, output):
                if isinstance(output, torch.Tensor):
                    return output.clone()
                if isinstance(output, tuple):
                    return tuple(o.clone() if isinstance(o, torch.Tensor) else o
                                 for o in output)
                if hasattr(output, "sample") and isinstance(output.sample, torch.Tensor):
                    output.sample = output.sample.clone()
                return output

            pipe.transformer.register_forward_pre_hook(_mark_step_begin, with_kwargs=True)
            pipe.transformer.register_forward_hook(_clone_output_hook, with_kwargs=True)
            if compile_vae:
                pipe.vae.decoder.register_forward_pre_hook(_mark_step_begin, with_kwargs=True)
                pipe.vae.decoder.register_forward_hook(_clone_output_hook, with_kwargs=True)

    if guidance_scale <= 1.0:
        print("CFG disabled (guidance_scale<=1.0): ~2x faster per step; "
              "negative prompt is ignored.")
        negative_prompt = ""

    if teacache and hasattr(pipe.transformer, "reset_teacache"):
        pipe.transformer.reset_teacache()

    # --- Generate -----------------------------------------------------------
    if not seq_parallel:
        _main_device = "cuda:0" if cfg_parallel else "cuda"
    generator = torch.Generator(device=_main_device)
    if seed is not None:
        generator.manual_seed(seed)

    def _run_pipe(steps=None, gen=None):
        return pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=steps if steps is not None else num_inference_steps,
            guidance_scale=guidance_scale,
            generator=gen if gen is not None else generator,
        )

    # --- Warmup -------------------------------------------------------------
    if warmup is None:
        warmup = 1 if compile else 0
    if warmup > 0:
        print(f"Warmup: {warmup} run(s) to trigger compilation "
              f"(excluded from reported time below).")
        warmup_gen = torch.Generator(device=_main_device)
        warmup_steps = max(2, min(num_inference_steps, 4))
        for _ in range(warmup):
            _run_pipe(steps=warmup_steps, gen=warmup_gen)
            if teacache and hasattr(pipe.transformer, "reset_teacache"):
                pipe.transformer.reset_teacache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("Warmup complete.")

    print(f"Generating {num_frames} frames at {width}×{height}, "
          f"{num_inference_steps} steps …")

    import time as _time

    def _sync_all():
        if torch.cuda.is_available():
            if seq_parallel:
                torch.cuda.synchronize()  # syncs the current rank's device
            else:
                torch.cuda.synchronize("cuda:0")
                if cfg_parallel:
                    torch.cuda.synchronize("cuda:1")

    _sync_all()
    if monitor_memory and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _mem_resident = torch.cuda.memory_allocated()
    _gen_start = _time.perf_counter()

    if profile:
        import os
        from torch.profiler import (profile as torch_profile, ProfilerActivity,
                                    tensorboard_trace_handler)

        os.makedirs(profile_dir, exist_ok=True)
        print(f"Profiling enabled → writing traces to {profile_dir}")
        with torch_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=True, with_stack=True,
            on_trace_ready=tensorboard_trace_handler(profile_dir),
        ) as prof:
            result = _run_pipe()
        print("\n=== Top 20 ops by CUDA time ===")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        chrome_trace = os.path.join(profile_dir, "trace.json")
        prof.export_chrome_trace(chrome_trace)
        print(f"Chrome trace: {chrome_trace}")
        print(f"TensorBoard:  tensorboard --logdir {profile_dir}")
    else:
        result = _run_pipe()

    _sync_all()
    _gen_elapsed = _time.perf_counter() - _gen_start

    _is_main = (not seq_parallel) or (_seq_rank == 0)
    if _is_main:
        print(f"Generation time (post-warmup): {_gen_elapsed:.2f}s "
              f"({_gen_elapsed / num_inference_steps:.2f}s/step).")

    if monitor_memory and torch.cuda.is_available():
        gb = 1024 ** 3
        peak_alloc = torch.cuda.max_memory_allocated() / gb
        peak_resv = torch.cuda.max_memory_reserved() / gb
        rank_tag = f"rank{_seq_rank} " if seq_parallel else ""
        print(f"GPU memory {rank_tag}— resident weights: {_mem_resident / gb:.2f} GiB; "
              f"peak allocated: {peak_alloc:.2f} GiB; "
              f"peak reserved: {peak_resv:.2f} GiB "
              f"({torch.cuda.get_device_name()}).")

    if _is_main:
        frames = result.frames[0]
        export_to_video(frames, output_path, fps=fps)
        print(f"✓ Video saved to {output_path}  ({len(frames)} frames, {fps} fps)")

    if seq_parallel:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wan2.1-T2V-1.3B text-to-video. 50-step full model."
    )
    parser.add_argument(
        "prompt", nargs="?",
        default=(
            "A cat walking through a sunlit garden, "
            "cinematic lighting, high quality, detailed"
        ),
        help="Text prompt describing the video.",
    )
    parser.add_argument("--negative-prompt", type=str, default="",
                        help="Negative prompt.")
    parser.add_argument("--output", type=str, default="output.mp4",
                        help="Output file path.")
    parser.add_argument("--num-frames", type=int, default=33,
                        help="Number of frames (should be 4k+1).")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=50,
                        help="Denoising steps.")
    parser.add_argument("--guidance-scale", type=float, default=5.0,
                        help="CFG scale. Set to 1.0 to disable CFG (~2x faster/step).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="float16",
                        help="Transformer/text-encoder dtype (vae always fp32).")

    # speed knobs
    parser.add_argument("--cfg-parallel", action="store_true",
                        help="Split CFG batch across cuda:0 and cuda:1 in parallel. "
                             "Requires two GPUs and guidance_scale > 1.0.")
    parser.add_argument("--seq-parallel", action="store_true",
                        help="Enable sequence parallelism (Ulysses/Ring/Unified) across "
                             "N GPUs. Must be launched with torchrun --nproc-per-node N. "
                             "Mutually exclusive with --cfg-parallel and --cpu-offload.")
    parser.add_argument("--ulysses-degree", type=int, default=1,
                        help="Ulysses attention degree (number of GPUs for head "
                             "parallelism). Default 1 (off). Use with --seq-parallel.")
    parser.add_argument("--ring-degree", type=int, default=2,
                        help="Ring attention degree (number of GPUs for sequence "
                             "parallelism). Default 2. Use with --seq-parallel. "
                             "Set both --ulysses-degree and --ring-degree > 1 for "
                             "unified sequence parallelism (requires ≥4 GPUs).")
    parser.add_argument("--scheduler", choices=["unipc", "dpm", "default"],
                        default="default",
                        help="Scheduler. unipc is fastest; default is the model's own.")
    parser.add_argument("--flow-shift", type=float, default=3.0,
                        help="UniPC flow_shift: ~3.0 @480p, ~5.0 @720p.")
    parser.add_argument("--teacache", action="store_true",
                        help="TeaCache step-skipping (experimental).")
    parser.add_argument("--teacache-threshold", type=float, default=0.08,
                        help="TeaCache skip threshold (higher = more skips, lower quality).")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="enable_model_cpu_offload for <12 GB VRAM (slower).")

    # compile / profile
    parser.add_argument("--compile", action="store_true",
                        help="JIT torch.compile the DiT via regional compilation.")
    parser.add_argument("--compile-mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode.")
    parser.add_argument("--compile-vae", action="store_true",
                        help="Also torch.compile the VAE decoder.")
    parser.add_argument("--warmup", type=int, default=None,
                        help="Warmup runs before the timed generation. "
                             "Default: 1 when --compile is set, else 0.")
    parser.add_argument("--monitor-memory", action="store_true",
                        help="Report peak GPU memory (allocated/reserved) for the run.")
    parser.add_argument("--profile", action="store_true",
                        help="Enable torch.profiler around the inference run.")
    parser.add_argument("--profile-dir", type=str, default="./profiler_logs",
                        help="Directory for profiler trace + TensorBoard logs.")

    args = parser.parse_args()

    generate_video(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        output_path=args.output,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        fps=args.fps,
        dtype=args.dtype,
        scheduler=args.scheduler,
        flow_shift=args.flow_shift,
        cfg_parallel=args.cfg_parallel,
        teacache=args.teacache,
        teacache_threshold=args.teacache_threshold,
        cpu_offload=args.cpu_offload,
        seq_parallel=args.seq_parallel,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        compile=args.compile,
        compile_mode=args.compile_mode,
        compile_vae=args.compile_vae,
        warmup=args.warmup,
        monitor_memory=args.monitor_memory,
        profile=args.profile,
        profile_dir=args.profile_dir,
    )
