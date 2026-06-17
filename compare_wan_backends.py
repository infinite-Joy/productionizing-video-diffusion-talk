#!/usr/bin/env python3
"""
compare_wan_backends.py
=======================

Load Wan-AI/Wan2.1-T2V-1.3B-Diffusers, torch.compile() the transformer, and dump
*exactly which kernels each backend produces* so you can diff Inductor vs
Torch-TensorRT.

What "kernel replacement" means for each backend
------------------------------------------------
* Inductor lowers the captured aten/prims graph into FUSED kernels
  (Triton on GPU, C++ on CPU). Each generated kernel is named after the aten
  ops it fused, e.g. `triton_poi_fused_add_mul_native_layer_norm_3`. We capture
  these via TORCH_COMPILE_DEBUG=1 (writes an `output_code.py` per graph) and the
  `output_code` / `fusion` log channels.

* Torch-TensorRT PARTITIONS the graph: supported subgraphs are converted into
  TRT engines (`_run_on_acc_*` submodules), everything else stays in Torch
  (`_run_on_gpu_*`). We capture this via the `dryrun` report (fast, no engine
  build) and/or `debug` logging (real build). The report lists every aten op and
  which engine absorbed it.

The script writes a `summary_<backend>.json` per run, then `--compare` diffs them.

Usage
-----
    # 1) Inductor run (captures generated Triton kernels)
    python compare_wan_backends.py --backend inductor

    # 2) TensorRT run. Use --dryrun first: it shows the partition WITHOUT the
    #    (slow) engine build, which is all you need to see op replacement.
    python compare_wan_backends.py --backend tensorrt --dryrun
    #    ...or do the real build:
    python compare_wan_backends.py --backend tensorrt

    # 3) Diff the two
    python compare_wan_backends.py --compare

Requirements
------------
    pip install "diffusers>=0.33" transformers accelerate ftfy
    # CUDA + Triton come with a normal GPU torch build (Inductor needs nothing extra)
    # For the TensorRT backend you also need a matching torch-tensorrt + TensorRT:
    pip install torch-tensorrt        # must match your torch / CUDA version
A CUDA GPU is required. The 1.3B transformer is >1B params, so the TRT engine
build can be slow and memory-hungry on the first forward pass; prefer --dryrun
while exploring.
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Arg parsing FIRST: several Inductor knobs are environment variables that must
# be set *before* `import torch`, so we parse args before importing torch.
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["inductor", "tensorrt"],
                   help="Which torch.compile backend to run.")
    p.add_argument("--compare", action="store_true",
                   help="Diff summary_inductor.json and summary_tensorrt.json.")
    p.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--outdir", default="./backend_debug",
                   help="Where logs / debug artifacts / summaries are written.")
    # inference size — kept tiny; compilation triggers on the first forward
    # regardless of step count, so 1-2 steps is enough to capture kernels.
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=9, help="must be 4*k+1")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--mode", default="default",
                   choices=["default", "reduce-overhead",
                            "max-autotune", "max-autotune-no-cudagraphs"],
                   help="Inductor compile mode. 'default' gives the cleanest "
                        "kernel listing; max-autotune emits more tuning kernels.")
    # TensorRT-specific
    p.add_argument("--dryrun", action="store_true",
                   help="[tensorrt] partition + report only, skip engine build.")
    p.add_argument("--min-block-size", type=int, default=5,
                   help="[tensorrt] smaller => more aggressive op->TRT conversion.")
    p.add_argument("--trt-precision", default="bf16",
                   choices=["bf16", "fp16", "fp32"])
    return p.parse_args()


ARGS = parse_args()
os.makedirs(ARGS.outdir, exist_ok=True)

# ---------------------------------------------------------------------------
# Inductor debug env vars (no effect on the TRT path, harmless to always set).
# TORCH_COMPILE_DEBUG=1 dumps a full artifact tree incl. output_code.py.
# TORCH_LOGS selects the textual channels we also mirror to a file.
# ---------------------------------------------------------------------------
if ARGS.backend == "inductor":
    INDUCTOR_DEBUG_DIR = os.path.join(os.path.abspath(ARGS.outdir), "torch_compile_debug")
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = INDUCTOR_DEBUG_DIR
    os.environ.setdefault("TORCH_LOGS", "output_code,fusion,graph_breaks,recompiles")

if ARGS.backend == "tensorrt":
    # TORCH_COMPILE_DEBUG captures output_code.py for torch-fallback subgraphs
    # (the GPU-fallback segments are still lowered by Inductor, so this is useful).
    TRT_COMPILE_DEBUG_DIR = os.path.join(os.path.abspath(ARGS.outdir), "torch_compile_debug")
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = TRT_COMPILE_DEBUG_DIR
    # +dynamo shows per-node partition decisions (Supported / Unsupported).
    # output_code captures Inductor kernels for the torch-fallback segments.
    os.environ.setdefault("TORCH_LOGS", "+dynamo,output_code,graph_breaks,recompiles")

import json
import glob
import logging
import re
import textwrap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def attach_file_logger(logger_names, path, level=logging.DEBUG):
    """Mirror the named loggers into `path` so we can parse them afterwards."""
    fh = logging.FileHandler(path, mode="w")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
    for name in logger_names:
        lg = logging.getLogger(name)
        lg.setLevel(level)
        lg.addHandler(fh)
    return fh


def build_pipeline(model_id):
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline

    print(f"[load] {model_id}")
    # VAE stays fp32 for decode quality (per model card); transformer is bf16.
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae",
                                           torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae,
                                       torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    return pipe


def run_inference(pipe):
    """Triggers compilation on the first forward through the transformer."""
    import torch
    prompt = "A cat walks on the grass, realistic"
    neg = "blurred, low quality, jpeg artifacts, distorted"
    print(f"[run] {ARGS.width}x{ARGS.height}  frames={ARGS.num_frames}  "
          f"steps={ARGS.steps}  (first call also compiles)")
    with torch.no_grad():
        pipe(prompt=prompt, negative_prompt=neg,
             height=ARGS.height, width=ARGS.width,
             num_frames=ARGS.num_frames, num_inference_steps=ARGS.steps,
             guidance_scale=5.0)
    print("[run] done")


# ===========================================================================
# INDUCTOR
# ===========================================================================
def run_inductor():
    import torch

    text_log = os.path.join(ARGS.outdir, "inductor_textual.log")
    attach_file_logger(["torch._inductor", "torch._dynamo",
                        "torch._inductor.compile_fx"], text_log)
    # Also push the structured channels to file via the logging API.
    try:
        torch._logging.set_logs(output_code=True, fusion=True,
                                graph_breaks=True, recompiles=True)
    except Exception as e:  # older/newer torch — env var already covers us
        print(f"[inductor] set_logs partial: {e}")

    pipe = build_pipeline(ARGS.model_id)
    print(f"[inductor] torch.compile(transformer, backend='inductor', mode='{ARGS.mode}')")
    pipe.transformer = torch.compile(pipe.transformer,
                                     backend="inductor", mode=ARGS.mode,
                                     fullgraph=False)
    run_inference(pipe)

    summary = parse_inductor_artifacts(INDUCTOR_DEBUG_DIR, text_log)
    save_summary("inductor", summary)
    print_inductor_summary(summary)


def parse_inductor_artifacts(debug_dir, text_log):
    """Extract generated kernels + the aten ops fused into each."""
    kernels = {}        # kernel_name -> sorted list of source aten ops
    output_code_files = glob.glob(os.path.join(debug_dir, "**", "output_code.py"),
                                  recursive=True)

    # Pattern: Inductor prefixes each kernel call in `call()` with a comment
    #   # Topologically Sorted Source Nodes: [aten.add, aten.mul, ...]
    # immediately followed by a `triton_..._fused_...` / `cpp_fused_...` use.
    src_re = re.compile(r"Topologically Sorted Source Nodes:\s*\[(.*?)\]")
    kname_re = re.compile(r"\b((?:triton_(?:poi|red|per|tem|mem|unk)_)?(?:cpp_)?fused_[a-z0-9_]+_\d+)\b")
    # also catch async_compile.triton('name', ...) definitions
    def_re = re.compile(r"async_compile\.triton\(\s*['\"]([a-z0-9_]+)['\"]")

    for f in sorted(output_code_files):
        with open(f) as fh:
            lines = fh.readlines()
        for name in def_re.findall("".join(lines)):
            kernels.setdefault(name, [])
        pending_ops = None
        for ln in lines:
            m = src_re.search(ln)
            if m:
                pending_ops = [o.strip() for o in m.group(1).split(",") if o.strip()]
                continue
            for kn in kname_re.findall(ln):
                kernels.setdefault(kn, [])
                if pending_ops:
                    kernels[kn] = sorted(set(kernels[kn]) | set(pending_ops))
            pending_ops = None

    # Tally the distinct aten ops that ended up inside ANY fused kernel.
    fused_ops = sorted({op for ops in kernels.values() for op in ops})
    return {
        "backend": "inductor",
        "mode": ARGS.mode,
        "num_generated_kernels": len(kernels),
        "kernels": {k: kernels[k] for k in sorted(kernels)},
        "fused_aten_ops": fused_ops,
        "output_code_files": output_code_files,
        "textual_log": text_log,
    }


def print_inductor_summary(s):
    print("\n" + "=" * 78)
    print(f"INDUCTOR — {s['num_generated_kernels']} generated kernels "
          f"(mode={s['mode']})")
    print("=" * 78)
    for name, ops in list(s["kernels"].items())[:40]:
        ops_str = ", ".join(ops) if ops else "(no source-node annotation)"
        print(f"  {name}")
        print(textwrap.fill(ops_str, width=74,
                            initial_indent="      fuses: ",
                            subsequent_indent="             "))
    if s["num_generated_kernels"] > 40:
        print(f"  ... (+{s['num_generated_kernels'] - 40} more; see summary json)")
    print(f"\n  Full Triton/C++ source: {s['output_code_files']}")


# ===========================================================================
# TENSORRT
# ===========================================================================
def run_tensorrt():
    import contextlib
    import torch
    try:
        import torch_tensorrt
    except ImportError:
        sys.exit("torch-tensorrt is not installed. `pip install torch-tensorrt` "
                 "(must match your torch / CUDA / TensorRT versions).")

    trt_log = os.path.join(ARGS.outdir, "tensorrt.log")
    debugger_dir = os.path.join(os.path.abspath(ARGS.outdir), "trt_debugger")
    os.makedirs(debugger_dir, exist_ok=True)

    # Set TRT Python logging to Debug BEFORE any compile call so the
    # partitioner + converter messages are captured from the start.
    try:
        torch_tensorrt.logging.set_reportable_log_level(
            torch_tensorrt.logging.Level.Debug)
    except Exception as e:
        print(f"[tensorrt] set_reportable_log_level: {e}")

    # Enable dynamo structured logs: per-node partition decisions show up as
    # "aten.op: Supported/Unsupported" in the +dynamo channel.
    try:
        torch._logging.set_logs(graph=True, graph_breaks=True, recompiles=True)
    except Exception as e:
        print(f"[tensorrt] set_logs partial: {e}")

    prec = {"bf16": torch.bfloat16, "fp16": torch.float16,
            "fp32": torch.float32}[ARGS.trt_precision]

    options = {
        "debug": True,                          # verbose per-layer / conversion log
        "dryrun": ARGS.dryrun,                  # partition report, skip engine build
        "min_block_size": ARGS.min_block_size,  # smaller => more ops pulled into TRT
        "enabled_precisions": {prec, torch.float32},
        "use_python_runtime": True,
    }

    pipe = build_pipeline(ARGS.model_id)
    print(f"[tensorrt] torch.compile(transformer, backend='torch_tensorrt', "
          f"dryrun={ARGS.dryrun}, min_block_size={ARGS.min_block_size}, "
          f"precision={ARGS.trt_precision})")

    # Capture every Python logger that carries useful TRT / dynamo info.
    attach_file_logger([
        "torch_tensorrt",
        "torch_tensorrt.dynamo",
        "torch_tensorrt.dynamo.partitioning",
        "torch_tensorrt.dynamo.conversion",
        "torch_tensorrt.dynamo.lowering",
        "torch._dynamo",
        "torch._dynamo.backends",
        "torch._inductor",          # Inductor kernels for torch-fallback segments
    ], trt_log)

    # Build a stack of debug contexts so we can enter all of them together.
    # Compilation is lazy: the actual engine build / partition happens on the
    # first forward pass (inside run_inference), so contexts must wrap that call.
    debug_contexts = []

    # 1. Debugger — writes layer info JSON + engine profiles to debugger_dir.
    #    save_layer_info=True gives per-layer {Name, LayerType} we parse later.
    try:
        from torch_tensorrt.dynamo import Debugger
        debug_contexts.append(Debugger(
            log_level="debug",
            logging_dir=debugger_dir,
            save_layer_info=True,
        ))
        print("[tensorrt] Debugger → " + debugger_dir)
    except Exception as e:
        print(f"[tensorrt] Debugger unavailable ({e}); falling back to logging ctx")

    # 2. graphs() — emits IR of each converted subgraph to the log.
    try:
        debug_contexts.append(torch_tensorrt.logging.graphs())
    except Exception:
        pass

    # 3. debug() — enables verbose TRT engine-build messages.
    try:
        debug_contexts.append(torch_tensorrt.logging.debug())
    except Exception:
        pass

    if not debug_contexts:
        print("[tensorrt] WARNING: no debug context available; logs will be sparse.")

    pipe.transformer = torch.compile(pipe.transformer,
                                     backend="torch_tensorrt",
                                     dynamic=False, options=options,
                                     fullgraph=False)

    with contextlib.ExitStack() as stack:
        for ctx in debug_contexts:
            stack.enter_context(ctx)
        run_inference(pipe)

    # Collect every log file produced by Debugger + our own attach_file_logger.
    candidate_logs = [
        trt_log,
        os.path.join(ARGS.outdir, "torch_tensorrt_logging.log"),
        *glob.glob(os.path.join(debugger_dir, "*.log")),
        *glob.glob(os.path.join(debugger_dir, "**", "*.log"), recursive=True),
    ]
    # Debugger writes per-engine layer info as JSON when save_layer_info=True.
    layer_info_files = [
        *glob.glob(os.path.join(debugger_dir, "*.json")),
        *glob.glob(os.path.join(debugger_dir, "**", "*.json"), recursive=True),
    ]

    summary = parse_tensorrt_logs(
        [p for p in candidate_logs if os.path.exists(p)],
        [p for p in layer_info_files if os.path.exists(p)],
    )
    save_summary("tensorrt", summary)
    print_tensorrt_summary(summary)


def parse_tensorrt_logs(log_paths, layer_info_files=None):
    """Extract TRT engines, converted aten ops, torch-fallback ops, and TRT layer types."""
    text = ""
    for p in log_paths:
        with open(p, errors="ignore") as fh:
            text += fh.read() + "\n"

    # Partition segments: TRT engines are `_run_on_acc_N`, torch fallbacks are
    # `_run_on_gpu_N`. The dryrun/debug report lists ops per segment.
    acc_segments = sorted(set(re.findall(r"_run_on_acc_\d+", text)))
    gpu_segments = sorted(set(re.findall(r"_run_on_gpu_\d+", text)))

    # Op coverage summary line from the dryrun report.
    coverage = re.findall(r"consists of\s+\d+\s+Total Operators.*?supported.*", text)

    # --- ops converted into TRT engines ---
    # Multiple log formats from different torch-tensorrt versions / log levels:
    #   "Converting node: target=aten.add.Tensor"
    #   "  aten.add.Tensor : Supported"          (partition decision table)
    #   "Adding aten.add.Tensor to TRT subgraph"
    converted_ops = sorted(set(
        re.findall(r"Converting node[^\n]*?(?:target=)?(aten\.[a-z0-9_\.]+)", text)
        + re.findall(r"(aten\.[a-z0-9_\.]+)\s*[:\|]\s*(?:Supported|True)\b", text)
        + re.findall(r"Adding\s+(aten\.[a-z0-9_\.]+)[^\n]*TRT", text)
        + re.findall(r"Supported[^\n]*(aten\.[a-z0-9_\.]+)", text)
    ))

    # --- ops left in Torch (fallback) ---
    torch_ops = sorted(set(
        re.findall(
            r"(?:Unsupported|run in Torch|torch_executed|Not supported|"
            r"Cannot convert|Falling back)[^\n]*(aten\.[a-z0-9_\.]+)", text)
        + re.findall(r"(aten\.[a-z0-9_\.]+)\s*[:\|]\s*(?:Unsupported|False)\b", text)
    ))

    # Remove false positives: ops that appear in both lists (partition log may
    # emit "Supported" for the op name and then "Unsupported" for a variant).
    # Keep only ops that are unambiguously in one list.
    both = set(converted_ops) & set(torch_ops)
    converted_ops = [o for o in converted_ops if o not in both]
    torch_ops = [o for o in torch_ops if o not in both]
    ambiguous_ops = sorted(both)

    # --- TRT layer info from Debugger JSON files ---
    # Debugger writes per-engine files when save_layer_info=True.
    # Format: {"Layers": [{"Name": "...", "LayerType": "Activation", ...}, ...]}
    trt_engines_layer_info = {}
    layer_type_histogram = {}
    for json_path in (layer_info_files or []):
        try:
            with open(json_path) as fh:
                data = json.load(fh)
            engine_name = os.path.basename(json_path).replace(".json", "")
            raw_layers = (data.get("Layers") or data) if isinstance(data, dict) else data
            layers = []
            if isinstance(raw_layers, list):
                for layer in raw_layers:
                    lname = layer.get("Name") or layer.get("name", "")
                    ltype = layer.get("LayerType") or layer.get("type", "Unknown")
                    layers.append({"name": lname, "type": ltype})
                    layer_type_histogram[ltype] = layer_type_histogram.get(ltype, 0) + 1
            if layers:
                trt_engines_layer_info[engine_name] = layers
        except Exception as e:
            print(f"[tensorrt] could not parse layer info {json_path}: {e}")

    return {
        "backend": "tensorrt",
        "dryrun": ARGS.dryrun,
        "min_block_size": ARGS.min_block_size,
        "precision": ARGS.trt_precision,
        "num_trt_engines": len(acc_segments),
        "num_torch_fallback_segments": len(gpu_segments),
        "trt_engine_segments": acc_segments,
        "torch_fallback_segments": gpu_segments,
        "coverage_report_lines": coverage,
        "ops_converted_to_trt": converted_ops,
        "ops_left_in_torch": torch_ops,
        "ambiguous_ops": ambiguous_ops,   # appeared in both lists — worth inspecting
        "trt_layer_type_histogram": layer_type_histogram,
        "trt_engines_layer_info": trt_engines_layer_info,
        "log_files": log_paths,
        "layer_info_files": layer_info_files or [],
    }


def print_tensorrt_summary(s):
    print("\n" + "=" * 78)
    print(f"TENSORRT — {s['num_trt_engines']} TRT engine(s), "
          f"{s['num_torch_fallback_segments']} torch-fallback segment(s) "
          f"(min_block_size={s['min_block_size']}, dryrun={s['dryrun']})")
    print("=" * 78)
    for line in s["coverage_report_lines"]:
        print("  " + line.strip())

    if s.get("trt_layer_type_histogram"):
        print(f"\n  TRT kernel/layer types across all engines "
              f"(total layers: {sum(s['trt_layer_type_histogram'].values())}):")
        for lt, cnt in sorted(s["trt_layer_type_histogram"].items(),
                               key=lambda x: -x[1]):
            print(f"    {cnt:4d}  {lt}")
    else:
        print("\n  (TRT layer type info not available — run without --dryrun "
              "and ensure Debugger save_layer_info=True)")

    print(f"\n  aten ops absorbed into TRT engines ({len(s['ops_converted_to_trt'])}):")
    print(textwrap.fill(", ".join(s["ops_converted_to_trt"]) or "(none parsed)",
                        width=74, initial_indent="    ", subsequent_indent="    "))
    print(f"\n  aten ops kept in Torch ({len(s['ops_left_in_torch'])}):")
    print(textwrap.fill(", ".join(s["ops_left_in_torch"]) or "(none parsed)",
                        width=74, initial_indent="    ", subsequent_indent="    "))
    if s.get("ambiguous_ops"):
        print("\n  ambiguous (appeared in both lists — inspect logs):")
        print(textwrap.fill(", ".join(s["ambiguous_ops"]),
                            width=74, initial_indent="    ", subsequent_indent="    "))
    if s.get("layer_info_files"):
        print(f"\n  Layer info JSON(s): {s['layer_info_files']}")
    print(f"\n  Full log(s): {s['log_files']}")


# ===========================================================================
# Shared: save + compare
# ===========================================================================
def save_summary(backend, summary):
    path = os.path.join(ARGS.outdir, f"summary_{backend}.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[saved] {path}")


def compare():
    ind_p = os.path.join(ARGS.outdir, "summary_inductor.json")
    trt_p = os.path.join(ARGS.outdir, "summary_tensorrt.json")
    if not (os.path.exists(ind_p) and os.path.exists(trt_p)):
        sys.exit("Run both backends first:\n"
                 "  python compare_wan_backends.py --backend inductor\n"
                 "  python compare_wan_backends.py --backend tensorrt --dryrun")
    ind = json.load(open(ind_p))
    trt = json.load(open(trt_p))

    ind_ops = set(ind["fused_aten_ops"])
    trt_ops = set(trt["ops_converted_to_trt"])

    print("=" * 78)
    print("BACKEND COMPARISON — kernel / op replacement")
    print("=" * 78)
    print(f"Inductor: {ind['num_generated_kernels']} fused kernels generated, "
          f"covering {len(ind_ops)} distinct aten ops.")
    print(f"TensorRT: {trt['num_trt_engines']} TRT engine(s) "
          f"(+{trt['num_torch_fallback_segments']} torch fallbacks), "
          f"absorbing {len(trt_ops)} distinct aten ops.")

    print("\n-- aten ops handled by BOTH (Inductor fused them / TRT absorbed them) --")
    print(textwrap.fill(", ".join(sorted(ind_ops & trt_ops)) or "(none)",
                        width=78))
    print("\n-- only Inductor fused into a Triton/C++ kernel (TRT left in Torch) --")
    print(textwrap.fill(", ".join(sorted(ind_ops - trt_ops)) or "(none)",
                        width=78))
    print("\n-- only TRT absorbed into an engine (Inductor handled differently) --")
    print(textwrap.fill(", ".join(sorted(trt_ops - ind_ops)) or "(none)",
                        width=78))

    if trt.get("trt_layer_type_histogram"):
        print("\n-- TRT kernel/layer type breakdown (from Debugger layer info) --")
        for lt, cnt in sorted(trt["trt_layer_type_histogram"].items(),
                               key=lambda x: -x[1]):
            print(f"  {cnt:4d}  {lt}")

    print("\nNotes:")
    print("  * Inductor 'kernels' are fine-grained fusions (one Triton kernel =")
    print("    several pointwise/reduction aten ops). The kernel prefix tells you")
    print("    the type: triton_poi=pointwise, triton_red=reduction,")
    print("    triton_per=persistent-reduction, cpp_fused=CPU C++ kernel.")
    print("  * TRT 'engines' are coarse subgraphs; within each engine TRT picks")
    print("    its own kernel (ElementWise, Activation, MatrixMultiply, etc.).")
    print("    The layer type histogram above shows the TRT kernel breakdown.")
    print("  * Ops in 'only Inductor' or 'only TRT torch-fallback' are the real")
    print("    divergence points — investigate those for performance differences.")


def main():
    if ARGS.compare:
        compare()
        return
    if ARGS.backend == "inductor":
        run_inductor()
    elif ARGS.backend == "tensorrt":
        run_tensorrt()
    else:
        sys.exit("Pass --backend {inductor,tensorrt} or --compare. See --help.")


if __name__ == "__main__":
    main()
