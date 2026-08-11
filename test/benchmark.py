#!/usr/bin/env python3
"""Repeatable end-to-end LGCR performance and correctness baseline.

The full preset spans the matrix described in the performance plan. Every
measured request uses a distinct frame number so VapourSynth's frame cache
cannot turn a hot measurement into a cache lookup. JSONL contains one record
per frame plus one summary per case.
"""
import argparse
import hashlib
import json
import os
import platform
import queue
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np
import vapoursynth as vs


STAGES = (
    "input_conversion",
    "luma_resample",
    "buildGuideMaps",
    "buildTrustMask",
    "buildMutualGate",
    "plainChroma",
    "reconstructChroma",
    "buildLGF",
    "selectorBlend",
    "buildAffineMaps",
    "buildDetail",
    "detailTransfer",
    "collaborative_chroma",
    "downsample_base",
    "downsample_guide",
    "downsample_candidate_score",
    "downsample_output",
    "backProject",
    "output_conversion",
)

CPU_SLOTS = (
    "guide_sobel", "guide_tensor", "guide_lcMap",
    "affine_candidate_box", "affine_candidate_bilinear",
    "affine_candidate_bicubic", "affine_minmax_u", "affine_minmax_v",
    "affine_window_moments", "affine_finalize_median",
    "guided_active_rows", "guided_metadata", "guided_tap_accumulation",
    "guided_normalization_selector",
    "trust_seed", "trust_dilate", "mutual_gradients", "mutual_gate",
    "plain_horizontal", "plain_vertical", "lgf_moments", "lgf_finalize",
    "affine_rolling", "affine_consume", "detail_reconstruct",
    "detail_nyquist", "detail_transfer",
)

KERNELS = (
    ("bilinear", {}),
    ("bicubic", {}),
    ("spline16", {}),
    ("spline36", {}),
    ("lanczos3", {"kernel": "lanczos", "taps": 3}),
    ("lanczos4", {"kernel": "lanczos", "taps": 4}),
    ("jinc3", {"kernel": "jinc", "taps": 3}),
)

SUBSAMPLING = ((1, 1, "420"), (1, 0, "422"), (0, 0, "444"))


@dataclass(frozen=True)
class Case:
    width: int
    height: int
    subsampling_w: int
    subsampling_h: int
    subsampling: str
    kernel: str
    algo: str
    sparse: bool
    options: tuple = ()
    operation: str = "Recon"

    @property
    def key(self):
        suffix = ",".join(f"{key}={value}" for key, value in self.options)
        operation = "" if self.operation == "Recon" else f"{self.operation}/"
        return (f"{self.width}x{self.height}/{self.subsampling}/{operation}{self.kernel}/"
                f"{self.algo}/sparse={int(self.sparse)}/{suffix}")

    def kwargs(self):
        if self.operation == "Downsample":
            return dict(self.options)
        result = dict(next(values for name, values in KERNELS if name == self.kernel))
        result.setdefault("kernel", self.kernel)
        result["strength"] = 0.0 if self.algo == "plain" else 0.8
        result["sparse"] = int(self.sparse)
        if self.algo != "plain":
            result["algo"] = int(self.algo)
        result.update(self.options)
        return result


def checksum(frame):
    digest = hashlib.sha256()
    for plane in range(3):
        digest.update(np.asarray(frame[plane]).tobytes())
    return digest.hexdigest()[:16]


def memory_kib(field):
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def reset_peak_rss():
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as handle:
            handle.write("5\n")
    except OSError:
        pass


def make_source(core, case, length, static_cache=False):
    width, height = case.width, case.height
    fmt444 = core.query_video_format(vs.YUV, vs.FLOAT, 32, 0, 0)
    fmt = core.query_video_format(
        vs.YUV, vs.FLOAT, 32, case.subsampling_w, case.subsampling_h)
    blank = core.std.BlankClip(
        width=width, height=height, format=fmt444, length=length)
    yy, xx = np.mgrid[0:height, 0:width]
    edge = (xx > width // 2).astype(np.float32)
    diagonal = (xx * height > yy * width).astype(np.float32)
    texture = (0.01 * np.sin(xx * 0.071) * np.cos(yy * 0.053)).astype(np.float32)
    planes = (
        (0.2 + 0.25 * edge + 0.03 * diagonal + texture).astype(np.float32),
        (-0.25 + 0.55 * edge + 0.5 * texture).astype(np.float32),
        (0.30 - 0.60 * edge - 0.5 * texture).astype(np.float32),
    )

    def fill(n, f):
        result = f.copy()
        # A tiny deterministic frame term keeps every request distinct while
        # remaining far below the edge amplitudes used by the workload.
        delta = np.float32((n % 17) * 1e-7)
        for plane in range(3):
            np.copyto(np.asarray(result[plane]), planes[plane] + delta)
        return result

    source = blank.std.ModifyFrame(blank, fill)
    if case.subsampling_w or case.subsampling_h:
        source = source.resize.Bilinear(format=fmt)
    if static_cache:
        # Cache after format conversion so the throughput graph contains only
        # Recon and a cheap frame-number remap. Distinct downstream frame
        # numbers still prevent Recon's own cache from satisfying requests.
        # VapourSynth inserts its own cache for graph nodes; prefetch the sole
        # source frame before Loop begins remapping downstream frame numbers.
        source.get_frame(0)
        source = source.std.Loop()
    return source


def matrix_for(args):
    if args.width or args.height:
        sizes = ((args.width or 1920, args.height or 1080),)
    elif args.preset == "full":
        sizes = ((1920, 1080), (3840, 2160))
    elif args.preset == "standard":
        sizes = ((1920, 1080),)
    else:
        sizes = ((640, 360),)

    if args.preset == "smoke":
        cases = []
        width, height = sizes[0]
        for kernel, _ in KERNELS:
            cases.append(Case(width, height, 1, 1, "420", kernel, "plain", False))
        for algo in ("2", "4", "6"):
            cases.append(Case(width, height, 1, 1, "420", "lanczos3", algo, True))
        for quality in (0, 1, 2):
            cases.append(Case(
                width, height, 0, 0, "444", "spline36", f"q{quality}", False,
                (("quality", quality), ("kernel", "spline36")), "Downsample"))
        return cases

    cases = []
    for width, height in sizes:
        for sw, sh, label in SUBSAMPLING:
            for kernel, _ in KERNELS:
                cases.append(Case(width, height, sw, sh, label, kernel, "plain", False))
                for algo in ("2", "4", "6"):
                    for sparse in (False, True):
                        cases.append(Case(width, height, sw, sh, label, kernel, algo, sparse))

        # Control ablations are orthogonal to the kernel/format matrix.
        for name, values in (
                ("ms", (0.0, 1.0)),
                ("ridge", (0, 1)),
                ("cedge", (0, 1)),
                ("bm", (0, 1))):
            for value in values:
                cases.append(Case(width, height, 1, 1, "420", "lanczos3", "2", True,
                                  ((name, value),)))
        for value in (0.0, 1.0):
            cases.append(Case(width, height, 1, 1, "420", "lanczos3", "6", True,
                              (("qgate", value),)))
        for kernel in ("spline36", "lanczos3", "binomial"):
            for quality in (0, 1, 2):
                cases.append(Case(
                    width, height, 0, 0, "444", kernel, f"q{quality}", False,
                    (("quality", quality), ("kernel", kernel)), "Downsample"))
    return cases


def throughput_matrix_for(args):
    """The CPU scaling matrix intentionally excludes BM and Downsample."""
    if args.width or args.height:
        sizes = ((args.width or 1920, args.height or 1080),)
    elif args.preset == "full":
        sizes = ((1920, 1080), (3840, 2160))
    elif args.preset == "standard":
        sizes = ((1920, 1080),)
    else:
        sizes = ((640, 360),)
    return [Case(width, height, 1, 1, "420", "lanczos3", algo, True)
            for width, height in sizes for algo in ("2", "4", "6")]


def frame_profile(frame):
    profile = {}
    pixels = {}
    taps = {}
    for stage in STAGES:
        profile[stage] = float(frame.props.get(f"_LGCR_{stage}_us", 0.0))
        pixels[stage] = int(frame.props.get(f"_LGCR_{stage}_pixels", 0))
        taps[stage] = int(frame.props.get(f"_LGCR_{stage}_taps", 0))
    cpu = {slot: float(frame.props.get(f"_LGCR_cpu_{slot}_us", 0.0))
           for slot in CPU_SLOTS}
    return profile, pixels, taps, cpu


def first_line(command):
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
        return output.splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def cpu_model():
    try:
        with open("/proc/cpuinfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def build_config(root):
    path = os.path.join(root, ".lgcr-build-config")
    try:
        with open(path, encoding="ascii") as handle:
            return handle.read().strip()
    except OSError:
        return None


def max_frame_error(left, right):
    return max(float(np.max(np.abs(
        np.asarray(left[plane]).astype(np.float64) -
        np.asarray(right[plane]).astype(np.float64)))) for plane in range(3))


def load_baseline(path):
    if not path:
        return {}
    result = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("record_type") == "summary":
                result[record["case"]] = record
    return result


def parse_cpu_list(value):
    cpus = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first_text, last_text = item.split("-", 1)
            first, last = int(first_text), int(last_text)
            if first < 0 or last < first:
                raise ValueError(f"invalid CPU range {item!r}")
            cpus.update(range(first, last + 1))
        else:
            cpu = int(item)
            if cpu < 0:
                raise ValueError(f"invalid CPU {cpu}")
            cpus.add(cpu)
    if not cpus:
        raise ValueError("CPU list is empty")
    return sorted(cpus)


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def run_async_window(node, start_frame, inflight, duration, collect_profiles):
    """Keep a fixed request window full and timestamp completion callbacks."""
    completions = queue.Queue()
    active = {}
    next_frame = start_frame

    def submit(frame_number):
        submitted_ns = time.perf_counter_ns()
        future = node.get_frame_async(frame_number)
        active[future] = (frame_number, submitted_ns)
        future.add_done_callback(
            lambda done, stamp=submitted_ns: completions.put(
                (done, stamp, time.perf_counter_ns())))

    begin_ns = time.perf_counter_ns()
    deadline_ns = begin_ns + int(duration * 1e9)
    for _ in range(inflight):
        submit(next_frame)
        next_frame += 1

    samples = []

    def consume(item, refill):
        nonlocal next_frame
        future, submitted_ns, completed_ns = item
        frame_number, _ = active.pop(future)
        frame = future.result()
        if completed_ns <= deadline_ns:
            sample = {
                "frame": frame_number,
                "latency_ms": (completed_ns - submitted_ns) / 1e6,
            }
            if collect_profiles:
                stages, _, _, cpu_profile = frame_profile(frame)
                sample["stage_us"] = stages
                sample["cpu_profile_us"] = cpu_profile
                sample["plugin_total_ms"] = float(
                    frame.props.get("_LGCR_total_us", 0.0)) / 1000.0
            samples.append(sample)
        if refill and time.perf_counter_ns() < deadline_ns:
            submit(next_frame)
            next_frame += 1

    while time.perf_counter_ns() < deadline_ns:
        timeout = max(0.0, (deadline_ns - time.perf_counter_ns()) / 1e9)
        try:
            item = completions.get(timeout=timeout)
        except queue.Empty:
            break
        consume(item, True)

    # Count callbacks which completed inside the formal window but were queued
    # while Python was processing another result. Do not refill after deadline.
    while True:
        try:
            item = completions.get_nowait()
        except queue.Empty:
            break
        if item[0] in active:
            consume(item, False)

    # VapourSynth futures are not reliably cancellable once scheduled. Drain
    # the tail so one case cannot leak work into the next case's measurement.
    for future in list(active):
        future.result()
    elapsed_s = (deadline_ns - begin_ns) / 1e9
    return samples, next_frame, elapsed_s


def run_throughput_case(node, case, args):
    next_frame = 0
    _, next_frame, _ = run_async_window(
        node, next_frame, args.inflight, args.warmup, False)
    reset_peak_rss()
    samples, _, elapsed_s = run_async_window(
        node, next_frame, args.inflight, args.duration, not args.no_profile)
    if not samples:
        raise RuntimeError(f"no frames completed for {case.key}")

    latencies = [sample["latency_ms"] for sample in samples]
    stage_samples = {stage: [] for stage in STAGES}
    cpu_samples = {slot: [] for slot in CPU_SLOTS}
    plugin_total_samples = []
    for sample in samples:
        for stage, value in sample.get("stage_us", {}).items():
            stage_samples[stage].append(value)
        for slot, value in sample.get("cpu_profile_us", {}).items():
            cpu_samples[slot].append(value)
        if "plugin_total_ms" in sample:
            plugin_total_samples.append(sample["plugin_total_ms"])
    return {
        "record_type": "summary",
        "case": case.key,
        "operation": case.operation,
        "source_precached": True,
        "mode": "throughput",
        "threads": args.threads,
        "inflight": args.inflight,
        "duration_s": elapsed_s,
        "frames": len(samples),
        "fps": len(samples) / elapsed_s,
        # Retain latency summary names used by existing JSONL consumers.
        "median_ms": percentile(latencies, 50),
        "min_ms": min(latencies),
        "completion_latency_p50_ms": percentile(latencies, 50),
        "completion_latency_p95_ms": percentile(latencies, 95),
        "plugin_total_median_ms": percentile(plugin_total_samples, 50),
        "stage_median_us": {
            stage: percentile(values, 50) if values else 0.0
            for stage, values in stage_samples.items()
        },
        "cpu_profile_median_us": {
            slot: percentile(values, 50) if values else 0.0
            for slot, values in cpu_samples.items()
        },
        "checksums": [],
        "scalar_max_error": None,
        "peak_rss_kib": memory_kib("VmHWM"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("latency", "throughput"),
                        default="latency")
    parser.add_argument("--preset", choices=("smoke", "standard", "full"),
                        default="smoke")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--cpu", type=int,
                        help="logical CPU to pin to (default: first available)")
    parser.add_argument("--cpu-list",
                        help="CPU affinity list/ranges, for example 0-7,16")
    parser.add_argument("--threads", type=int, default=1,
                        help="VapourSynth worker count")
    parser.add_argument("--inflight", type=int,
                        help="async requests kept in flight (default: 2 x threads)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="throughput measurement duration in seconds")
    parser.add_argument("--warmup", type=float, default=2.0,
                        help="throughput warm-up duration in seconds")
    parser.add_argument("--source-mode", choices=("auto", "dynamic", "static"),
                        default="auto")
    parser.add_argument("--namespace", default="lgcr",
                        help="loaded VapourSynth plugin namespace")
    parser.add_argument("--output", default="-")
    parser.add_argument("--baseline", help="prior JSONL file to compare")
    parser.add_argument("--compare-scalar", action="store_true")
    parser.add_argument("--no-profile", action="store_true",
                        help="measure production wall time without LGCR instrumentation")
    parser.add_argument("--filter", default="", help="substring matched against case id")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.inflight is None:
        args.inflight = 2 * args.threads
    if args.inflight < 1:
        parser.error("--inflight must be positive")
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.warmup <= 0.0:
        parser.error("--warmup must be positive")
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    matrix = throughput_matrix_for(args) if args.mode == "throughput" else matrix_for(args)
    cases = [case for case in matrix if args.filter in case.key]
    if args.list:
        for case in cases:
            print(case.key)
        return
    if not cases:
        parser.error("no benchmark cases matched")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    if args.cpu is not None and args.cpu_list:
        parser.error("--cpu and --cpu-list are mutually exclusive")
    try:
        selected_cpus = (parse_cpu_list(args.cpu_list) if args.cpu_list else
                         [args.cpu] if args.cpu is not None else
                         affinity[:args.threads])
    except ValueError as error:
        parser.error(str(error))
    if selected_cpus and hasattr(os, "sched_setaffinity"):
        unavailable = sorted(set(selected_cpus) - set(affinity)) if affinity else []
        if unavailable:
            parser.error(f"CPUs {unavailable} are outside available affinity {affinity}")
        if len(selected_cpus) < args.threads:
            parser.error("CPU list must contain at least --threads CPUs")
        os.sched_setaffinity(0, set(selected_cpus))
    if not args.no_profile:
        os.environ["LGCR_PROFILE"] = "1"
    else:
        os.environ.pop("LGCR_PROFILE", None)
    core = vs.core
    core.num_threads = args.threads
    core.std.LoadPlugin(os.environ.get(
        "LGCR_PLUGIN", os.path.join(root, "liblgcr.so")))
    scalar_enabled = args.compare_scalar
    if scalar_enabled:
        core.std.LoadPlugin(os.environ.get(
            "LGCR_PLUGIN_SCALAR", os.path.join(root, "liblgcr_scalar.so")))

    baseline = load_baseline(args.baseline)
    output = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    metadata = {
        "record_type": "metadata",
        "schema": 2,
        "mode": args.mode,
        "preset": args.preset,
        "iterations": args.iterations,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "vapoursynth": str(vs.__version__),
        "profile": not args.no_profile,
        "cpu": selected_cpus[0] if len(selected_cpus) == 1 else None,
        "cpu_list": selected_cpus,
        "threads": args.threads,
        "inflight": args.inflight if args.mode == "throughput" else None,
        "duration_s": args.duration if args.mode == "throughput" else None,
        "warmup_s": args.warmup if args.mode == "throughput" else None,
        "cpu_model": cpu_model(),
        "compiler": first_line([os.environ.get("CXX", "g++"), "--version"]),
        "build_config": build_config(root),
    }
    print(json.dumps(metadata, sort_keys=True), file=output, flush=True)

    try:
        for case in cases:
            static_source = (args.source_mode == "static" or
                             (args.source_mode == "auto" and args.mode == "throughput"))
            source_length = 1 if static_source else args.iterations + 2
            source = make_source(core, case, source_length, static_source)
            kwargs = case.kwargs()
            plugin = getattr(core, args.namespace)
            node = getattr(plugin, case.operation)(source, **kwargs)
            scalar = (getattr(core.lgcr_scalar, case.operation)(source, **kwargs)
                      if scalar_enabled else None)

            if args.mode == "throughput":
                if not static_source:
                    parser.error("throughput mode requires --source-mode auto or static")
                if scalar is not None:
                    parser.error("--compare-scalar is only supported in latency mode")
                summary = run_throughput_case(node, case, args)
                prior = baseline.get(case.key)
                if prior and prior.get("fps"):
                    reference_threads = int(prior.get("threads", 1))
                    speedup = summary["fps"] / float(prior["fps"])
                    summary["baseline_fps"] = float(prior["fps"])
                    summary["baseline_threads"] = reference_threads
                    summary["speedup"] = speedup
                    summary["scaling_efficiency"] = (
                        speedup / (args.threads / reference_threads))
                print(json.dumps(summary, sort_keys=True), file=output, flush=True)
                continue

            # Warm geometry, LUTs, code paths, and the frame allocator twice.
            node.get_frame(0)
            node.get_frame(1)
            if scalar is not None:
                scalar.get_frame(0)
                scalar.get_frame(1)
            reset_peak_rss()

            wall_samples = []
            plugin_total_samples = []
            stage_samples = {stage: [] for stage in STAGES}
            cpu_samples = {slot: [] for slot in CPU_SLOTS}
            scalar_errors = []
            checksums = []
            for frame_number in range(2, args.iterations + 2):
                # Downsample has explicit per-filter millisecond gates. Keep its
                # distinct input frame resident so the measurement excludes
                # this benchmark's NumPy ModifyFrame source construction.
                source_precached = case.operation == "Downsample"
                if source_precached:
                    source.get_frame(frame_number)
                rss_before = memory_kib("VmRSS")
                start = time.perf_counter_ns()
                frame = node.get_frame(frame_number)
                wall_ns = time.perf_counter_ns() - start
                rss_after = memory_kib("VmRSS")
                stages, stage_pixels, stage_taps, cpu_profile = frame_profile(frame)
                plugin_total_ms = float(frame.props.get("_LGCR_total_us", 0.0)) / 1000.0
                digest = checksum(frame)
                error = None
                if scalar is not None:
                    scalar_frame = scalar.get_frame(frame_number)
                    error = max_frame_error(frame, scalar_frame)
                    scalar_errors.append(error)
                wall_ms = wall_ns / 1e6
                wall_samples.append(wall_ms)
                plugin_total_samples.append(plugin_total_ms)
                checksums.append(digest)
                for stage, value in stages.items():
                    stage_samples[stage].append(value)
                for slot, value in cpu_profile.items():
                    cpu_samples[slot].append(value)
                record = {
                    "record_type": "frame",
                    "case": case.key,
                    "frame": frame_number,
                    "config": kwargs,
                    "operation": case.operation,
                    "source_precached": source_precached,
                    "width": case.width,
                    "height": case.height,
                    "subsampling": case.subsampling,
                    "kernel": case.kernel,
                    "algo": case.algo,
                    "sparse": case.sparse,
                    "wall_ms": wall_ms,
                    "plugin_total_ms": plugin_total_ms,
                    "stage_us": stages,
                    "stage_pixels": stage_pixels,
                    "stage_taps": stage_taps,
                    "cpu_profile_us": cpu_profile,
                    "checksum": digest,
                    "rss_kib": rss_after,
                    "rss_delta_kib": (rss_after - rss_before
                                      if rss_before is not None and rss_after is not None else None),
                    "peak_rss_kib": memory_kib("VmHWM"),
                    "output_pixels": int(frame.props.get("_LGCR_output_pixels", 0)),
                    "taps_visited": int(frame.props.get("_LGCR_taps_visited", 0)),
                    "sparse_active_ratio": float(
                        frame.props.get("_LGCR_sparse_active_ratio", 1.0)),
                    "scalar_max_error": error,
                }
                print(json.dumps(record, sort_keys=True), file=output, flush=True)

            summary = {
                "record_type": "summary",
                "case": case.key,
                "operation": case.operation,
                "source_precached": case.operation == "Downsample",
                "frames": args.iterations,
                "median_ms": float(np.median(wall_samples)),
                "min_ms": min(wall_samples),
                "plugin_total_median_ms": (
                    float(np.median(plugin_total_samples)) if not args.no_profile else None),
                "stage_median_us": {
                    stage: float(np.median(values)) for stage, values in stage_samples.items()
                },
                "cpu_profile_median_us": {
                    slot: float(np.median(values)) for slot, values in cpu_samples.items()
                },
                "checksums": checksums,
                "scalar_max_error": max(scalar_errors) if scalar_errors else None,
                "peak_rss_kib": memory_kib("VmHWM"),
            }
            prior = baseline.get(case.key)
            if prior:
                old = float(prior["median_ms"])
                summary["baseline_ms"] = old
                summary["speedup"] = old / summary["median_ms"]
                prior_stages = prior.get("stage_median_us", {})
                summary["stage_baseline_us"] = prior_stages
                summary["stage_speedup"] = {
                    stage: float(value) / summary["stage_median_us"][stage]
                    for stage, value in prior_stages.items()
                    if stage in summary["stage_median_us"] and
                    float(value) > 0.0 and summary["stage_median_us"][stage] > 0.0
                }
                prior_checksums = prior.get("checksums", [])
                if prior_checksums:
                    common = min(len(prior_checksums), len(checksums))
                    summary["baseline_checksum_match"] = (
                        prior_checksums[:common] == checksums[:common])
                if prior.get("peak_rss_kib") is not None:
                    summary["baseline_peak_rss_kib"] = prior["peak_rss_kib"]
            print(json.dumps(summary, sort_keys=True), file=output, flush=True)
    finally:
        if output is not sys.stdout:
            output.close()


if __name__ == "__main__":
    main()
