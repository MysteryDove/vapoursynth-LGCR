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

    @property
    def key(self):
        suffix = ",".join(f"{key}={value}" for key, value in self.options)
        return (f"{self.width}x{self.height}/{self.subsampling}/{self.kernel}/"
                f"{self.algo}/sparse={int(self.sparse)}/{suffix}")

    def kwargs(self):
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


def make_source(core, case, length):
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
                ("cedge", (0, 1))):
            for value in values:
                cases.append(Case(width, height, 1, 1, "420", "lanczos3", "2", True,
                                  ((name, value),)))
        for value in (0.0, 1.0):
            cases.append(Case(width, height, 1, 1, "420", "lanczos3", "6", True,
                              (("qgate", value),)))
    return cases


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("smoke", "standard", "full"),
                        default="smoke")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--cpu", type=int,
                        help="logical CPU to pin to (default: first available)")
    parser.add_argument("--output", default="-")
    parser.add_argument("--baseline", help="prior JSONL file to compare")
    parser.add_argument("--compare-scalar", action="store_true")
    parser.add_argument("--no-profile", action="store_true",
                        help="measure production wall time without LGCR instrumentation")
    parser.add_argument("--filter", default="", help="substring matched against case id")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    cases = [case for case in matrix_for(args) if args.filter in case.key]
    if args.list:
        for case in cases:
            print(case.key)
        return
    if not cases:
        parser.error("no benchmark cases matched")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    selected_cpu = args.cpu if args.cpu is not None else (affinity[0] if affinity else None)
    if selected_cpu is not None and hasattr(os, "sched_setaffinity"):
        if affinity and selected_cpu not in affinity:
            parser.error(f"CPU {selected_cpu} is outside available affinity {affinity}")
        os.sched_setaffinity(0, {selected_cpu})
    if not args.no_profile:
        os.environ["LGCR_PROFILE"] = "1"
    else:
        os.environ.pop("LGCR_PROFILE", None)
    core = vs.core
    core.num_threads = 1
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
        "preset": args.preset,
        "iterations": args.iterations,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "vapoursynth": str(vs.__version__),
        "profile": not args.no_profile,
        "cpu": selected_cpu,
        "cpu_model": cpu_model(),
        "compiler": first_line([os.environ.get("CXX", "g++"), "--version"]),
        "build_config": build_config(root),
    }
    print(json.dumps(metadata, sort_keys=True), file=output, flush=True)

    try:
        for case in cases:
            source = make_source(core, case, args.iterations + 2)
            kwargs = case.kwargs()
            node = core.lgcr.Recon(source, **kwargs)
            scalar = core.lgcr_scalar.Recon(source, **kwargs) if scalar_enabled else None

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
