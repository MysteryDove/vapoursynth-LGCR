#!/usr/bin/env python3
"""Run the independent-process throughput matrix and annotate JSONL output."""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile


def run_case(args, plugin, namespace, width, height, algo, threads, cpu_list, repeat):
    command = [
        args.python, os.path.join(args.root, "test", "benchmark.py"),
        "--mode", "throughput", "--preset", "full",
        "--width", str(width), "--height", str(height),
        "--filter", f"/{algo}/", "--threads", str(threads),
        "--cpu-list", cpu_list, "--inflight", str(2 * threads),
        "--duration", str(args.duration), "--warmup", str(args.warmup),
        "--source-mode", "static", "--namespace", namespace,
    ]
    if args.no_profile:
        command.append("--no-profile")
    environment = os.environ.copy()
    environment["LGCR_PLUGIN"] = plugin
    environment.pop("LGCR_PLUGIN_SCALAR", None)
    result = subprocess.run(command, cwd=args.root, env=environment,
                            check=True, text=True, capture_output=True)
    records = []
    for line in result.stdout.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        record["variant"] = namespace
        record["repeat"] = repeat
        records.append(record)
    if not any(record.get("record_type") == "summary" for record in records):
        raise RuntimeError(f"benchmark produced no summary for {command!r}\n{result.stdout}")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-plugin", required=True)
    parser.add_argument("--candidate-plugin", default="liblgcr.so")
    parser.add_argument("--baseline-namespace", default="lgcr_baseline")
    parser.add_argument("--candidate-namespace", default="lgcr")
    parser.add_argument("--output", default="-")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--width", type=int, action="append")
    parser.add_argument("--height", type=int, action="append")
    parser.add_argument("--threads", type=int, action="append")
    parser.add_argument("--cpu-list", action="append")
    args = parser.parse_args()
    args.threads = args.threads or [1, 8, 16]
    args.cpu_list = args.cpu_list or ["0", "0-7", "0-15"]
    if args.runs < 1 or len(args.threads) != len(args.cpu_list):
        parser.error("--runs must be positive and --threads/--cpu-list lengths must match")
    if ((args.width is None) != (args.height is None) or
            (args.width is not None and len(args.width) != len(args.height))):
        parser.error("--width and --height must be supplied in matching pairs")
    args.root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sizes = list(zip(args.width or [1920, 3840], args.height or [1080, 2160]))
    output = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8")
    try:
        metadata = {"record_type": "metadata", "schema": 3, "mode": "throughput",
                    "runs": args.runs, "sizes": sizes, "threads": args.threads,
                    "cpu_list": args.cpu_list}
        print(json.dumps(metadata, sort_keys=True), file=output)
        single_thread_fps = {}
        summaries = {}
        for repeat in range(args.runs):
            variants = [(args.baseline_namespace, args.baseline_plugin),
                        (args.candidate_namespace, args.candidate_plugin)]
            if repeat & 1:
                variants.reverse()
            for width, height in sizes:
                for algo, (threads, cpu_list) in __import__("itertools").product(
                        ("2", "4", "6"), zip(args.threads, args.cpu_list)):
                    for namespace, plugin in variants:
                        records = run_case(args, plugin, namespace, width, height,
                                           algo, threads, cpu_list, repeat)
                        for record in records:
                            if record.get("record_type") == "summary":
                                key = (repeat, namespace, record["case"])
                                summaries.setdefault(
                                    (namespace, record["case"], threads), []).append(record)
                                if threads == 1:
                                    single_thread_fps[key] = float(record["fps"])
                                elif key in single_thread_fps:
                                    speedup = float(record["fps"]) / single_thread_fps[key]
                                    record["single_thread_fps"] = single_thread_fps[key]
                                    record["speedup"] = speedup
                                    record["scaling_efficiency"] = speedup / threads
                            print(json.dumps(record, sort_keys=True), file=output)
                            output.flush()

        aggregates = {}
        for (namespace, case, threads), records in summaries.items():
            fps = [float(record["fps"]) for record in records]
            rss = [int(record["peak_rss_kib"]) for record in records
                   if record.get("peak_rss_kib") is not None]
            latency50 = [float(record["completion_latency_p50_ms"])
                         for record in records]
            latency95 = [float(record["completion_latency_p95_ms"])
                         for record in records]
            aggregate = {
                "record_type": "aggregate",
                "variant": namespace,
                "case": case,
                "threads": threads,
                "runs": len(records),
                "fps_median": statistics.median(fps),
                "fps_min": min(fps),
                "fps_max": max(fps),
                "completion_latency_p50_median_ms": statistics.median(latency50),
                "completion_latency_p95_median_ms": statistics.median(latency95),
                "peak_rss_median_kib": statistics.median(rss) if rss else None,
            }
            aggregates[(namespace, case, threads)] = aggregate

        for (namespace, case, threads), aggregate in aggregates.items():
            one = aggregates.get((namespace, case, 1))
            if one:
                speedup = aggregate["fps_median"] / one["fps_median"]
                aggregate["single_thread_fps_median"] = one["fps_median"]
                aggregate["speedup"] = speedup
                aggregate["scaling_efficiency"] = speedup / threads

            other_namespace = (args.candidate_namespace
                               if namespace == args.baseline_namespace
                               else args.baseline_namespace)
            other = aggregates.get((other_namespace, case, threads))
            if other:
                if namespace == args.candidate_namespace:
                    aggregate["baseline_fps_median"] = other["fps_median"]
                    aggregate["candidate_speedup"] = (
                        aggregate["fps_median"] / other["fps_median"])
                    if (aggregate["peak_rss_median_kib"] is not None and
                            other["peak_rss_median_kib"]):
                        aggregate["peak_rss_ratio"] = (
                            aggregate["peak_rss_median_kib"] /
                            other["peak_rss_median_kib"])
                    if "scaling_efficiency" in aggregate and "scaling_efficiency" in other:
                        aggregate["scaling_efficiency_delta"] = (
                            aggregate["scaling_efficiency"] -
                            other["scaling_efficiency"])
            print(json.dumps(aggregate, sort_keys=True), file=output)
            output.flush()
    finally:
        if output is not sys.stdout:
            output.close()


if __name__ == "__main__":
    main()
