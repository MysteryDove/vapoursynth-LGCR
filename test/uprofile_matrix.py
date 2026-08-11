#!/usr/bin/env python3
"""Collect the LGCR CPU profiling matrix with AMD uProf 5.3 or newer."""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time


LAYOUTS = ((1, "0", "ccx=0"), (8, "0-7", "ccx=0"),
           (16, "0-15", "package=0"))
ALGOS = (2, 4, 6)


def benchmark_command(args, width, height, algo, threads, cpu_list, output,
                      duration=None, warmup=None, inflight=None):
    if duration is None:
        duration = args.duration
    if warmup is None:
        warmup = args.warmup
    if inflight is None:
        inflight = 2 * threads
    return [
        args.python, os.path.join(args.root, "test", "benchmark.py"),
        "--mode", "throughput", "--preset", "full",
        "--width", str(width), "--height", str(height),
        "--filter", f"/{algo}/", "--threads", str(threads),
        "--cpu-list", cpu_list, "--inflight", str(inflight),
        "--duration", str(duration), "--warmup", str(warmup),
        "--source-mode", "static", "--namespace", args.namespace,
        "--no-profile", "--output", output,
    ]


def benchmark_summary(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("record_type") == "summary":
                return record
    raise RuntimeError(f"no benchmark summary in {path}")


def pcm_metrics(path):
    wanted = {
        "IPC (Sys + User)": "ipc",
        "Eff Freq (MHz)": "effective_frequency_mhz",
        "Utilization (%)": "utilization_percent",
        "L3 Miss %": "l3_miss_percent",
        "Total Mem Bw (GB/s)": "dram_total_gbps",
        "Total Mem RdBw (GB/s)": "dram_read_gbps",
        "Total Mem WrBw (GB/s)": "dram_write_gbps",
    }
    result = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 2 or row[0] not in wanted:
                continue
            try:
                result[wanted[row[0]]] = float(row[1])
            except ValueError:
                pass
    return result


def run_pcm(args, summary_output):
    stream = {}
    for threads, cpu_list, _ in LAYOUTS:
        command = [args.stream, "--threads", str(threads),
                   "--cpu-list", cpu_list, "--duration", "2"]
        record = json.loads(subprocess.check_output(
            command, cwd=args.root, text=True))
        stream[threads] = float(record["sustainable_gbps"])
        record["record_type"] = "stream"
        print(json.dumps(record, sort_keys=True), file=summary_output, flush=True)

    for width, height in ((1920, 1080), (3840, 2160)):
        for algo in ALGOS:
            for threads, cpu_list, component in LAYOUTS:
                name = f"{width}x{height}-algo{algo}-t{threads}"
                benchmark_path = os.path.join(args.output_dir, name + ".jsonl")
                pcm_path = os.path.join(args.output_dir, name + "-pcm.csv")
                command = [
                    args.pcm, "-m", "ipc,l1,l2,l3,memory",
                    "-c", component, "-A", "system", "-C",
                    "-o", pcm_path, "--",
                    *benchmark_command(args, width, height, algo, threads,
                                       cpu_list, benchmark_path),
                ]
                complete = False
                if args.resume and os.path.isfile(pcm_path) and os.path.isfile(benchmark_path):
                    try:
                        complete = bool(pcm_metrics(pcm_path)) and bool(
                            benchmark_summary(benchmark_path))
                    except (OSError, ValueError, RuntimeError):
                        complete = False
                if not complete:
                    for attempt in range(3):
                        for path in (pcm_path, benchmark_path):
                            try:
                                os.unlink(path)
                            except FileNotFoundError:
                                pass
                        try:
                            subprocess.run(command, cwd=args.root,
                                           env=args.environment, check=True)
                            benchmark_summary(benchmark_path)
                            if not pcm_metrics(pcm_path):
                                raise RuntimeError(f"incomplete PCM report for {name}")
                            break
                        except (subprocess.CalledProcessError, RuntimeError,
                                OSError, ValueError):
                            if attempt == 2:
                                raise
                            print(f"uProf PCM retry {attempt + 2}/3 for {name}",
                                  file=sys.stderr, flush=True)
                            time.sleep(1.0)
                benchmark = benchmark_summary(benchmark_path)
                record = {
                    "record_type": "pcm",
                    "case": benchmark["case"],
                    "width": width,
                    "height": height,
                    "algo": algo,
                    "threads": threads,
                    "cpu_list": cpu_list,
                    "fps": float(benchmark["fps"]),
                    "peak_rss_kib": benchmark.get("peak_rss_kib"),
                    "stream_sustainable_gbps": stream[threads],
                    **pcm_metrics(pcm_path),
                }
                if "dram_total_gbps" in record:
                    record["bytes_per_frame"] = (
                        record["dram_total_gbps"] * 1e9 / record["fps"])
                    record["bandwidth_ratio"] = (
                        record["dram_total_gbps"] / stream[threads])
                print(json.dumps(record, sort_keys=True),
                      file=summary_output, flush=True)
                time.sleep(0.25)


def cli_cases(config):
    if config in ("hotspots", "data_access"):
        layouts = (LAYOUTS[0], LAYOUTS[2])
    else:
        layouts = (LAYOUTS[1], LAYOUTS[2])
    return [(3840, 2160, algo, layout) for algo in ALGOS for layout in layouts]


def run_cli(args, config):
    base = os.path.join(args.output_dir, config)
    os.makedirs(base, exist_ok=args.resume)
    # Hardware data-access and threading instrumentation can make each frame
    # orders of magnitude slower. Keep CLI sessions bounded; PCM remains the
    # long-window source for bandwidth and scaling metrics.
    cli_duration = args.cli_duration
    cli_warmup = min(args.warmup, 0.5)
    if config == "data_access":
        # data_access records every sampled memory operation. At 4K/16T even a
        # short unrestricted run can exhaust the host while uProf buffers data.
        cli_duration = min(cli_duration, 0.25)
        cli_warmup = min(cli_warmup, 0.1)
        print("uProf data_access: using bounded 0.25s/0.1s session and one in-flight frame",
              file=sys.stderr, flush=True)
    for width, height, algo, (threads, cpu_list, _) in cli_cases(config):
        name = f"{width}x{height}-algo{algo}-t{threads}"
        session = os.path.join(base, name)
        benchmark_path = os.path.join(base, name + ".jsonl")
        collect_options = [args.cli, "collect", "--config", config]
        if config == "hotspots":
            collect_options.extend(("--call-graph", "dwarf:2048"))
        command = [
            *collect_options, "-o", session,
            "--affinity", cpu_list, "-w", args.root,
            *benchmark_command(args, width, height, algo, threads,
                               cpu_list, benchmark_path,
                               duration=cli_duration, warmup=cli_warmup,
                               inflight=1 if config in ("data_access", "threading")
                               else None),
        ]
        report_path = os.path.join(base, name + "-report.csv")
        if not (args.resume and os.path.isfile(report_path)):
            if os.path.isdir(session):
                shutil.rmtree(session)
            subprocess.run(command, cwd=args.root, env=args.environment, check=True)
        if not os.path.isfile(report_path):
            subprocess.run([
            args.cli, "report", "-i", session, "--detail", "--cutoff", "0",
            "--report-output", report_path,
            ], cwd=args.root, env=args.environment, check=True)
        time.sleep(0.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group", action="append",
                        choices=("pcm", "hotspots", "data_access", "threading"))
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--cli-duration", type=float, default=2.0,
                        help="bounded duration for uProf CLI sessions")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--plugin", default="liblgcr.so")
    parser.add_argument("--namespace", default="lgcr")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--uprofile-bin", default="/opt/AMDuProf_5.3-521/bin")
    args = parser.parse_args()
    if args.duration <= 0.0 or args.warmup <= 0.0 or args.cli_duration <= 0.0:
        parser.error("--duration, --warmup, and --cli-duration must be positive")
    args.root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args.output_dir = os.path.abspath(args.output_dir)
    if (not args.resume and os.path.exists(args.output_dir) and
            os.listdir(args.output_dir)):
        parser.error("--output-dir must be absent or empty")
    os.makedirs(args.output_dir, exist_ok=True)
    args.pcm = os.path.join(args.uprofile_bin, "AMDuProfPcm")
    args.cli = os.path.join(args.uprofile_bin, "AMDuProfCLI")
    args.stream = os.path.join(args.root, "test", "stream")
    for executable in (args.pcm, args.cli, args.stream):
        if not os.path.isfile(executable) or not os.access(executable, os.X_OK):
            parser.error(f"required executable is unavailable: {executable}")
    args.environment = os.environ.copy()
    args.environment["LGCR_PLUGIN"] = os.path.abspath(args.plugin)
    # data_access is opt-in because its raw event stream can consume multiple
    # GiB on a 4K/16T case; PCM and the symbolized CLI groups remain default.
    groups = args.group or ["pcm", "hotspots", "threading"]
    summary_path = os.path.join(args.output_dir, "summary.jsonl")
    with open(summary_path, "w", encoding="utf-8") as summary_output:
        metadata = {
            "record_type": "metadata", "schema": 1, "groups": groups,
            "duration_s": args.duration, "warmup_s": args.warmup,
            "cli_duration_s": args.cli_duration,
            "plugin": args.environment["LGCR_PLUGIN"],
            "namespace": args.namespace,
        }
        print(json.dumps(metadata, sort_keys=True),
              file=summary_output, flush=True)
        if "pcm" in groups:
            run_pcm(args, summary_output)
        for config in ("hotspots", "data_access", "threading"):
            if config in groups:
                run_cli(args, config)


if __name__ == "__main__":
    main()
