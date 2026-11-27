"""Run a PD-serving experiment and export a Trace SLO-compatible timeline."""

import argparse
import json
from pathlib import Path
import sys

from .experiment import write_experiment
from .model import ROUTERS, load_cluster, load_workload
from .simulator import replay, summarize
from .vllm import timed_trace


def simulate(argv):
    parser = argparse.ArgumentParser(description=__doc__, epilog="Use 'experiment' to compare runs, or "
                                     "'vllm-export' to prepare an upstream load test.")
    parser.add_argument("cluster", type=Path, help="cluster parameters as JSON")
    parser.add_argument("workload", type=Path, help="request arrivals and token counts as JSONL")
    parser.add_argument("--router", choices=ROUTERS, default="cache-aware")
    parser.add_argument("--trace", type=Path, required=True, help="new output path for completed requests")
    args = parser.parse_args(argv)

    try:
        cluster = load_cluster(args.cluster)
        rows = replay(cluster, load_workload(args.workload), args.router)
        with args.trace.open("x", encoding="utf-8", newline="\n") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summarize(rows, cluster, args.router), indent=2))
    return 0


def experiment(argv):
    parser = argparse.ArgumentParser(prog="python -m pdserve experiment",
                                     description="Run an experiment and check candidate regressions.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True, help="new report bundle directory")
    args = parser.parse_args(argv)
    try:
        report = write_experiment(args.manifest, args.out_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{args.out_dir / 'report.md'} (gates: {report['gate_status']})")
    return int(report["gate_status"] == "fail")


def vllm_export(argv):
    parser = argparse.ArgumentParser(prog="python -m pdserve vllm-export",
                                     description="Export a workload for vLLM bench serve --dataset-name timed_trace.")
    parser.add_argument("workload", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-tokens", type=int, default=16)
    args = parser.parse_args(argv)
    try:
        entries = list(timed_trace(load_workload(args.workload), args.chunk_tokens))
        with args.output.open("x", encoding="utf-8", newline="\n") as output:
            for row in entries:
                output.write(json.dumps(row) + "\n")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{args.output} ({len(entries)} requests, {args.chunk_tokens} tokens per chunk, timestamps in us)")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "experiment":
        return experiment(argv[1:])
    if argv and argv[0] == "vllm-export":
        return vllm_export(argv[1:])
    return simulate(argv)


if __name__ == "__main__":
    raise SystemExit(main())
