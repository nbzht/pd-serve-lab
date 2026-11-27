"""Run a PD-serving experiment and export a Trace SLO-compatible timeline."""

import argparse
import json
from pathlib import Path

from .model import ROUTERS, load_cluster, load_workload
from .simulator import replay, summarize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
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


if __name__ == "__main__":
    main()
