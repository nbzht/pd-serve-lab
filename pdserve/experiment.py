"""Run a versioned experiment and publish a verifiable report bundle."""

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import re

from . import __version__
from .events import assemble_events
from .metrics import summarize_run
from .model import ROUTERS, integer, ordered_workload, parse_cluster, parse_workload
from .simulator import replay
from .vllm import import_detailed_result


ARM_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,47}")
DELTA_GATES = {"max_ttft_p95_delta_us": "ttft", "max_e2e_p95_delta_us": "e2e"}


@dataclass(frozen=True)
class InputFile:
    path: str
    content: bytes

    @classmethod
    def read(cls, base, path):
        if not isinstance(path, str) or not path.strip():
            raise ValueError("input paths must be nonempty strings")
        return cls(path, (base / path).read_bytes())

    def text(self):
        return self.content.decode("utf-8-sig")

    def provenance(self):
        return {"path": self.path, "sha256": sha256(self.content).hexdigest()}


def fields(value, required, optional, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(f"{label} fields: missing={sorted(missing)}, unknown={sorted(unknown)}")


def read_manifest(source):
    spec = json.loads(source.text())
    fields(spec, ("schema_version", "workload", "baseline", "arms", "slo"), ("gates",), "experiment")
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("experiment schema_version must be 1")
    fields(spec["slo"], ("ttft_us", "tpot_us"), (), "slo")
    for key, value in spec["slo"].items():
        integer(value, key)
    gates = spec.get("gates", {})
    fields(gates, (), (*DELTA_GATES, "min_slo_attainment"), "gates")
    for key in DELTA_GATES.keys() & gates.keys():
        integer(gates[key], key)
    if "min_slo_attainment" in gates:
        value = gates["min_slo_attainment"]
        if type(value) not in (int, float) or not 0 <= value <= 1:
            raise ValueError("min_slo_attainment must be a number from 0 to 1")
    arms = spec["arms"]
    if not isinstance(arms, list) or len(arms) < 2:
        raise ValueError("an experiment needs at least two arms")
    names = set()
    for arm in arms:
        fields(arm, ("name",), ("cluster", "router", "events", "vllm_result"), "arm")
        name = arm["name"]
        if not isinstance(name, str) or not ARM_NAME.fullmatch(name) or name in names:
            raise ValueError("arm names must be unique and contain 1–48 lowercase letters, digits or hyphens")
        names.add(name)
        keys = arm.keys() - {"name"}
        if keys == {"cluster", "router"}:
            if arm["router"] not in ROUTERS:
                raise ValueError(f"invalid router for arm {name!r}")
        elif keys not in ({"events"}, {"vllm_result"}):
            raise ValueError(f"arm {name!r} needs cluster + router, events, or vllm_result")
    if not isinstance(spec["baseline"], str) or spec["baseline"] not in names:
        raise ValueError("baseline must name an arm")
    return spec


def run_arm(base, arm, workload, slo):
    if "cluster" in arm:
        source = InputFile.read(base, arm["cluster"])
        cluster = parse_cluster(source.text(), source.path)
        rows = replay(cluster, workload, arm["router"])
        provenance = {"kind": "simulation", "router": arm["router"], "window_basis": "request span"}
        duration = None
    elif "events" in arm:
        source = InputFile.read(base, arm["events"])
        rows = assemble_events(source.text(), source.path, workload)
        provenance = {"kind": "pd-events", "window_basis": "request span"}
        duration = None
    else:
        source = InputFile.read(base, arm["vllm_result"])
        rows, duration, capture = import_detailed_result(source.text(), workload)
        provenance = {"kind": "vllm-bench", "window_basis": "vLLM benchmark duration", "capture": capture}
    return rows, summarize_run(rows, slo, duration), provenance | source.provenance()


def compare(baseline, candidate, gates):
    delta = {}
    for name in ("ttft", "tpot", "e2e"):
        earlier = baseline["latency_us"][name]["p95"]
        later = candidate["latency_us"][name]["p95"]
        delta[f"{name}_p95_us"] = later - earlier if earlier is not None and later is not None else None
    delta["goodput_rps"] = candidate["goodput_rps"] - baseline["goodput_rps"]
    delta["slo_attainment"] = candidate["slo_attainment"] - baseline["slo_attainment"]
    checks = {}
    for key, metric in DELTA_GATES.items():
        if key in gates:
            observed = delta[f"{metric}_p95_us"]
            checks[key] = {"limit": gates[key], "observed": observed,
                           "passed": observed is not None and observed <= gates[key]}
    if "min_slo_attainment" in gates:
        floor = gates["min_slo_attainment"]
        measured = Fraction(candidate["slo_passing_requests"], candidate["requests"])
        checks["min_slo_attainment"] = {"limit": floor, "observed": float(measured),
                                       "passed": measured >= Fraction(str(floor))}
    return {"delta_from_baseline": delta, "checks": checks,
            "passed": all(check["passed"] for check in checks.values())}


def report_markdown(report):
    def display(value, scale=1):
        return "—" if value is None else f"{value / scale:.3f}"

    slo = report["slo"]
    lines = ["# PD serving experiment", "", f"Baseline `{report['baseline']}` · Gates **{report['gate_status']}**",
             f"SLO bounds are TTFT ≤ {slo['ttft_us'] / 1000:g} ms and TPOT ≤ {slo['tpot_us'] / 1000:g} ms.", "",
             "| Arm | Source | OK / total | p95 TTFT ms | p95 TPOT ms | p95 E2E ms | SLO attainment | Goodput req/s |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, arm in report["arms"].items():
        stats = arm["metrics"]
        latency = [display(stats["latency_us"][metric]["p95"], 1000) for metric in ("ttft", "tpot", "e2e")]
        lines.append(f"| {name} | {arm['input']['kind']} | {stats['successful_requests']}/{stats['requests']} | "
                     f"{' | '.join(latency)} | {stats['slo_attainment']:.1%} | {stats['goodput_rps']:.3f} |")
    lines.extend(["", "## Stage observations", "",
                  "| Arm | p95 prefill queue ms | p95 transfer queue ms | p95 decode queue ms | Cache samples | KV transfer samples |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for name, arm in report["arms"].items():
        stats = arm["metrics"]
        queues = [display(stats["queue_us"][stage]["p95"], 1000) for stage in ("prefill", "transfer", "decode")]
        lines.append(f"| {name} | {' | '.join(queues)} | {stats['cache_observations']} | {stats['kv_transfer_observations']} |")
    lines.extend(["", "## KV movement", "",
                  "| Arm | Path | Observed byte samples | Total MiB | Timing samples | p95 elapsed ms |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"])
    for name, arm in report["arms"].items():
        for path, movement in arm["metrics"]["kv_movement"].items():
            elapsed = movement["elapsed_us"]
            lines.append(f"| {name} | {path} | {movement['byte_samples']} | {display(movement['total_bytes'], 2**20)} | "
                         f"{elapsed['count']} | {display(elapsed['p95'], 1000)} |")
    if report["gate_status"] != "not-configured":
        lines.extend(["", "## Regression checks", "",
                      "| Candidate | Check | Observed | Limit | Result |", "| --- | --- | ---: | ---: | --- |"])
        for name, result in report["comparisons"].items():
            for check, outcome in result["checks"].items():
                verdict = "pass" if outcome["passed"] else "fail"
                lines.append(f"| {name} | `{check}` | {display(outcome['observed'])} | "
                             f"{display(outcome['limit'])} | {verdict} |")
    lines.extend(["", "Window definitions and stage sample counts are recorded in `comparison.json`.", ""])
    return "\n".join(lines)


def write_experiment(manifest, output_dir):
    source = InputFile.read(manifest.parent, manifest.name)
    spec = read_manifest(source)
    workload_source = InputFile.read(manifest.parent, spec["workload"])
    workload = ordered_workload(parse_workload(workload_source.text(), workload_source.path))
    arms = {}
    trace_files = {}
    for arm in spec["arms"]:
        name = arm["name"]
        try:
            rows, stats, provenance = run_arm(manifest.parent, arm, workload, spec["slo"])
        except (OSError, ValueError) as exc:
            raise ValueError(f"arm {name}: {exc}") from exc
        trace_name = f"trace-{name}.jsonl"
        trace_files[trace_name] = "".join(json.dumps(row) + "\n" for row in rows)
        arms[name] = {"input": provenance, "trace": trace_name, "metrics": stats}

    baseline = spec["baseline"]
    gates = spec.get("gates", {})
    comparisons = {name: compare(arms[baseline]["metrics"], arm["metrics"], gates)
                   for name, arm in arms.items() if name != baseline}
    status = "pass" if all(result["passed"] for result in comparisons.values()) else "fail"
    report = {"schema_version": 1, "baseline": baseline, "slo": spec["slo"], "arms": arms,
              "comparisons": comparisons, "gate_status": status if gates else "not-configured"}
    bundle = trace_files | {"comparison.json": json.dumps(report, indent=2) + "\n",
                            "report.md": report_markdown(report)}
    provenance = {"schema_version": 1, "tool_version": __version__, "manifest": source.provenance(),
                  "workload": workload_source.provenance(),
                  "artifacts": {name: sha256(text.encode("utf-8")).hexdigest() for name, text in bundle.items()}}
    output_dir.mkdir(parents=True)
    for name, text in bundle.items():
        (output_dir / name).write_bytes(text.encode("utf-8"))
    (output_dir / "run.json").write_bytes((json.dumps(provenance, indent=2) + "\n").encode("utf-8"))
    return report
