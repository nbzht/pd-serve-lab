from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from pdserve.events import assemble_events
from pdserve.experiment import write_experiment
from pdserve.metrics import summarize_run
from pdserve.model import Request, load_workload
from pdserve.vllm import import_detailed_result, timed_trace


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
INTEROP = EXAMPLES / "interop"


def event(request_id, name, at, **data):
    return json.dumps({"request_id": request_id, "event": name, "timestamp_us": at, **data})


class ExperimentTests(unittest.TestCase):
    def test_report_and_trace_bundle_reproduce_and_verify(self):
        scenario = EXAMPLES / "routing-experiment.json"
        with TemporaryDirectory() as temp:
            path = Path(temp)
            first = path / "first"
            second = path / "second"
            report = write_experiment(scenario, first)
            self.assertEqual(report["gate_status"], "pass")
            self.assertEqual(report["comparisons"]["cache-aware"]["delta_from_baseline"]["ttft_p95_us"], 3176)
            self.assertEqual(report["arms"]["cache-aware"]["metrics"]["queue_us"]["decode"]["count"], 10)
            self.assertEqual(report, write_experiment(scenario, second))
            audit = json.loads((first / "run.json").read_bytes())
            self.assertEqual(audit["manifest"]["sha256"], sha256(scenario.read_bytes()).hexdigest())
            for name, expected in audit["artifacts"].items():
                self.assertEqual(sha256((first / name).read_bytes()).hexdigest(), expected)
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            with self.assertRaises(FileExistsError):
                write_experiment(scenario, first)

    def test_failed_gate_has_a_cli_exit_code_and_preserves_artifacts(self):
        with TemporaryDirectory() as temp:
            example_dir = Path(temp) / "examples"
            shutil.copytree(EXAMPLES, example_dir)
            manifest = example_dir / "routing-experiment.json"
            spec = json.loads(manifest.read_bytes())
            spec["gates"]["max_ttft_p95_delta_us"] = 3175
            manifest.write_text(json.dumps(spec), encoding="utf-8")
            output = Path(temp) / "results"
            command = [sys.executable, "-m", "pdserve", "experiment", str(manifest), "--out-dir", str(output)]
            run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertTrue((output / "run.json").is_file())
            report = json.loads((output / "comparison.json").read_bytes())
            check = report["comparisons"]["cache-aware"]["checks"]["max_ttft_p95_delta_us"]
            self.assertEqual(check, {"limit": 3175, "observed": 3176, "passed": False})
            spec["arms"][1]["name"] = "../escape"
            manifest.write_text(json.dumps(spec), encoding="utf-8")
            invalid_output = Path(temp) / "invalid"
            run = subprocess.run(command[:-1] + [str(invalid_output)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertFalse(invalid_output.exists())

    def test_vllm_detailed_uses_reported_duration_and_keeps_stage_samples_missing(self):
        with TemporaryDirectory() as temp:
            output = Path(temp) / "results"
            report = write_experiment(INTEROP / "experiment.json", output)
            stats = report["arms"]["candidate"]["metrics"]
            self.assertEqual(stats["observation_us"], 82000)
            self.assertAlmostEqual(stats["goodput_rps"], 2 / 0.082)
            self.assertEqual(stats["queue_us"]["prefill"]["count"], 0)
            self.assertIsNone(stats["cache_residency"])
            self.assertEqual(report["arms"]["candidate"]["input"]["capture"]["model_id"], "synthetic-format-fixture")
            self.assertEqual(report["comparisons"]["candidate"]["delta_from_baseline"]["ttft_p95_us"], -7000)

    def test_vllm_checks_lengths_and_preserves_failed_requests(self):
        workload = load_workload(INTEROP / "workload.jsonl")
        result = json.loads((INTEROP / "vllm-baseline.json").read_bytes())
        result.update(completed=1, failed=1, output_lens=[3, 0], errors=["", "timeout"], ttfts=[0.020, 0.0])
        rows, duration, _ = import_detailed_result(json.dumps(result), workload)
        self.assertEqual([row["status"] for row in rows], ["ok", "error"])
        stats = summarize_run(rows, {"ttft_us": 25_000, "tpot_us": 2000}, duration)
        self.assertEqual(stats["slo_attainment"], 0.5)
        result["input_lens"] = [64, 32]
        with self.assertRaisesRegex(ValueError, "different input length"):
            import_detailed_result(json.dumps(result), workload)

    def test_export_preserves_full_prefix_chunks_and_unique_suffixes(self):
        workload = [Request("late", 100, 5, 1, "shared", 3), Request("early", 0, 5, 4, "shared", 3)]
        first, second = timed_trace(workload, chunk_tokens=2)
        self.assertEqual((first["timestamp"], first["output_length"]), (0, 4))
        self.assertEqual(second["timestamp"], 100)
        self.assertEqual(len(first["hash_ids"]), 3)
        self.assertEqual(first["hash_ids"][:1], second["hash_ids"][:1])
        self.assertTrue(set(first["hash_ids"][1:]).isdisjoint(second["hash_ids"][1:]))

    def test_events_join_out_of_order_and_reject_conflicts_and_bad_clock_order(self):
        requests = [Request("a", 0, 8, 3)]
        arrival = event("a", "arrival", 0, prompt_tokens=8)
        complete = event("a", "complete", 40, status="ok", output_tokens=3)
        payload = [complete, arrival, event("a", "first_token", 20), arrival,
                   event("a", "transfer_end", 12, kv_transfer_bytes=256), event("a", "decode_start", 15)]
        rows = assemble_events("\n".join(payload), "events", requests)
        stats = summarize_run(rows, {"ttft_us": 20, "tpot_us": 10})
        self.assertEqual(stats["queue_us"]["decode"]["p95"], 3)
        self.assertEqual(stats["queue_us"]["prefill"]["count"], 0)
        self.assertEqual(stats["slo_passing_requests"], 1)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            assemble_events("\n".join(payload + [event("a", "arrival", 1, prompt_tokens=8)]), "events", requests)
        with self.assertRaisesRegex(ValueError, "non-monotonic"):
            assemble_events("\n".join(payload + [event("a", "prefill_ready", 25)]), "events", requests)
        with self.assertRaisesRegex(ValueError, "both arrival and complete"):
            assemble_events(arrival, "events", requests)

    def test_instrumented_arm_exports_same_trace_contract_as_simulation(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)
            shutil.copy(EXAMPLES / "cluster.json", path)
            workload = {"id": "req", "arrival_us": 0, "prompt_tokens": 8, "output_tokens": 2}
            (path / "workload.jsonl").write_text(json.dumps(workload) + "\n", encoding="utf-8")
            events = [event("req", "arrival", 10, prompt_tokens=8),
                      event("req", "prefill_start", 13, cache_tier="miss"),
                      event("req", "first_token", 60),
                      event("req", "complete", 75, status="ok", output_tokens=2)]
            (path / "events.jsonl").write_text("\n".join(events), encoding="utf-8")
            manifest = {
                "schema_version": 1, "workload": "workload.jsonl", "baseline": "model",
                "arms": [{"name": "model", "cluster": "cluster.json", "router": "round-robin"},
                         {"name": "instrumented", "events": "events.jsonl"}],
                "slo": {"ttft_us": 1000, "tpot_us": 250},
            }
            spec = path / "experiment.json"
            spec.write_text(json.dumps(manifest), encoding="utf-8")
            output = path / "results"
            report = write_experiment(spec, output)
            stats = report["arms"]["instrumented"]["metrics"]
            self.assertEqual(stats["latency_us"]["ttft"]["p95"], 50)
            self.assertEqual(stats["queue_us"]["prefill"]["p95"], 3)
            self.assertIsNone(stats["queue_us"]["transfer"]["p95"])
            row = json.loads((output / "trace-instrumented.jsonl").read_bytes())
            self.assertEqual(set(row) & {"id", "status", "start_us", "end_us", "first_token_us", "output_tokens"},
                             {"id", "status", "start_us", "end_us", "first_token_us", "output_tokens"})


if __name__ == "__main__":
    unittest.main()
