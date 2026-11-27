# Contributing to PD Serve Lab

Start with the [benchmarking guide](docs/benchmarking.md) and [roadmap](ROADMAP.md). A useful first contribution is a small trace fixture that exposes a scheduling, cache or measurement edge case. Include expected timestamps, token counts and the smallest command that reproduces the behavior.

Before opening a pull request, run the suite and both reference experiments from the repository root.

```sh
python -m unittest discover -s tests -v
python -m pdserve experiment examples/routing-experiment.json --out-dir runs/check-routing
python -m pdserve experiment examples/interop/experiment.json --out-dir runs/check-interop
```

Use fresh output directories on reruns. When extending an engine adapter, name the upstream release, describe the clock and token-count contract and add a fixture for any new field. When changing a routing policy, keep the workload fixed and report the stage queue, TTFT, TPOT and goodput effects. Include a result where the policy trade-off reverses when practical. For live measurements, record deploy configuration and hardware alongside the result and share a sanitized fixture.
