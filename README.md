# PD Serve Lab

PD Serve Lab replays the same request arrivals through different prefill/decode deployments. Its per-request timeline shows where first-token latency comes from, including prefill queues, prefix-cache hits, host offload, KV transfer contention and decode queues. It runs on a CPU with Python 3.10 or newer and uses only the standard library.

## Reproduce the first experiment

```sh
python -m pdserve examples/cluster.json examples/prefix-bursts.jsonl --router round-robin --trace rr.trace.jsonl
python -m pdserve examples/cluster.json examples/prefix-bursts.jsonl --router least-loaded --trace load.trace.jsonl
python -m pdserve examples/cluster.json examples/prefix-bursts.jsonl --router cache-aware --trace cache.trace.jsonl
```

Use fresh trace filenames on each run. The summary goes to stdout. The trace contains one timeline per request, in arrival order. Use [Trace SLO](https://github.com/nbzht/trace-slo) for TTFT, TPOT and goodput comparisons under a fixed SLO.

```sh
python ../trace-slo/trace_slo.py cache.trace.jsonl --baseline rr.trace.jsonl --ttft-ms 50 --tpot-ms 0.25
```

| Router | GPU hits | Host hits | p95 prefill queue | p95 decode queue | p95 TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Round robin | 1 | 3 | 21.285 ms | 17.280 ms | 51.657 ms |
| Least loaded | 1 | 3 | 21.285 ms | 17.280 ms | 59.303 ms |
| Cache aware | 4 | 2 | 14.470 ms | 31.424 ms | 54.833 ms |

These results come from the bundled synthetic workload. In this scenario, cache affinity reduces prefill work and increases p95 decode queueing. Check the stage timeline before attributing a TTFT change to the router or to the cache.

## Workload and cluster inputs

Each JSONL line has an `id`, `arrival_us`, `prompt_tokens` and `output_tokens`. Supply `prefix_key` and `prefix_tokens` together for requests that share an exact prompt prefix. A key represents a fixed prefix token sequence within a run. Use a separate key when the sequence changes. Omit both fields for requests without a reusable prefix. The bundled [cluster profile](examples/cluster.json) defines worker counts, per-worker cache capacity in tokens, KV bytes per prompt token, fixed costs in microseconds, and link rates in bytes per microsecond.

- **Prefill** uses one FIFO queue per worker and linear cost for uncached tokens. The GPU and host prefix caches use worker-local LRU. Host hits pay a restore cost. GPU evictions can spill into host memory and charge the prefill worker a write cost. A host-only prefix can be reused through the worker's transient staging space.
- **KV transfer** uses one shared FIFO link. Every request ships the full prompt KV payload. A prefill cache hit reduces prefill work and leaves the transfer payload size unchanged in this model.
- **Decode** reserves one worker until the request finishes. The first output token arrives after a fixed cost. Further tokens have a fixed inter-token interval.
- **Routing** uses separate round-robin counters for prefill and decode. Least-loaded routing estimates prefill work ahead and picks the earliest available decode worker. Cache-aware routing adds the incoming request's estimated prefill cost at its current cache tier. Pending requests are scored at uncached cost, while their actual cache hit is determined at execution time.

The discrete-event model assumes serial workers, one transfer link, fixed token counts and homogeneous service rates per pool. It omits continuous batching, GPU memory traffic, network overlap and decode-side prefix reuse. Its synthetic numbers describe the configured model. See [DistServe](https://arxiv.org/abs/2401.09670) and [vLLM disaggregated prefilling](https://docs.vllm.ai/en/stable/features/disagg_prefill/) for real serving implementations and measurement context.

## Development

```sh
python -m unittest discover -s tests -v
```

The [roadmap](ROADMAP.md) defines the experiments, adapters and acceptance checks for the next milestones.
