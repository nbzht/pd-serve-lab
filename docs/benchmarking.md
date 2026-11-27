# Live benchmarking and trace contracts

PD Serve Lab uses the existing vLLM load generator. The adapter targets the [vLLM v0.34.0 detailed result arrays](https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/benchmarks/serve.py#L1317-L1337) and the [vLLM timed trace dataset](https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/benchmarks/datasets/datasets.py#L1417). Load generation and tokenizer handling stay in vLLM.

## Compare two running deployments

These commands use two existing endpoints and the same model and tokenizer. Keep the load generator version, seed, model, tokenization, warmup policy and workload fixed. The bundled trial has ten requests and is a smoke workload. Use a representative trace and multiple independent runs for an experiment you intend to publish.

```sh
mkdir -p runs/live
python -m pdserve vllm-export examples/prefix-bursts.jsonl --output runs/live/timed.jsonl
export PYTHONHASHSEED=0
MODEL=your-model-id

for arm in baseline candidate; do
  if [ "$arm" = baseline ]; then port=8000; else port=8001; fi
  vllm bench serve --backend vllm --base-url "http://127.0.0.1:$port" --model "$MODEL" \
    --dataset-name timed_trace --dataset-path runs/live/timed.jsonl --num-prompts 10 \
    --disable-shuffle --no-oversample --self-timed --seed 0 --ignore-eos \
    --timed-trace-chunk-hash-size 16 --timed-trace-sec-multiplier 0.000001 \
    --save-result --save-detailed --result-dir runs/live --result-filename "$arm.json"
done
```

`vllm-export` sorts by arrival and preserves file order for ties. It maps reusable prefixes to full token chunks and assigns separate suffix chunks per request. Use the same chunk size for export and vLLM. Fix `PYTHONHASHSEED` across benchmark processes because the pinned dataset implementation seeds token expansion from Python's hash of the chunk key. The exact endpoint and arguments supported by your installation are listed in the [upstream CLI reference](https://docs.vllm.ai/en/stable/cli/bench/serve/).

Save the following manifest at `runs/live/experiment.json`, then run `python -m pdserve experiment runs/live/experiment.json --out-dir runs/live-report`.

```json
{
  "schema_version": 1,
  "workload": "../../examples/prefix-bursts.jsonl",
  "baseline": "baseline",
  "arms": [
    {"name": "baseline", "vllm_result": "baseline.json"},
    {"name": "candidate", "vllm_result": "candidate.json"}
  ],
  "slo": {"ttft_us": 50000, "tpot_us": 250},
  "gates": {"max_ttft_p95_delta_us": 5000, "min_slo_attainment": 0.8}
}
```

All paths are relative to the manifest. The SLO example uses the reference workload's illustrative bounds. Choose thresholds for the deployment under test. Manifest arms accept either `cluster` plus `router`, `vllm_result`, or `events`. Gates are optional and apply to each candidate. Available gates bound p95 TTFT and E2E increases in microseconds and set a minimum SLO attainment. A configured latency gate fails when either arm lacks the measurement.

vLLM's saved arrays are aligned to request submission order and omit stable request IDs. Import matches them against the exported workload order and checks request counts and prompt lengths. Identical prompt lengths make some reorderings impossible to detect from the result alone. Preserve the exported order, disable shuffling and keep the same dataset and tokenizer. Generated text and raw error messages are excluded from report bundles. The raw vLLM result files can contain both.

For vLLM results, latency is client-observed from send to tokens and goodput uses the benchmark's declared duration. For replay and event files, the observation window spans the first arrival to the last completion. SLO attainment uses all requests, including failures. Latency percentiles use successful, observable samples and nearest rank. A one-token success contributes to TTFT, E2E and goodput without contributing a TPOT sample. Zero-token successes have no TTFT and do not meet the configured TTFT/TPOT SLO.

## Import PD stage events

Choose an arm such as `{"name":"instrumented","events":"pd-events.jsonl"}`. Each JSONL row requires `request_id`, `event` and integer `timestamp_us`. Use one normalized clock domain per run. A distributed collector needs to align worker clocks before exporting events. The `arrival` event includes `prompt_tokens`, while `complete` includes `output_tokens` and a status of `ok` or `error`. Emit `first_token` whenever the output token count is positive.

```jsonl
{"request_id":"alpha","event":"arrival","timestamp_us":0,"prompt_tokens":32}
{"request_id":"alpha","event":"prefill_start","timestamp_us":30,"cache_tier":"gpu"}
{"request_id":"alpha","event":"prefill_ready","timestamp_us":150}
{"request_id":"alpha","event":"transfer_start","timestamp_us":180}
{"request_id":"alpha","event":"transfer_end","timestamp_us":220,"kv_transfer_bytes":1048576}
{"request_id":"alpha","event":"decode_start","timestamp_us":260}
{"request_id":"alpha","event":"first_token","timestamp_us":300}
{"request_id":"alpha","event":"complete","timestamp_us":380,"status":"ok","output_tokens":2}
```

Stages between arrival and completion are optional. The example produces prefill, transfer and decode queue measurements of 30, 30 and 40 microseconds. File order may vary. Identical events are deduplicated, conflicting events and non-monotonic stage times fail validation, and missing stage pairs produce no latency sample. Every workload request needs a terminal row, including failures. The benchmark result alone contributes client observations. The events adapter is the boundary for adding server-side instrumentation later.
