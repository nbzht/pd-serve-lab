# PD Serve Lab roadmap

The project studies how routing, cache locality, KV movement and worker queues affect TTFT, TPOT and SLO goodput in disaggregated serving. Work proceeds in weekly increments. Keep identical workloads and SLOs between comparisons and change one parameter at a time.

| Milestone | Scope | Acceptance check |
| --- | --- | --- |
| 0.1, replay baseline | Deterministic replay, three routing policies, GPU/host prefix LRU, one shared KV link, stage timelines and Trace SLO export | Run the bundled experiment from a fresh checkout, reproduce the documented comparison, and pass the hand-calculated queue and cache tests. |
| 0.2, experiment harness | Run baseline and candidate manifests, export vLLM timed traces, ingest detailed vLLM benchmark results and PD stage events, and publish reproducible reports with regression gates. | Reproduce both bundled experiments. Verify artifact hashes and test duplicate, missing and out-of-order events, missing stage measurements and a failed regression gate. |
| 0.3, KV transfer and offload | Track queue wait, bytes, copy time and cache tier. Compare transfer bandwidth, staging buffers and host spill/restore. Add one documented backend experiment, starting with an available vLLM connector. | Hold model, arrival trace and SLO fixed. Verify bytes against model geometry and captured events. Report measured versus modeled timings with the engine version and hardware. |
| 0.4, router and load balancing | Compare least-load, cache affinity and affinity with queue limits under hot-prefix, burst and uneven-worker traffic. | Replay identical workloads across policies. Report TTFT, TPOT, goodput, stage queue tails and per-worker skew. Include a case where higher cache hits make a tail metric worse. |
| 0.5, repeatable engine comparisons | Add a second engine adapter after the first is stable. Build a repeatable comparison across engine versions, transfer backends and worker ratios. | Provide a clean-checkout command, config, provenance and measurement output for each published result. Identify the hypotheses that did not hold. |

The next live milestone is one pinned vLLM 1P1D run that correlates request IDs across prefill, KV transfer and decode. Verify clock alignment and stage coverage before tuning the simulator. Pair new behavior with a fixture or a reproducible command. Reserve claims about live performance for recorded runs with sufficient hardware and version metadata.
