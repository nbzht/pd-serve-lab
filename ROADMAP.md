# PD Serve Lab roadmap

The project studies how routing, cache locality, KV movement and worker queues affect TTFT, TPOT and SLO goodput in disaggregated serving. Work proceeds in weekly increments. Keep identical workloads and SLOs between comparisons and change one parameter at a time.

| Milestone | Scope | Acceptance check |
| --- | --- | --- |
| 0.1, replay baseline | Deterministic replay, three routing policies, GPU/host prefix LRU, one shared KV link, stage timelines and Trace SLO export | Run the bundled experiment from a fresh checkout, reproduce the documented comparison, and pass the hand-calculated queue and cache tests. |
| 0.2, measured trace ingestion | Define a versioned event schema and write the first adapter for a pinned vLLM 1P1D setup. Capture arrival, prefill, KV transfer, decode and completion with request IDs. | Replay an archived fixture with duplicates, missing events and mixed request order. Check clock alignment and reconcile request counts, durations and server-side events. Label incomplete timing spans. |
| 0.3, KV transfer and offload | Track queue wait, bytes, copy time and cache tier. Compare transfer bandwidth, staging buffers and host spill/restore. Add one documented backend experiment, starting with an available vLLM connector. | Hold model, arrival trace and SLO fixed. Verify bytes against model geometry and captured events. Report measured versus modeled timings with the engine version and hardware. |
| 0.4, router and load balancing | Compare least-load, cache affinity and affinity with queue limits under hot-prefix, burst and uneven-worker traffic. | Replay identical workloads across policies. Report TTFT, TPOT, goodput, stage queue tails and per-worker skew. Include a case where higher cache hits make a tail metric worse. |
| 0.5, repeatable engine comparisons | Add a second engine adapter after the first is stable. Build a repeatable comparison across engine versions, transfer backends and worker ratios. | Provide a clean-checkout command, config, provenance and measurement output for each published result. Identify the hypotheses that did not hold. |

Prioritize the smallest testable item in the earliest unfinished milestone. Pair it with a fixture or a reproducible command. Use the synthetic runner for experiments that need only controlled queue and cache behavior. Reserve claims about live performance for recorded runs with sufficient hardware and version metadata.
