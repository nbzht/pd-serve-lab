"""Comparable metrics for simulated and recorded requests."""

from fractions import Fraction


QUEUE_SPANS = {
    "prefill": ("start_us", "prefill_start_us"),
    "transfer": ("prefill_ready_us", "kv_transfer_start_us"),
    "decode": ("kv_transfer_end_us", "decode_start_us"),
}


def nearest_rank(ordered, percentile):
    if not ordered:
        return None
    return ordered[(len(ordered) * percentile + 99) // 100 - 1]


def distribution(samples):
    ordered = sorted(samples)
    stats = {"count": len(ordered)}
    for percentile in (50, 95, 99):
        value = nearest_rank(ordered, percentile)
        stats[f"p{percentile}"] = float(value) if isinstance(value, Fraction) else value
    return stats


def summarize_run(rows, slo, observation_us=None):
    span = max(row["end_us"] for row in rows) - min(row["start_us"] for row in rows)
    window_us = span if observation_us is None else observation_us
    if window_us + 1 < span:
        raise ValueError("the declared observation window is shorter than its request span")
    if window_us <= 0:
        raise ValueError("the observation window must span a positive duration")

    successful = [row for row in rows if row["status"] == "ok"]
    latencies = {"ttft": [], "tpot": [], "e2e": []}
    passing = 0
    for row in successful:
        first = row["first_token_us"]
        output = row["output_tokens"]
        latencies["e2e"].append(row["end_us"] - row["start_us"])
        if first is None:
            continue
        ttft = first - row["start_us"]
        latencies["ttft"].append(ttft)
        tpot = Fraction(row["end_us"] - first, output - 1) if output > 1 else None
        if tpot is not None:
            latencies["tpot"].append(tpot)
        passing += ttft <= slo["ttft_us"] and (tpot is None or tpot <= slo["tpot_us"])

    queue_us = {
        stage: distribution(row[end] - row[start] for row in successful if start in row and end in row)
        for stage, (start, end) in QUEUE_SPANS.items()
    }
    cache_observed = [row for row in successful if "cache_tier" in row]
    transfer_observed = [row["kv_transfer_bytes"] for row in successful if "kv_transfer_bytes" in row]
    output_tokens = sum(row["output_tokens"] for row in successful)
    return {
        "requests": len(rows), "successful_requests": len(successful),
        "failed_requests": len(rows) - len(successful), "output_tokens": output_tokens,
        "observation_us": window_us, "latency_us": {name: distribution(values) for name, values in latencies.items()},
        "queue_us": queue_us, "slo_passing_requests": passing, "slo_attainment": passing / len(rows),
        "goodput_rps": passing * 1_000_000 / window_us,
        "request_throughput_rps": len(successful) * 1_000_000 / window_us,
        "output_throughput_tps": output_tokens * 1_000_000 / window_us,
        "cache_observations": len(cache_observed),
        "cache_residency": {tier: sum(row["cache_tier"] == tier for row in cache_observed)
                            for tier in ("gpu", "host", "miss")} if cache_observed else None,
        "kv_transfer_observations": len(transfer_observed),
        "kv_transfer_bytes": sum(transfer_observed) if transfer_observed else None,
    }
