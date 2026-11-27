"""Assemble a request timeline from the PD event stream contract."""

from .model import integer, jsonl_objects


EVENT_TIMES = {
    "arrival": "start_us",
    "prefill_start": "prefill_start_us",
    "prefill_ready": "prefill_ready_us",
    "transfer_start": "kv_transfer_start_us",
    "transfer_end": "kv_transfer_end_us",
    "decode_start": "decode_start_us",
    "first_token": "first_token_us",
    "complete": "end_us",
}


def parse_event(raw):
    request_id = raw.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be a nonempty string")
    event = raw.get("event")
    if not isinstance(event, str) or event not in EVENT_TIMES:
        raise ValueError(f"unknown event {event!r}")
    timestamp = raw.get("timestamp_us")
    integer(timestamp, "timestamp_us")
    fields = {EVENT_TIMES[event]: timestamp}
    if event == "arrival":
        tokens = raw.get("prompt_tokens")
        integer(tokens, "prompt_tokens", 1)
        fields["prompt_tokens"] = tokens
    elif event == "complete":
        status = raw.get("status")
        if status not in ("ok", "error"):
            raise ValueError("complete status must be 'ok' or 'error'")
        tokens = raw.get("output_tokens")
        integer(tokens, "output_tokens")
        fields.update(status=status, output_tokens=tokens)
    elif event == "prefill_start" and "cache_tier" in raw:
        cache_tier = raw["cache_tier"]
        if cache_tier not in ("gpu", "host", "miss"):
            raise ValueError("cache_tier must be 'gpu', 'host', or 'miss'")
        fields["cache_tier"] = cache_tier
    elif event == "transfer_end" and "kv_transfer_bytes" in raw:
        size = raw["kv_transfer_bytes"]
        integer(size, "kv_transfer_bytes")
        fields["kv_transfer_bytes"] = size
    return request_id, event, fields


def assemble_events(text, source, workload):
    expected = {request.id: request for request in workload}
    grouped = {}
    for line_number, raw in jsonl_objects(text, source):
        try:
            request_id, event, fields = parse_event(raw)
            if request_id not in expected:
                raise ValueError(f"request {request_id!r} is absent from the workload")
            events = grouped.setdefault(request_id, {})
            if events.setdefault(event, fields) != fields:
                raise ValueError(f"request {request_id!r} has conflicting {event!r} events")
        except ValueError as exc:
            raise ValueError(f"{source}:{line_number}: {exc}") from exc

    rows = []
    for request_id, request in expected.items():
        events = grouped.get(request_id, {})
        if "arrival" not in events or "complete" not in events:
            raise ValueError(f"{source}: request {request_id!r} needs both arrival and complete events")
        row = {"id": request_id, "first_token_us": None}
        for fields in events.values():
            row.update(fields)

        if row["prompt_tokens"] != request.prompt_tokens:
            raise ValueError(f"{source}: request {request_id!r} differs from the workload prompt token count")
        if (row["output_tokens"] > 0) != (row["first_token_us"] is not None):
            raise ValueError(f"{source}: request {request_id!r} needs a first_token event exactly when output_tokens > 0")
        times = [row[key] for key in EVENT_TIMES.values() if row.get(key) is not None]
        if times != sorted(times):
            raise ValueError(f"{source}: request {request_id!r} has non-monotonic stage timestamps")
        rows.append(row)
    return rows
