"""Interop with vLLM's timed_trace dataset and detailed serving benchmark."""

from decimal import Decimal
import json

from .model import MAX_TRACE_US, integer, ordered_workload


DETAILED_FIELDS = ("input_lens", "output_lens", "start_times", "ttfts", "latencies", "errors")
CAPTURE_FIELDS = ("backend", "model_id", "tokenizer_id", "num_prompts", "request_rate", "max_concurrency")


def timed_trace(requests, chunk_tokens):
    integer(chunk_tokens, "chunk_tokens", 1)
    prefix_ids = {}
    next_id = 0
    for request in ordered_workload(requests):
        hashes = []
        for offset in range(0, request.prompt_tokens, chunk_tokens):
            if offset + chunk_tokens <= request.prefix_tokens:
                key = (request.prefix_key, offset)
                if key not in prefix_ids:
                    prefix_ids[key] = next_id
                    next_id += 1
                hashes.append(prefix_ids[key])
            else:
                hashes.append(next_id)
                next_id += 1
        yield {"timestamp": request.arrival_us, "input_length": request.prompt_tokens,
               "output_length": request.output_tokens, "hash_ids": hashes}


def seconds(value, field):
    if type(value) not in (int, Decimal) or not 0 <= value <= MAX_TRACE_US:
        raise ValueError(f"{field} must be a finite nonnegative number of seconds")
    return Decimal(value)


def microseconds(value, field):
    converted = round(value * 1_000_000)
    integer(converted, field)
    return converted


def import_detailed_result(text, workload):
    result = json.loads(text, parse_float=Decimal, parse_constant=lambda value: None)
    if not isinstance(result, dict):
        raise ValueError("expected a single vLLM result object")
    size = len(workload)
    for key in DETAILED_FIELDS:
        values = result.get(key)
        if not isinstance(values, list) or len(values) != size:
            raise ValueError(f"{key} must have one entry per workload row; use --save-result --save-detailed")
    duration = microseconds(seconds(result.get("duration"), "duration"), "duration_us")
    starts = [seconds(value, "start_times") for value in result["start_times"]]
    origin = min(starts)
    rows = []
    arrays = (result[key] for key in DETAILED_FIELDS)
    for request, fields in zip(workload, zip(*arrays), strict=True):
        prompt, output, start, ttft, latency, error = fields
        integer(prompt, "input_lens", 1)
        integer(output, "output_lens")
        if prompt != request.prompt_tokens:
            raise ValueError(f"request {request.id!r} has a different input length; verify workload order and tokenizer")
        if not isinstance(error, str):
            raise ValueError(f"request {request.id!r} has a non-string error entry")
        start = seconds(start, "start_times") - origin
        latency = seconds(latency, "latencies")
        ttft = seconds(ttft, "ttfts")
        failed = bool(error) or (not output and not ttft and not latency)
        if not failed and ttft > latency:
            raise ValueError(f"request {request.id!r} has TTFT greater than its E2E latency")
        rows.append({
            "id": request.id, "status": "error" if failed else "ok", "prompt_tokens": prompt,
            "output_tokens": 0 if failed else output, "start_us": microseconds(start, "start_us"),
            "first_token_us": microseconds(start + ttft, "first_token_us") if output and not failed else None,
            "end_us": microseconds(start + latency, "end_us"),
        })
    completed = result.get("completed")
    integer(completed, "completed")
    if completed != sum(row["status"] == "ok" for row in rows):
        raise ValueError("per-request status disagrees with the vLLM completed count")
    if "failed" in result:
        integer(result["failed"], "failed")
        if result["failed"] != size - completed:
            raise ValueError("per-request status disagrees with the vLLM failed count")
    capture = {}
    for field in CAPTURE_FIELDS:
        value = result.get(field)
        if type(value) in (str, int):
            capture[field] = value
        elif isinstance(value, Decimal):
            capture[field] = str(value)
    return rows, duration, capture
