"""Replay a workload through prefill, a shared transfer link, and decode."""

from collections import deque
from heapq import heappop, heappush
from itertools import count

from .model import MAX_TRACE_US, ROUTERS, PrefixCache


def copy_us(size_bytes, bytes_per_us):
    return (size_bytes + bytes_per_us - 1) // bytes_per_us


class PrefillWorker:
    def __init__(self, cluster):
        self.cache = PrefixCache(cluster.gpu_cache_tokens, cluster.host_cache_tokens)
        self.queue = deque()
        self.busy_until = None


class Simulator:
    def __init__(self, cluster, router):
        if router not in ROUTERS:
            raise ValueError(f"unsupported router {router!r}")
        self.cluster = cluster
        self.router = router
        self.prefill = [PrefillWorker(cluster) for _ in range(cluster.prefill_workers)]
        self.decode_ready = [0] * cluster.decode_workers
        self.link_ready = 0
        self.next_prefill = count()
        self.next_decode = count()
        self.events = []
        self.sequence = count()
        self.completed = {}

    def schedule(self, at, action, *args, arrival=False):
        # Complete work before routing new arrivals at the same timestamp.
        heappush(self.events, (at, int(arrival), next(self.sequence), action, args))

    def prefill_duration(self, request, tier):
        cluster = self.cluster
        cached = request.prefix_tokens if tier != "miss" else 0
        restore = 0
        if tier == "host":
            restore = copy_us(cached * cluster.kv_bytes_per_token, cluster.host_copy_bytes_per_us)
        return cluster.prefill_setup_us + (request.prompt_tokens - cached) * cluster.prefill_us_per_token + restore

    def prefill_score(self, now, worker, request):
        remaining = 0 if worker.busy_until is None else worker.busy_until - now
        queued = sum(self.prefill_duration(waiting, "miss") for waiting in worker.queue)
        if self.router != "cache-aware":
            return remaining + queued
        return remaining + queued + self.prefill_duration(request, worker.cache.locate(request.prefix_key))

    def arrive(self, now, request):
        if self.router == "round-robin":
            worker_id = next(self.next_prefill) % len(self.prefill)
        else:
            worker_id = min(range(len(self.prefill)),
                            key=lambda index: self.prefill_score(now, self.prefill[index], request))
        worker = self.prefill[worker_id]
        worker.queue.append(request)
        if worker.busy_until is None:
            self.start_prefill(now, worker_id)

    def start_prefill(self, now, worker_id):
        worker = self.prefill[worker_id]
        request = worker.queue.popleft()
        tier = worker.cache.locate(request.prefix_key)
        hit_tokens = request.prefix_tokens if tier != "miss" else 0
        cluster = self.cluster
        restore_bytes = hit_tokens * cluster.kv_bytes_per_token if tier == "host" else 0
        row = {
            "id": request.id, "status": "ok", "start_us": request.arrival_us,
            "prompt_tokens": request.prompt_tokens, "output_tokens": request.output_tokens,
            "prefix_tokens": request.prefix_tokens, "cache_tier": tier, "cached_prefix_tokens": hit_tokens,
            "host_restore_bytes": restore_bytes,
            "prefill_worker": worker_id, "prefill_start_us": now,
            "prefill_compute_start_us": now + copy_us(restore_bytes, cluster.host_copy_bytes_per_us),
        }
        worker.busy_until = now + self.prefill_duration(request, tier)
        self.schedule(worker.busy_until, self.prefill_computed, worker_id, request, row)

    def prefill_computed(self, now, worker_id, request, row):
        worker = self.prefill[worker_id]
        cluster = self.cluster
        written = worker.cache.remember(request.prefix_key, request.prefix_tokens)
        row["host_write_bytes"] = written * cluster.kv_bytes_per_token
        row["prefill_compute_end_us"] = now
        worker.busy_until = now + copy_us(row["host_write_bytes"], cluster.host_copy_bytes_per_us)
        self.schedule(worker.busy_until, self.prefill_ready, worker_id, row)

    def prefill_ready(self, now, worker_id, row):
        cluster = self.cluster
        kv_bytes = row["prompt_tokens"] * cluster.kv_bytes_per_token
        transfer_start = max(now, self.link_ready)
        self.link_ready = transfer_start + cluster.transfer_setup_us + copy_us(kv_bytes, cluster.transfer_bytes_per_us)
        row.update(prefill_ready_us=now, kv_transfer_bytes=kv_bytes,
                   kv_transfer_start_us=transfer_start, kv_transfer_end_us=self.link_ready)
        self.schedule(self.link_ready, self.transfer_ready, row)
        worker = self.prefill[worker_id]
        worker.busy_until = None
        if worker.queue:
            self.start_prefill(now, worker_id)

    def transfer_ready(self, now, row):
        if self.router == "round-robin":
            worker_id = next(self.next_decode) % len(self.decode_ready)
        else:
            worker_id = min(range(len(self.decode_ready)), key=self.decode_ready.__getitem__)
        cluster = self.cluster
        decode_start = max(now, self.decode_ready[worker_id])
        first = decode_start + cluster.decode_first_us
        end = first + (row["output_tokens"] - 1) * cluster.decode_us_per_token
        if end > MAX_TRACE_US:
            raise ValueError(f"request {row['id']!r} exceeds the supported microsecond range")
        self.decode_ready[worker_id] = end
        row.update(decode_worker=worker_id, decode_start_us=decode_start, first_token_us=first, end_us=end)
        self.completed[row["id"]] = row


def replay(cluster, requests, router):
    ordered = sorted(requests, key=lambda request: request.arrival_us)
    if not ordered:
        raise ValueError("the workload contains no requests")
    ids = set()
    prefixes = {}
    for request in ordered:
        if request.id in ids:
            raise ValueError(f"duplicate request id {request.id!r}")
        ids.add(request.id)
        if request.prefix_key is not None:
            known_size = prefixes.setdefault(request.prefix_key, request.prefix_tokens)
            if known_size != request.prefix_tokens:
                raise ValueError(f"prefix {request.prefix_key!r} has inconsistent token counts")

    simulator = Simulator(cluster, router)
    for request in ordered:
        simulator.schedule(request.arrival_us, simulator.arrive, request, arrival=True)
    while simulator.events:
        now, _, _, action, args = heappop(simulator.events)
        action(now, *args)
    return [simulator.completed[request.id] for request in ordered]


def summarize(rows, cluster, router):
    def p95(values):
        ordered = sorted(values)
        return ordered[(len(ordered) * 95 + 99) // 100 - 1]

    prefill = [0] * cluster.prefill_workers
    decode = [0] * cluster.decode_workers
    hits = dict.fromkeys(("gpu", "host", "miss"), 0)
    for row in rows:
        prefill[row["prefill_worker"]] += 1
        decode[row["decode_worker"]] += 1
        hits[row["cache_tier"]] += 1

    return {
        "router": router, "requests": len(rows), "cache_residency_at_prefill_start": hits,
        "cacheable_requests": sum(row["prefix_tokens"] > 0 for row in rows),
        "cached_prefix_tokens": sum(row["cached_prefix_tokens"] for row in rows),
        "kv_transfer_bytes": sum(row["kv_transfer_bytes"] for row in rows),
        "host_restore_bytes": sum(row["host_restore_bytes"] for row in rows),
        "host_write_bytes": sum(row["host_write_bytes"] for row in rows),
        "prefill_requests_by_worker": prefill, "decode_requests_by_worker": decode,
        "p95_queue_us": {
            "prefill": p95(row["prefill_start_us"] - row["start_us"] for row in rows),
            "transfer": p95(row["kv_transfer_start_us"] - row["prefill_ready_us"] for row in rows),
            "decode": p95(row["decode_start_us"] - row["kv_transfer_end_us"] for row in rows),
        },
        "p95_ttft_us": p95(row["first_token_us"] - row["start_us"] for row in rows),
        "p95_e2e_us": p95(row["end_us"] - row["start_us"] for row in rows),
    }
