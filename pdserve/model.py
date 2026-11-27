"""Experiment inputs and the worker-local prefix cache."""

from collections import OrderedDict
from dataclasses import dataclass
import json


ROUTERS = ("round-robin", "least-loaded", "cache-aware")
MAX_TRACE_US = 2**63 - 1


def integer(value, name, minimum=0):
    if type(value) is not int or not minimum <= value <= MAX_TRACE_US:
        raise ValueError(f"{name} must be an integer between {minimum} and {MAX_TRACE_US}")


@dataclass(frozen=True)
class Cluster:
    prefill_workers: int
    decode_workers: int
    gpu_cache_tokens: int
    host_cache_tokens: int
    kv_bytes_per_token: int
    prefill_setup_us: int
    prefill_us_per_token: int
    transfer_setup_us: int
    transfer_bytes_per_us: int
    host_copy_bytes_per_us: int
    decode_first_us: int
    decode_us_per_token: int

    def __post_init__(self):
        zero_allowed = {"gpu_cache_tokens", "host_cache_tokens", "prefill_setup_us", "transfer_setup_us"}
        for name, value in vars(self).items():
            integer(value, name, 0 if name in zero_allowed else 1)


@dataclass(frozen=True)
class Request:
    id: str
    arrival_us: int
    prompt_tokens: int
    output_tokens: int
    prefix_key: str | None = None
    prefix_tokens: int = 0

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("id must be a nonempty string")
        integer(self.arrival_us, "arrival_us")
        integer(self.prompt_tokens, "prompt_tokens", 1)
        integer(self.output_tokens, "output_tokens", 1)
        integer(self.prefix_tokens, "prefix_tokens")
        if self.prefix_tokens > self.prompt_tokens:
            raise ValueError("prefix_tokens cannot exceed prompt_tokens")
        if self.prefix_tokens and (not isinstance(self.prefix_key, str) or not self.prefix_key.strip()):
            raise ValueError("a positive prefix_tokens requires a nonempty prefix_key")
        if not self.prefix_tokens and self.prefix_key is not None:
            raise ValueError("prefix_key must be absent or null when prefix_tokens is zero")


def load_cluster(path):
    try:
        return Cluster(**json.loads(path.read_text(encoding="utf-8-sig")))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_workload(path):
    requests = []
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                requests.append(Request(**json.loads(line)))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return requests


class TokenLRU:
    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.used = 0

    def remove(self, key):
        self.used -= self.entries.pop(key, 0)

    def insert(self, key, tokens):
        self.remove(key)
        evicted = []
        while self.used + tokens > self.capacity:
            old_key, old_tokens = self.entries.popitem(last=False)
            self.used -= old_tokens
            evicted.append((old_key, old_tokens))
        self.entries[key] = tokens
        self.used += tokens
        return evicted


class PrefixCache:
    def __init__(self, gpu_capacity, host_capacity):
        self.gpu = TokenLRU(gpu_capacity)
        self.host = TokenLRU(host_capacity)

    def locate(self, key):
        if key in self.gpu.entries:
            return "gpu"
        if key in self.host.entries:
            return "host"
        return "miss"

    def remember(self, key, tokens):
        """Update LRU tiers and return the token volume written from GPU to host."""
        if key is None:
            return 0
        if tokens <= self.gpu.capacity:
            self.host.remove(key)
            evicted = self.gpu.insert(key, tokens)
            written = 0
            for old_key, old_tokens in evicted:
                if old_tokens <= self.host.capacity:
                    self.host.insert(old_key, old_tokens)
                    written += old_tokens
            return written
        if tokens <= self.host.capacity:
            already_in_host = key in self.host.entries
            self.host.insert(key, tokens)
            return 0 if already_in_host else tokens
        return 0
