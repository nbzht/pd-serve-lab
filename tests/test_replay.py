import unittest

from pdserve.model import Cluster, Request
from pdserve.simulator import replay, summarize


def tiny_cluster(**overrides):
    parameters = dict(prefill_workers=1, decode_workers=1, gpu_cache_tokens=3, host_cache_tokens=6,
                      kv_bytes_per_token=10, prefill_setup_us=1, prefill_us_per_token=4,
                      transfer_setup_us=2, transfer_bytes_per_us=10, host_copy_bytes_per_us=5,
                      decode_first_us=3, decode_us_per_token=2)
    return Cluster(**(parameters | overrides))


class ReplayTests(unittest.TestCase):
    def test_hand_calculated_timeline_and_shared_link(self):
        cluster = tiny_cluster(prefill_workers=2, transfer_bytes_per_us=2)
        arrivals = [Request("one", 0, 5, 8), Request("two", 0, 5, 1)]
        first, second = replay(cluster, arrivals, "round-robin")
        self.assertEqual(first["prefill_ready_us"], 21)
        self.assertEqual(first["kv_transfer_start_us"], 21)
        self.assertEqual(first["kv_transfer_end_us"], 48)
        self.assertEqual(first["first_token_us"], 51)
        self.assertEqual(first["end_us"], 65)
        self.assertEqual(second["prefill_ready_us"], 21)
        self.assertEqual(second["kv_transfer_start_us"], 48)
        self.assertEqual(second["kv_transfer_end_us"], 75)
        self.assertEqual(second["end_us"], 78)

    def test_decode_queue_wait_does_not_change_per_token_latency(self):
        cluster = tiny_cluster(prefill_workers=2)
        arrivals = [Request("long", 0, 2, 10), Request("queued", 0, 2, 3)]
        first, second = replay(cluster, arrivals, "least-loaded")
        self.assertGreater(second["decode_start_us"], second["kv_transfer_end_us"])
        self.assertEqual(second["decode_start_us"], first["end_us"])
        self.assertEqual(second["end_us"] - second["first_token_us"], 4)

    def test_host_eviction_restore_and_full_kv_transfer_on_hit(self):
        cluster = tiny_cluster(host_cache_tokens=3)
        requests = [Request("a-first", 0, 4, 1, "a", 3), Request("b-first", 100, 4, 1, "b", 3),
                    Request("a-again", 200, 4, 1, "a", 3), Request("a-hot", 300, 4, 1, "a", 3)]
        first, evictor, restored, hot = replay(cluster, requests, "cache-aware")
        self.assertEqual([row["cache_tier"] for row in (first, evictor, restored, hot)],
                         ["miss", "miss", "host", "gpu"])
        self.assertEqual(evictor["host_write_bytes"], 30)
        self.assertEqual(evictor["prefill_ready_us"] - evictor["prefill_compute_end_us"], 6)
        self.assertEqual(restored["host_restore_bytes"], 30)
        self.assertEqual(restored["prefill_compute_start_us"] - restored["prefill_start_us"], 6)
        self.assertEqual(hot["host_restore_bytes"], 0)
        self.assertEqual(hot["kv_transfer_bytes"], first["kv_transfer_bytes"])
        self.assertEqual(hot["cached_prefix_tokens"], 3)

    def test_host_only_prefix_survives_but_oversize_prefix_does_not(self):
        cluster = tiny_cluster(gpu_cache_tokens=1, host_cache_tokens=3)
        requests = [Request("a", 0, 3, 1, "x", 3), Request("b", 100, 3, 1, "x", 3),
                    Request("c", 200, 4, 1, "y", 4), Request("d", 300, 4, 1, "y", 4)]
        rows = replay(cluster, requests, "round-robin")
        self.assertEqual([row["cache_tier"] for row in rows], ["miss", "host", "miss", "miss"])
        self.assertEqual([row["host_write_bytes"] for row in rows], [30, 0, 0, 0])

    def test_cache_aware_uses_completed_prefix_on_another_worker(self):
        cluster = tiny_cluster(prefill_workers=2)
        requests = [Request("a", 0, 4, 1, "x", 3), Request("b", 17, 4, 1, "x", 3)]
        round_robin = replay(cluster, requests, "round-robin")
        cache_aware = replay(cluster, requests, "cache-aware")
        self.assertEqual(round_robin[1]["prefill_worker"], 1)
        self.assertEqual(cache_aware[1]["prefill_worker"], 0)
        self.assertEqual(cache_aware[1]["cache_tier"], "gpu")
        self.assertLess(cache_aware[1]["first_token_us"], round_robin[1]["first_token_us"])
        self.assertEqual(summarize(cache_aware, cluster, "cache-aware")["prefill_requests_by_worker"], [2, 0])

    def test_cache_affinity_does_not_force_a_saturated_worker(self):
        cluster = tiny_cluster(prefill_workers=2)
        requests = [Request("warm", 0, 4, 1, "x", 3), Request("busy", 50, 30, 1),
                    Request("choose", 51, 4, 1, "x", 3)]
        rows = replay(cluster, requests, "cache-aware")
        self.assertEqual(rows[-1]["prefill_worker"], 1)
        self.assertEqual(rows[-1]["cache_tier"], "miss")

    def test_invalid_workload_rejected_before_replay(self):
        cluster = tiny_cluster()
        with self.assertRaisesRegex(ValueError, "inconsistent token counts"):
            replay(cluster, [Request("a", 0, 4, 1, "x", 3), Request("b", 1, 4, 1, "x", 4)], "cache-aware")
        with self.assertRaisesRegex(ValueError, "duplicate request id"):
            replay(cluster, [Request("a", 0, 4, 1), Request("a", 1, 4, 1)], "least-loaded")
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            Request("a", 0, 4, True)


if __name__ == "__main__":
    unittest.main()
