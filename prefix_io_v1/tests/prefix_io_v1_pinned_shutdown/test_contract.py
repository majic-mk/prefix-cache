"""P316 STOP contracts using the real native CPU reactor thread and Futures.

CUDA events/file CQEs are explicit CPU gates. A query gate becoming ready is
the fixture's completion witness; counters and Future.done are never substituted
for that witness. No GPU execution or performance assertion exists here.
"""
import threading

import pytest
from py_kvcache.reactor import _SharedPreloadSlot
from py_kvcache.staging_cache import StagingDataCache
from tests.prefix_io_v1_start_budget._fixture import rig, eventually
from tests.prefix_io_v1_parent_admission._fixture import common, physical_streams


def _prepare(x, *, cache=False):
    r = common(x, limit=8)
    physical_streams(x)
    r._prefix_init_capacity_observation()
    if cache:
        r._staging_cache = StagingDataCache(policy="lru", capacity=8)
    return r


def _cache_seed(r, key):
    slot = r.staging_pool.try_reserve()
    assert slot is not None
    assert r._staging_cache.put(key, slot.index) is None
    return r._staging_cache.get(key)


def _shared_seed(r, key, owners):
    pool_slot = r.staging_pool.try_reserve()
    assert pool_slot is not None
    slot = _SharedPreloadSlot(key, pool_slot.index, None, cached=False)
    r._shared_cache_insert(slot)
    r._preload_refcount[key] = len(owners)
    for owner in owners:
        r._preload_owned[(owner, key)] = None
    return slot


def _join_normal(x):
    x.r._worker.join(timeout=2)
    assert not x.r._worker.is_alive()
    assert not x.r._active
    assert not x.r._inflight
    assert not x.r._pending_copies
    assert not x.r._copy_ready


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_stop_preserves_pinned_cache_until_actual_copy_event(monkeypatch, policy):
    with rig(monkeypatch, enabled=False, slots=4, depth=1) as x:
        r = _prepare(x, cache=True)
        r._staging_cache = StagingDataCache(policy=policy, capacity=8)
        held = _cache_seed(r, b"held")
        clean = _cache_seed(r, b"clean")
        x.copy_ready.clear()
        future = x.load(job_id=20, hashes=[b"held"])
        x.start()
        eventually(lambda: bool(r._pending_copies) and held.copies_inflight == 1)
        assert not future.done()
        r.shutdown(wait=False)
        eventually(lambda: r._stop and clean.slot_index in r.staging_pool._free)
        assert not future.done()
        assert r._staging_cache.get(b"held") is held
        assert held.copies_inflight == 1
        assert held.slot_index not in r.staging_pool._free
        assert r.staging_pool.releases.count(held.slot_index) == 0
        assert r._worker.is_alive()
        x.copy_ready.set()
        future.result(timeout=2)
        _join_normal(x)
        assert r.staging_pool.releases.count(held.slot_index) == 1
        assert r.staging_pool.releases.count(clean.slot_index) == 1
        assert r.staging_pool.free_count == 4
        assert len(r._staging_cache) == 0
        r._copy_stream.synchronize.assert_not_called()


def test_stop_preserves_same_cache_slot_with_multiple_consumers(monkeypatch):
    with rig(monkeypatch, enabled=False, slots=4, depth=1) as x:
        r = _prepare(x, cache=True)
        held = _cache_seed(r, b"held")
        x.copy_ready.clear()
        first = x.load(job_id=20, hashes=[b"held"])
        second = x.load(job_id=21, hashes=[b"held"])
        x.start()
        eventually(lambda: held.copies_inflight == 2 and bool(r._pending_copies))
        r.shutdown(wait=False)
        eventually(lambda: r._stop)
        assert not first.done() and not second.done()
        assert held.slot_index not in r.staging_pool._free
        assert r.staging_pool.releases.count(held.slot_index) == 0
        x.copy_ready.set()
        first.result(timeout=2)
        second.result(timeout=2)
        _join_normal(x)
        assert held.copies_inflight == 0
        assert r.staging_pool.releases.count(held.slot_index) == 1


def test_stop_uncaches_shared_slot_without_releasing_pending_copy(monkeypatch):
    with rig(monkeypatch, enabled=False, slots=4, depth=1, share=True) as x:
        r = _prepare(x)
        held = _shared_seed(r, b"held", ["cpu-load-20", "later-owner"])
        x.copy_ready.clear()
        future = x.load(job_id=20, hashes=[b"held"])
        x.start()
        eventually(lambda: held.copies_inflight == 1 and bool(r._pending_copies))
        assert r._shared_cached[b"held"] is held
        r.shutdown(wait=False)
        eventually(lambda: r._stop and b"held" not in r._shared_cached)
        assert held.cached is False
        assert held.copies_inflight == 1
        assert not future.done()
        assert held.slot_index not in r.staging_pool._free
        assert r.staging_pool.releases.count(held.slot_index) == 0
        x.copy_ready.set()
        future.result(timeout=2)
        _join_normal(x)
        assert r.staging_pool.releases.count(held.slot_index) == 1
        assert r.staging_pool.free_count == 4
        r._copy_stream.synchronize.assert_not_called()


def test_stop_preserves_shared_slot_until_all_copy_consumers_settle(monkeypatch):
    with rig(monkeypatch, enabled=False, slots=4, depth=1, share=True) as x:
        r = _prepare(x)
        held = _shared_seed(r, b"held",
                            ["cpu-load-20", "cpu-load-21", "later-owner"])
        x.copy_ready.clear()
        first = x.load(job_id=20, hashes=[b"held"])
        second = x.load(job_id=21, hashes=[b"held"])
        x.start()
        eventually(lambda: held.copies_inflight == 2 and bool(r._pending_copies))
        r.shutdown(wait=False)
        eventually(lambda: r._stop and not r._shared_cached)
        assert held.copies_inflight == 2
        assert not first.done() and not second.done()
        assert r.staging_pool.releases.count(held.slot_index) == 0
        x.copy_ready.set()
        first.result(timeout=2)
        second.result(timeout=2)
        _join_normal(x)
        assert held.copies_inflight == 0
        assert r.staging_pool.releases.count(held.slot_index) == 1


def test_last_demand_uncache_is_not_a_copy_completion_witness(monkeypatch):
    with rig(monkeypatch, enabled=False, slots=4, depth=1, share=True) as x:
        r = _prepare(x)
        held = _shared_seed(r, b"held", ["cpu-load-20"])
        x.copy_ready.clear()
        future = x.load(job_id=20, hashes=[b"held"])
        x.start()
        eventually(lambda: held.copies_inflight == 1 and bool(r._pending_copies))
        assert b"held" not in r._shared_cached
        assert held.cached is False
        assert not future.done()
        assert held.slot_index not in r.staging_pool._free
        r.shutdown(wait=False)
        eventually(lambda: r._stop)
        assert r.staging_pool.releases.count(held.slot_index) == 0
        x.copy_ready.set()
        future.result(timeout=2)
        _join_normal(x)
        assert r.staging_pool.releases.count(held.slot_index) == 1


def test_stop_does_not_zero_real_preload_read_ownership(monkeypatch):
    with rig(monkeypatch, enabled=False, slots=4, depth=1, share=True) as x:
        r = _prepare(x)
        r.ring.auto = False
        x.preload([b"preload-held"])
        x.start()
        eventually(lambda: r._preload_inflight_total == 1 and bool(r.ring.pending))
        # Only the real open CQE is released; the subsequent read remains pending.
        r.ring.permits.release()
        eventually(lambda: x.reads == 1 and bool(r.ring.pending))
        free_before_stop = r.staging_pool.free_count
        releases_before_stop = list(r.staging_pool.releases)
        assert free_before_stop < 4
        r.shutdown(wait=False)
        eventually(lambda: r._stop)
        assert r._preload_inflight_total == 1
        assert r._preload_inflight_hashes[b"preload-held"] == 1
        assert r.staging_pool.free_count == free_before_stop
        assert r.staging_pool.releases == releases_before_stop
        assert r._worker.is_alive()
        r.ring.auto = True
        _join_normal(x)
        assert r._preload_inflight_total == 0
        assert not r._preload_inflight_hashes
        assert r.staging_pool.free_count == 4


def test_stop_unknown_sample_cannot_claim_released_pinned_capacity(monkeypatch):
    with rig(monkeypatch, enabled=False, slots=4, depth=1) as x:
        r = _prepare(x, cache=True)
        held = _cache_seed(r, b"held")
        x.copy_ready.clear()
        future = x.load(job_id=20, hashes=[b"held"])
        samples = []
        original_pump = r._pump_once
        def sample_on_owner():
            result = original_pump()
            if r._stop and len(samples) < 8:
                samples.append((threading.get_ident(), r._prefix_capacity_snapshot()))
            return result
        r._pump_once = sample_on_owner
        x.start()
        eventually(lambda: held.copies_inflight == 1 and bool(r._pending_copies))
        r.shutdown(wait=False)
        eventually(lambda: bool(samples))
        owner, sample = samples[-1]
        assert owner == r._worker.ident
        assert sample["valid"] is False
        assert sample["free_slots"] is None
        assert sample["free_reclaimable_staging_bytes"] is None
        assert sample["physical_release_credit"] is False
        assert held.slot_index not in r.staging_pool._free
        assert not future.done()
        x.copy_ready.set()
        future.result(timeout=2)
        _join_normal(x)
        assert r.staging_pool.releases.count(held.slot_index) == 1
