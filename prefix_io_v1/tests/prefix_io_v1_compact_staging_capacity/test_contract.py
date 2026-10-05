"""P316 capacity observation contracts; native objects, CPU only.

These tests never infer a DMA completion or a released resource from a scalar.
The root CPU matrix selects the isolated P316 native worktree on PYTHONPATH.
"""
from collections import OrderedDict
import threading
from types import SimpleNamespace as NS

import pytest
from py_kvcache.reactor import IoReactor, _SharedPreloadSlot
from py_kvcache.staging import StagingPool
from py_kvcache.staging_cache import StagingDataCache


class NoScanDict(dict):
    """Allow membership/cardinality/lookup; forbid registry enumeration."""
    def __iter__(self):
        raise AssertionError("capacity getter enumerated a registry")
    def values(self):
        raise AssertionError("capacity getter enumerated registry values")
    def items(self):
        raise AssertionError("capacity getter enumerated registry items")
    def keys(self):
        raise AssertionError("capacity getter enumerated registry keys")


@pytest.fixture(params=["lru", "arc"])
def cache(request):
    return StagingDataCache(policy=request.param, capacity=8)


def _native_model(slots=192, cache_capacity=128, initialize=True):
    r = IoReactor.__new__(IoReactor)
    r.staging_pool = StagingPool(slot_count=slots)
    r.file_store = NS(io_size=4096)
    r._staging_cache = StagingDataCache(policy="lru", capacity=cache_capacity)
    r._shared_cached = {}
    r._preload_slots = OrderedDict()
    r._preload_cached_total = 0
    r._preload_inflight_total = 0
    r._preload_inflight_hashes = {}
    r._preload_refcount = {}
    r._preload_owned = OrderedDict()
    r._max_preload_owned = 16384
    r._share_preload = True
    r._closed = r._stop = False
    r._native_drain_unknown = False
    r._worker = NS(ident=threading.get_ident())
    if initialize:
        r._prefix_init_capacity_observation()
    return r


def _reserve(r):
    slot = r.staging_pool.try_reserve()
    assert slot is not None
    return slot.index


def _cache_add(r, key):
    index = _reserve(r)
    assert r._staging_cache.put(key, index) is None
    return r._staging_cache.get(key)


def _shared_add(r, key):
    slot = _SharedPreloadSlot(key, _reserve(r), None, cached=False)
    r._shared_cache_insert(slot)
    return slot


def _ordinary_add(r, key):
    index = _reserve(r)
    r._preload_cache_push(key, index, None)
    return index


def _assert_cache(cache, clean, pinned):
    assert cache.observation_valid is True
    assert cache.clean_slot_count == clean
    assert cache.pinned_slot_count == pinned


def test_empty_cache_has_known_zero_counts(cache):
    _assert_cache(cache, 0, 0)
    assert cache.observation_error is None


def test_insert_and_touch_count_physical_slots(cache):
    assert cache.put(b"a", 3) is None
    first = cache.get(b"a")
    assert first.slot_index == 3
    assert cache.get(b"a") is first
    _assert_cache(cache, 1, 0)


def test_duplicate_put_returns_input_without_new_credit(cache):
    cache.put(b"a", 3)
    first = cache.get(b"a")
    assert cache.put(b"a", 7) == 7
    assert cache.get(b"a") is first
    _assert_cache(cache, 1, 0)


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_zero_capacity_returns_slot_and_counts_no_retention(policy):
    cache = StagingDataCache(policy=policy, capacity=0)
    assert cache.put(b"a", 9) == 9
    _assert_cache(cache, 0, 0)
    assert b"a" not in cache


def test_nested_copy_refs_pin_one_physical_slot(cache):
    cache.put(b"a", 3)
    slot = cache.get(b"a")
    cache.pin(slot)
    _assert_cache(cache, 0, 1)
    cache.pin(slot)
    assert slot.copies_inflight == 2
    _assert_cache(cache, 0, 1)
    cache.unpin(slot)
    assert slot.copies_inflight == 1
    _assert_cache(cache, 0, 1)
    cache.unpin(slot)
    _assert_cache(cache, 1, 0)


def test_evict_skips_pinned_and_returns_only_actual_clean_slot(cache):
    cache.put(b"a", 3)
    cache.put(b"b", 4)
    held = cache.get(b"a")
    cache.pin(held)
    assert cache.evict_one() == 4
    assert cache.get(b"a") is held
    assert cache.evict_one() is None
    _assert_cache(cache, 0, 1)


@pytest.mark.parametrize("policy", ["lru", "arc"])
def test_all_pinned_capacity_behavior_is_preserved(policy):
    cache = StagingDataCache(policy=policy, capacity=1)
    cache.put(b"a", 3)
    first = cache.get(b"a")
    cache.pin(first)
    # Author behavior permits insertion when no existing victim is clean.
    assert cache.put(b"b", 4) is None
    assert len(cache) == 2
    assert cache.get(b"a") is first
    _assert_cache(cache, 1, 1)


def test_stop_clean_drain_preserves_pinned_object_and_policy(cache):
    cache.put(b"a", 3)
    cache.put(b"b", 4)
    pinned = cache.get(b"a")
    cache.pin(pinned)
    assert cache.drain_unpinned() == [4]
    assert cache.get(b"a") is pinned
    assert b"a" in list(cache.policy.victims())
    _assert_cache(cache, 0, 1)
    cache.unpin(pinned)
    assert cache.drain_unpinned() == [3]
    assert cache.drain_unpinned() == []
    _assert_cache(cache, 0, 0)


def test_clean_legacy_drain_is_exactly_once(cache):
    cache.put(b"a", 3)
    cache.put(b"b", 4)
    assert sorted(cache.drain()) == [3, 4]
    assert cache.drain() == []
    _assert_cache(cache, 0, 0)


def test_invalidation_is_sticky_without_resource_mutation(cache):
    cache.put(b"a", 3)
    slot = cache.get(b"a")
    cache.invalidate_observation("first fault")
    cache.invalidate_observation("later fault")
    cache.pin(slot)
    cache.unpin(slot)
    assert cache.get(b"a") is slot
    assert slot.copies_inflight == 0
    assert cache.observation_valid is False
    assert cache.observation_error == "first fault"
    assert cache.clean_slot_count is None
    assert cache.pinned_slot_count is None


def test_stale_pin_keeps_original_ref_mutation_but_unknown(cache):
    cache.put(b"a", 3)
    old = cache.get(b"a")
    assert cache.evict_one() == 3
    cache.pin(old)
    assert old.copies_inflight == 1
    assert len(cache) == 0
    assert cache.clean_slot_count is None
    assert cache.pinned_slot_count is None


def test_underflow_cannot_become_clean_credit(cache):
    cache.put(b"a", 3)
    slot = cache.get(b"a")
    cache.unpin(slot)
    assert slot.copies_inflight == -1
    assert cache.clean_slot_count is None
    assert cache.pinned_slot_count is None
    assert cache.drain_unpinned() == []
    assert cache.get(b"a") is slot


@pytest.mark.parametrize("field,value", [
    ("_pinned_slots", -1), ("_pinned_slots", 2),
    ("_pinned_slots", True), ("_observed_slots", 9),
])
def test_corrupt_cache_scalar_is_unknown_without_clearing(field, value):
    cache = StagingDataCache(policy="lru", capacity=2)
    cache.put(b"a", 3)
    slot = cache.get(b"a")
    setattr(cache, field, value)
    assert cache.clean_slot_count is None
    assert cache.pinned_slot_count is None
    assert cache.get(b"a") is slot


def test_observation_hook_failure_does_not_change_native_pin(cache, monkeypatch):
    cache.put(b"a", 3)
    slot = cache.get(b"a")
    def fail(*args):
        raise RuntimeError("test observer fault")
    monkeypatch.setattr(cache, "_observe_copy_delta", fail)
    cache.pin(slot)
    assert slot.copies_inflight == 1
    assert cache.get(b"a") is slot
    assert cache.clean_slot_count is None


def test_size_hook_failure_retains_real_insert(cache, monkeypatch):
    def fail(*args):
        raise RuntimeError("test size observer fault")
    monkeypatch.setattr(cache, "_observe_size_delta", fail)
    assert cache.put(b"a", 3) is None
    assert cache.get(b"a").slot_index == 3
    assert cache.clean_slot_count is None


def test_original_policy_failure_propagates_and_cannot_leave_known_count(cache, monkeypatch):
    def fail(*args):
        raise RuntimeError("original policy fault")
    monkeypatch.setattr(cache.policy, "admit", fail)
    with pytest.raises(RuntimeError, match="original policy fault"):
        cache.put(b"a", 3)
    assert b"a" in cache
    assert cache.clean_slot_count is None


def test_large_cache_getters_do_not_enumerate_registry():
    cache = StagingDataCache(policy="lru", capacity=160)
    for i in range(130):
        cache.put(str(i).encode(), i)
    for i in range(11):
        cache.pin(cache.get(str(i).encode()))
    cache._slots = NoScanDict(cache._slots)
    for _ in range(10):
        _assert_cache(cache, 119, 11)


def test_foreground_snapshot_counts_disjoint_native_slots_once():
    r = _native_model(slots=10)
    _cache_add(r, b"cache")
    shared = _shared_add(r, b"shared")
    _ordinary_add(r, b"ordinary")
    r._shared_pin(shared)
    snap = r._prefix_capacity_snapshot()
    assert snap["valid"] is True
    assert snap["free_slots"] == 7
    assert snap["cache_clean_slots"] == 1
    assert snap["shared_cached_slots"] == 1
    assert snap["shared_cached_pinned_slots"] == 1
    assert snap["ordinary_preload_clean_slots"] == 1
    assert snap["preload_cached_slots"] == 2
    assert snap["foreground_reclaimable_slots"] == 9
    assert r._prefix_clean_reclaimable_bytes() == 9 * 4096
    assert snap["free_reclaimable_staging_bytes"] == 9 * 4096
    assert snap["capacity_semantics"] == "foreground_clean_reclaimable"
    assert snap["physical_release_credit"] is False


def test_large_mixed_registry_getter_is_constant_space_and_no_scan():
    r = _native_model(slots=256)
    for i in range(90):
        _cache_add(r, ("c%d" % i).encode())
    shared = [_shared_add(r, ("s%d" % i).encode()) for i in range(70)]
    for i in range(20):
        _ordinary_add(r, ("p%d" % i).encode())
    for slot in shared[:13]:
        r._shared_pin(slot)
    r._staging_cache._slots = NoScanDict(r._staging_cache._slots)
    r._shared_cached = NoScanDict(r._shared_cached)
    r._preload_slots = NoScanDict(r._preload_slots)
    for _ in range(5):
        snap = r._prefix_capacity_snapshot()
        assert snap["valid"] is True
        assert snap["shared_cached_pinned_slots"] == 13
        assert snap["foreground_reclaimable_slots"] == 243
        assert r._prefix_clean_reclaimable_bytes() == 243 * 4096
        assert len(snap) <= 24
        assert not any(isinstance(value, (dict, list, set)) for value in snap.values())


def test_shared_nested_pin_and_uncache_preserve_real_owners():
    r = _native_model(slots=4)
    slot = _shared_add(r, b"s")
    assert r._prefix_clean_reclaimable_bytes() == 4 * 4096
    r._shared_pin(slot)
    r._shared_pin(slot)
    assert r._prefix_clean_reclaimable_bytes() == 3 * 4096
    r._shared_unpin(slot)
    assert r._prefix_clean_reclaimable_bytes() == 3 * 4096
    r._shared_uncache(slot)
    assert r._prefix_clean_reclaimable_bytes() == 3 * 4096
    assert slot.copies_inflight == 1
    assert slot.slot_index not in r.staging_pool._free
    r._shared_unpin(slot)
    # Observation alone does not release an uncached slot.
    assert slot.slot_index not in r.staging_pool._free
    r._maybe_release_shared(slot)
    assert r._prefix_clean_reclaimable_bytes() == 4 * 4096


def test_fresh_shared_copy_before_cache_insert_counts_busy_once():
    r = _native_model(slots=4)
    slot = _SharedPreloadSlot(b"s", _reserve(r), None, cached=False)
    r._shared_pin(slot)
    r._shared_pin(slot)
    r._shared_cache_insert(slot)
    snap = r._prefix_capacity_snapshot()
    assert snap["shared_cached_pinned_slots"] == 1
    assert snap["shared_clean_slots"] == 0
    assert snap["foreground_reclaimable_slots"] == 3
    r._shared_unpin(slot)
    r._shared_unpin(slot)
    assert r._prefix_capacity_snapshot()["shared_clean_slots"] == 1


def test_shared_demand_without_copy_is_clean_and_native_evictable():
    r = _native_model(slots=1)
    slot = _shared_add(r, b"s")
    r._preload_refcount[b"s"] = 99
    r._preload_owned[("request", b"s")] = None
    assert r.staging_pool.free_count == 0
    assert r._prefix_clean_reclaimable_bytes() == 4096
    reserved = r._reserve_foreground_slot()
    assert reserved.index == slot.slot_index
    assert b"s" not in r._shared_cached
    assert b"s" not in r._preload_refcount


def test_duplicate_ordinary_hash_counts_physical_slots_and_fifo():
    r = _native_model(slots=4)
    first = _ordinary_add(r, b"p")
    second = _ordinary_add(r, b"p")
    snap = r._prefix_capacity_snapshot()
    assert snap["ordinary_preload_clean_slots"] == 2
    assert snap["preload_cached_slots"] == 2
    assert r._preload_cache_pop(b"p") == (first, None)
    assert r._preload_cache_pop(b"p") == (second, None)
    assert r._preload_cache_pop(b"p") is None
    # Pop transfers ownership; it does not count as a pool release.
    assert r.staging_pool.free_count == 2
    assert r._prefix_clean_reclaimable_bytes() == 2 * 4096


def test_speculative_reservation_does_not_spend_foreground_preload_capacity():
    r = _native_model(slots=1)
    # Ordinary retained preload is reachable in the original non-shared mode.
    # Shared mode selects _evict_one_shared_slot instead of the ordinary FIFO.
    r._share_preload = False
    index = _ordinary_add(r, b"p")
    assert r._prefix_clean_reclaimable_bytes() == 4096
    assert r._reserve_preload_slot() is None
    assert r._preload_slots[b"p"][0][0] == index
    assert r._reserve_foreground_slot().index == index


def test_missing_or_wrong_owner_is_unknown_not_zero():
    r = _native_model()
    r._worker = NS(ident=None)
    assert r._prefix_clean_reclaimable_bytes() is None
    snap = r._prefix_capacity_snapshot()
    assert snap["valid"] is False
    assert snap["free_slots"] is None
    r._worker = NS(ident=threading.get_ident() + 1)
    assert r._prefix_clean_reclaimable_bytes() is None


@pytest.mark.parametrize("field,value", [
    ("_prefix_shared_cached_pinned_slots", -1),
    ("_prefix_shared_cached_pinned_slots", True),
    ("_preload_cached_total", -1),
    ("_preload_cached_total", True),
])
def test_invalid_native_scalar_is_sticky_unknown_without_releasing(field, value):
    r = _native_model(slots=4)
    slot = _shared_add(r, b"s")
    before = tuple(r.staging_pool._free)
    setattr(r, field, value)
    assert r._prefix_clean_reclaimable_bytes() is None
    snap = r._prefix_capacity_snapshot()
    assert snap["valid"] is False
    assert snap["observation_valid"] is False
    assert snap["foreground_reclaimable_slots"] is None
    assert tuple(r.staging_pool._free) == before
    assert r._shared_cached[b"s"] is slot
    # Repairing a scalar is not evidence to reset a detected fault.
    setattr(r, field, 0)
    assert r._prefix_clean_reclaimable_bytes() is None


def test_cache_observation_fault_propagates_unknown_without_native_eviction():
    r = _native_model(slots=4)
    slot = _cache_add(r, b"c")
    r._staging_cache.invalidate_observation("cache proof unavailable")
    assert r._prefix_clean_reclaimable_bytes() is None
    assert r._staging_cache.get(b"c") is slot
    assert r.staging_pool.free_count == 3


def test_nonempty_shared_initialization_cannot_scan_or_backfill():
    r = _native_model(slots=4, initialize=False)
    _shared_add(r, b"s")
    r._shared_cached = NoScanDict(r._shared_cached)
    r._prefix_init_capacity_observation()
    assert r._prefix_clean_reclaimable_bytes() is None
    assert r._prefix_capacity_snapshot()["observation_valid"] is False
    assert len(r._shared_cached) == 1


def test_native_unknown_drain_never_publishes_release_credit():
    r = _native_model(slots=4)
    slot = _shared_add(r, b"s")
    r._shared_pin(slot)
    r._native_drain_unknown = True
    assert r._prefix_clean_reclaimable_bytes() is None
    snap = r._prefix_capacity_snapshot()
    assert snap["valid"] is False
    assert snap["free_reclaimable_staging_bytes"] is None
    assert snap["physical_release_credit"] is False
    assert slot.slot_index not in r.staging_pool._free


def test_reinitialization_cannot_reset_a_known_observation_fault():
    r = _native_model(slots=4)
    r._prefix_invalidate_capacity("persistent native proof failure")
    r._prefix_init_capacity_observation()
    assert r._prefix_clean_reclaimable_bytes() is None
    assert r._prefix_capacity_snapshot()["error"] == "persistent native proof failure"
