"""Pure CPU, bounded extension of the existing native diagnostic observer.

This module imports no backend, holds no resource, and grants no dispatch or
release permit. make_observer_class reuses the original NativeFlushProbe's
opt-in lifecycle callbacks. Synthetic CPU fixtures do not prove GPU completion.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ReleaseView:
    run_id: str
    block: int
    source_generation: int | None
    current_generation: int | None
    protecting_jobs: tuple[int, ...]
    native_retired_jobs: tuple[int, ...]
    active_refs: int | None
    native_reusable: bool | None
    reusable_now_bytes: int
    potential_bytes: int
    reason: str


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid " + name)
    return value


class BoundedReleaseAdapter:
    """Consume original owner snapshots; never infer release from a Future.

    A normal flush cause proves a native fence dependency, not a measured wait
    duration or its effect on token latency. Missing sampled source membership
    remains unknown. There is intentionally no conversion to policy Resource,
    NativePublication capability, allocator calls, or dispatch acceptance.
    """
    MAX_RESOURCES = 8
    MAX_PARENTS = 32
    MAX_RECORDS = 96
    NATURAL_RESOURCE_CAUSES = frozenset(("new_allocation", "restore_destination"))

    def __init__(self, run_id, block_bytes, *, worker_completion_verified=False):
        if type(run_id) is not str or not run_id:
            raise ValueError("invalid run identity")
        self.run_id = run_id
        self.block_bytes = _integer(block_bytes, "actual single-group block bytes", 1)
        if type(worker_completion_verified) is not bool:
            raise ValueError("worker completion evidence must be explicit")
        self.worker_completion_verified = worker_completion_verified
        self._resources = {}
        self._records = deque(maxlen=self.MAX_RECORDS)
        self.omitted_resources = 0

    def observe_flush(self, event):
        if event.get("reason") not in self.NATURAL_RESOURCE_CAUSES:
            return
        if event.get("parents_truncated") is not False:
            return
        parents = event.get("parents")
        conflicts = event.get("conflict_blocks")
        if type(parents) is not list or type(conflicts) is not list:
            raise ValueError("invalid native flush snapshot")
        if len(parents) > self.MAX_PARENTS or len(conflicts) > self.MAX_RESOURCES:
            raise ValueError("native flush observation bound exceeded")
        parent_map = {}
        for parent in parents:
            jid = _integer(parent.get("job_id"), "parent job")
            if jid in parent_map:
                raise ValueError("duplicate native parent")
            parent_map[jid] = parent
        for conflict in conflicts:
            bid = _integer(conflict.get("block"), "block")
            pids = conflict.get("parent_ids")
            if type(pids) is not list or not pids or len(pids) > self.MAX_PARENTS:
                continue
            pids = tuple(_integer(x, "protecting job") for x in pids)
            if len(set(pids)) != len(pids):
                raise ValueError("ambiguous native protector identity")
            if conflict.get("parents_truncated") is not False:
                continue
            source_states = []
            source_complete = True
            for jid in pids:
                parent = parent_map.get(jid)
                if (parent is None or parent.get("is_store") is not True or
                        type(parent.get("pending_worker_acks")) is not int or
                        parent["pending_worker_acks"] <= 0 or
                        parent.get("source_sample_known") is not True):
                    source_complete = False
                    break
                states = parent.get("source_blocks")
                if type(states) is not list or len(states) > self.MAX_RESOURCES:
                    source_complete = False
                    break
                matching = [x for x in states if x.get("block") == bid]
                if len(matching) != 1:
                    source_complete = False
                    break
                source_states.append(matching[0])
            # Each protecting parent must explicitly name this same generation.
            # A job's first/last sample cannot prove membership of omitted blocks.
            generations = {s.get("source_generation") for s in source_states}
            gen = next(iter(generations)) if len(generations) == 1 else None
            if type(gen) is not int or gen < 1:
                source_complete = False
                gen = None
            key = (bid, gen)
            if key not in self._resources and len(self._resources) >= self.MAX_RESOURCES:
                self.omitted_resources += 1
                continue
            old = self._resources.get(key)
            # A second witnessed closure replaces the diagnostic row only if it
            # describes the same complete protecting set, otherwise fail closed.
            if old and old["parents"] != frozenset(pids):
                source_complete = False
            row = dict(block=bid, generation=gen, parents=frozenset(pids), retired=set(),
                       source_complete=source_complete, state=None)
            self._resources[key] = row
            if source_states:
                row["state"] = dict(source_states[0])
            self._publish(row)

    def observe_native_retirement(self, event):
        if event.get("kind") != "native_parent_retired":
            raise ValueError("only original owner native retirement is accepted")
        jid = _integer(event.get("job_id"), "retired parent job")
        states = event.get("source_blocks")
        if type(states) is not list or len(states) > self.MAX_RESOURCES:
            raise ValueError("native retirement source bound exceeded")
        for row in self._resources.values():
            if jid not in row["parents"]:
                continue
            matching = [s for s in states if s.get("block") == row["block"] and
                        s.get("source_generation") == row["generation"]]
            if len(matching) != 1:
                row["source_complete"] = False
            else:
                row["retired"].add(jid)
                row["state"] = dict(matching[0])
            self._publish(row)

    def _publish(self, row):
        state = row["state"] or {}
        refs = state.get("active_refs")
        gen = state.get("generation")
        now = potential = 0
        reusable = None
        reason = "unknown_owner_or_parent_closure"
        if (row["source_complete"] and state.get("owner_known") is True and
                type(refs) is int and refs >= 0 and type(gen) is int and gen > 0):
            if gen != row["generation"] or state.get("same_generation") is not True:
                reason = "generation_changed"
            elif state.get("parents_truncated") is not False:
                reason = "truncated_native_protectors"
            else:
                pending = state.get("protecting_jobs")
                count = state.get("protecting_jobs_count")
                if (type(pending) is not list or type(count) is not int or count < 0 or
                        count != len(pending) or len(pending) > self.MAX_PARENTS or
                        len(set(pending)) != len(pending) or
                        any(type(p) is not int or p < 0 for p in pending)):
                    reason = "invalid_native_protectors"
                elif not set(pending).issubset(row["parents"] - row["retired"]):
                    reason = "unobserved_native_protector"
                elif refs > 0:
                    reusable = False
                    reason = "active_native_reference"
                elif pending or row["retired"] != set(row["parents"]):
                    reusable = False
                    potential = self.block_bytes
                    reason = "awaiting_native_parent_retirement"
                elif state.get("is_null") is not False:
                    reusable = False
                    reason = "null_or_unknown_block"
                elif state.get("free_queue_linked") is not True:
                    reusable = False
                    reason = "not_on_native_free_queue"
                elif not self.worker_completion_verified:
                    reason = "worker_fence_semantics_not_verified"
                else:
                    # This is a diagnostic observation after all original worker
                    # acks and free-list updates, never an authority to recycle.
                    reusable = True
                    now = self.block_bytes
                    reason = "observed_native_allocator_availability"
        view = ReleaseView(self.run_id, row["block"], row["generation"], gen,
                           tuple(sorted(row["parents"])), tuple(sorted(row["retired"])),
                           refs if type(refs) is int else None, reusable, now, potential, reason)
        row["view"] = view
        self._records.append(view)

    def export(self):
        from dataclasses import asdict
        return dict(run_id=self.run_id, resources=[asdict(r["view"]) for r in self._resources.values()],
                    records=[asdict(r) for r in self._records], omitted_resources=self.omitted_resources,
                    worker_completion_semantics_verified=self.worker_completion_verified,
                    production_release_capability=False,
                    scope="bounded diagnostic only; no policy publication or permit")


def make_observer_class(native_flush_probe_class):
    """Subclass the already installed native observer without importing vLLM.

    The base class's original safe/install/uninstall hooks retain ownership,
    callback order, failure containment, native queues and method return values.
    Pass the actual SHA-bound NativeFlushProbe class, not a backend replacement.
    """
    class NativeFlushReleaseObserver(native_flush_probe_class):
        def __init__(self, run_id, block_bytes, *, worker_completion_verified=False):
            super().__init__(run_id)
            self.release_adapter = BoundedReleaseAdapter(
                run_id, block_bytes, worker_completion_verified=worker_completion_verified)

        def flush(self, scheduler, reason, req_id=None, block_ids=None, job_ids=None):
            before = len(self.causes)
            super().flush(scheduler, reason, req_id, block_ids, job_ids)
            if len(self.causes) > before:
                self.release_adapter.observe_flush(self.causes[-1])

        def completed(self, scheduler, output):
            before = self.retired_jobs
            super().completed(scheduler, output)
            count = self.retired_jobs - before
            # Base bounded records append exactly one record per observed native
            # retirement. It holds at most MAX_JOBS, less than its MAX_RECORDS.
            if count:
                for record in list(self.records)[-count:]:
                    self.release_adapter.observe_native_retirement(record)

        def export(self):
            result = super().export()
            result["release_observation"] = self.release_adapter.export()
            # Parent base publication remains conservative and unchanged.
            return result
    return NativeFlushReleaseObserver
