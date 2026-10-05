"""Private, evidence-closed pre-construction bridge and ordinary preload hook.

U never installs this wrapper. I retains the original reactor, coordinator,
owner queue, file identities, ages, mandatory progress and release protocol.
Only a table issued by the active package's real raw-GPU verifier is accepted.
"""
from __future__ import annotations
import ast
import inspect
from pathlib import Path
import threading
import time


def require(ok, reason):
    if not ok:
        raise ValueError("FINITE_STARTUP_REJECTED: " + reason)


def ordinary_method_source(original_source):
    """One explicit AST patch; no copy of the reactor engine or lifecycle."""
    tree = ast.parse(original_source)
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "IoReactor")
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "_prefix_stage_decide")
    targets = [node for node in ast.walk(method) if isinstance(node, ast.If)
               and isinstance(node.test, ast.BoolOp)
               and ast.unparse(node.test).startswith("p4.policy.single_file is not None and type(ready) is _ReadyFd")]
    require(len(targets) == 1, "one frozen original ordinary-ready qualification site")
    target = targets[0]
    before = ast.unparse(target.test)
    after = before.replace("p4.policy.single_file is not None", "(p4.policy.single_file is not None or _prefix_private_finite_enabled(self))", 1)
    after = after.replace("(ready.preload_info.total_files == 1)", "(ready.preload_info.total_files >= 1)", 1)
    after = after.replace("(ready.preload_info.file_index == 0)",
                          "(0 <= ready.preload_info.file_index < ready.preload_info.total_files)", 1)
    require(after != before and "total_files >= 1" in after, "exact ordinary-ready field patch preimage")
    target.test = ast.parse(after, mode="eval").body
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    return module, dict(original_method="IoReactor._prefix_stage_decide", modified_if_conditions=1,
        original_native_reserve_checks=True, original_sequence_open_time_hash_preload_info=True,
        original_release_and_completion_unchanged=True, new_queue_or_owner=False,
        finite_issue_deferral_only=True, production_batch_activation=False)


class FiniteStartup:
    def __init__(self, *, native_module, reactor_module, table, issuer, run_id, budget_ns,
                 root, runtime_refs, driver, binding_module, validate_drained, observation_only=False,
                 reserve_covered_cell_signatures=None):
        proof = issuer.qualified_identity(table)
        require(proof is not None and table.production_qualified is True,
                "real private GPU qualification required before any runtime mutation")
        from prefix_io_control.p4_cost_table import CostTable
        from prefix_io_control.p4_bridge import NativeP4Bridge
        require(type(table) is CostTable, "same active exact CostTable type; no second issuer instance")
        require(type(budget_ns) is int and budget_ns > 0, "independent frozen development budget")
        require(type(run_id) is str and run_id, "same original runtime run identity")
        for module in (native_module, reactor_module, issuer, binding_module):
            relative = Path(module.__file__).resolve().relative_to(root).as_posix()
            require(relative in runtime_refs, "actual startup source frozen")
            driver.check_ref(root, runtime_refs[relative])
        self.native, self.reactor, self.table, self.issuer = native_module, reactor_module, table, issuer
        self.run_id, self.budget, self.binding_module = run_id, budget_ns, binding_module
        require(type(observation_only) is bool, "explicit development shadow")
        self.observation_only = observation_only
        binding_module.covered_subset(proof.cells, reserve_covered_cell_signatures,
                                      observation_only=observation_only)
        self.reserve_covered_cell_signatures = reserve_covered_cell_signatures
        self.validate_drained = validate_drained
        self.original_factory = native_module.TransferCoordinator
        self.original_method = reactor_module.IoReactor._prefix_stage_decide
        self.pending = None
        self.binding = None
        self.install_error = None
        self.coordinator = None
        self.lock = threading.Lock()
        tree, self.overlay_proof = ordinary_method_source(Path(reactor_module.__file__).read_text(encoding="utf-8"))
        namespace = dict(vars(reactor_module))
        def qualified(reactor):
            bridge = getattr(reactor, "_prefix_p4_bridge", None)
            return (bridge is not None and type(bridge) is NativeP4Bridge and bridge.policy.table is table
                    and self.issuer.qualified_identity(table) is proof and self.binding is not None)
        namespace["_prefix_private_finite_enabled"] = qualified
        exec(compile(tree, str(Path(__file__).resolve()), "exec", dont_inherit=True), namespace)
        self.new_method = namespace["_prefix_stage_decide"]
        self.factory = self._factory
        reactor_module.IoReactor._prefix_stage_decide = self.new_method
        native_module.TransferCoordinator = self.factory

    def _factory(self, **kwargs):
        from prefix_io_control.p4_bridge import NativeP4Bridge, make_native_bridge
        require(self.coordinator is None, "one original coordinator construction")
        previous = kwargs.get("p4_bridge")
        require(type(previous) is NativeP4Bridge and previous.run_id == self.run_id and
                previous.mode == "interference" and previous.policy.table is None and
                previous.policy.config.internal_step_budget_ns == self.budget and
                previous.policy.single_file is None and not previous._bound,
                "original parsed same-run I config; no live table replacement")
        bridge = make_native_bridge(self.run_id, previous.policy.config, table=self.table)
        kwargs["p4_bridge"] = bridge
        prior_sink = kwargs.get("observation_sink")
        def sink(reactor):
            if prior_sink is not None:
                prior_sink(reactor)
            with self.lock:
                pending = self.pending
            if pending is None or self.binding is not None or self.install_error is not None:
                return
            try:
                require(reactor is self.coordinator.reactor and reactor._worker.ident == threading.get_ident(),
                        "installation only on original native owner thread")
                self.validate_drained(reactor._capture_owner_snapshot())
                bridge._owner()
                self.binding = self.binding_module.CurrentFiniteBinding(bridge, pending[0], pending[1], issuer=self.issuer,
                    observation_only=self.observation_only, host_observer=pending[2],
                    reserve_covered_cell_signatures=self.reserve_covered_cell_signatures)
            except Exception as exc:
                self.install_error = str(exc)[:2000]
        kwargs["observation_sink"] = sink
        self.coordinator = self.original_factory(**kwargs)
        return self.coordinator

    def install_after_original_drain(self, capture, identity, *, host_observer=None):
        require(self.coordinator is not None and self.pending is None and self.binding is None,
                "once before first workload request")
        with self.lock:
            self.pending = (capture, identity, host_observer)
        deadline = time.monotonic() + 2
        while self.binding is None and self.install_error is None and time.monotonic() < deadline:
            # The existing owner-snapshot request causes the unchanged owner
            # loop to process its passive observer; no new control queue.
            self.coordinator.inspect_snapshot(timeout=1)
            time.sleep(.001)
        require(self.binding is not None and self.install_error is None,
                "original owner attachment failed: " + str(self.install_error))
        return dict(owner_thread_install=True, before_first_request=True, drained_before_attach=True,
                    ordinary_preload_native_fields_bound=True, **self.overlay_proof)

    def detach_after_original_shutdown(self):
        require(self.coordinator is None or not self.coordinator.reactor._worker.is_alive(),
                "restore bridge wrappers only after original owner shutdown")
        okay = self.binding is None or self.binding.detach()
        if self.native.TransferCoordinator is self.factory:
            self.native.TransferCoordinator = self.original_factory
        else:
            okay = False
        if self.reactor.IoReactor._prefix_stage_decide is self.new_method:
            self.reactor.IoReactor._prefix_stage_decide = self.original_method
        else:
            okay = False
        self.pending = None
        return okay
