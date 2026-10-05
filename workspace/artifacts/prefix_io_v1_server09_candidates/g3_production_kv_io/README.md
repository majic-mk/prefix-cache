# G3 owner/tail observer candidate

Independent passive connection to the original registered py-kvcache handler. No executor, backend, policy, timer or live hook is installed. Old sources and receipts are unchanged.

`NativeOwnerObserver.verify_owner(worker, handler)` checks the original loaded module/class/code/globals/source graph, the actual vLLM KV singleton, both registry directions, original spec handler and native owners. `observe_transfer`, `observe_get_finished`, `observe_shutdown` call the original handler method once and preserve its return/exception identity. The result wrapper copies the returned batch; it does not make an extra consuming call. Only original shutdown releases, joins or closes. Its successful tail receives one original snapshot and compares real direct native/AIO/accounting fields using the frozen CPU drain validator.

The observer retains bounded immutable scalar records, not worker/handler/Future/job/tensor/event/callback/exception objects. Off reads none of the observation inputs. An observation failure invalidates evidence while preserving the original execution path.

`origin='cpu_fixture'` supports source AST/code replay against CPU actors only. It cannot establish runtime or native I/O truth. `source_bound_runtime` validates additional real loaded-source/Thread conditions but is still only an owner/source/counter fact gate: actual GPU backing, UUID, budget guard, ABI and payload equality require separate root evidence. Every receipt keeps GPU/production/cost/release/effect/byte-exact qualifications false; native I/O qualification is not asserted by this helper. GPU timing is `None`, frame I/O attribution is `UNKNOWN`.

The current module requires unchanged original instance methods. It does not install a monkeypatch or permit an unknown instance wrapper. A future hook installer needs its own source binding contract; root can use the observer directly at original handler boundaries for a correctness pilot.

Local CPU command:

```powershell
& 'C:/Users/mamengkui/AppData/Roaming/uv/python/cpython-3.12.14-windows-x86_64-none/python.exe' -B -I -S artifacts/prefix_io_v1_server09_candidates/g3_production_kv_io/test_g3_native_owner_observer.py -v
```

Server CPU command (project sources include both mirrored roots):

```sh
.venv/bin/python -B -I -S <deployment>/test_g3_native_owner_observer.py --source-root <ROOT> --support-root <ROOT> --drain-source <frozen-drain-artifact>/g2_native_drain_provider.py --connector-source <frozen-runtime-artifact>/p4_runtime_scalar_connector.py --adapter-source <frozen-frame-artifact>/p4_full_step_frame_adapter.py -v
```

The four physical stage fields are kept separately from logical KV bytes. AIO totals also include open/close. Stored staging is normally retained in cache, so an immediate reload cannot be assumed to touch SSD; the pilot needs a genuine file read, normal closure and independently verified original KV payload restoration. No I/J or cost strategy is enabled.
