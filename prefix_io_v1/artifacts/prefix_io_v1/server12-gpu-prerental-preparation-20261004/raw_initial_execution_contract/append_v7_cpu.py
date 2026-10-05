"""Append exact initial-execution metadata and fail-fast completed-window gate."""
import hashlib
from pathlib import Path

PREP=Path(__file__).resolve().parent.parent
source=PREP/"runner/strong_native_cost_runner_v6.py"
target=PREP/"runner/strong_native_cost_runner_v7.py"
raw=source.read_bytes()
assert hashlib.sha256(raw).hexdigest()=="be0660712e503cdf496cab6fa7aa48ce206aca78d944803f4862e0f67a540ca7"
text=raw.decode("utf-8")
changes=(
    ("INITIAL_EXECUTION_PRE_CONTEXT = PROMPT_TOKENS - 1", "INITIAL_EXECUTION_PRE_CONTEXT = 496"),
    ("cached_tokens=512, initial_execution_pre_context=511", "cached_tokens=512, initial_execution_pre_context=496"),
    ("initial_scheduled_tokens=1, cached_prompt_tokens_semantics", "initial_scheduled_tokens=16, cached_prompt_tokens_semantics"),
    ('original_cache_hit_rule="local_plus_external_frontend_stats_then_original_full_prompt_remote_last_token_recompute"',
     'original_cache_hit_rule="frontend_local_plus_external_stats512_then_actual_local_block_execution496_prefill16"'),
)
for before,after in changes:
    assert text.count(before)==1,before
    text=text.replace(before,after)
before="    new_tree = ast.parse(new_source)"
after='''    before = "            rows.append(row)"
    after = "            rows.append(row)\\n            _strict_completed_window(row,index)"
    require(new_source.count(before) == 1, "original completed child window fail-fast gate preimage")
    new_source = new_source.replace(before, after)
    new_tree = ast.parse(new_source)'''
assert text.count(before)==1
text=text.replace(before,after)
before="    old.verify_execution_inputs = checked_execution\n    exec(compile(tree"
after='''    old.verify_execution_inputs = checked_execution
    validation = R.load(root, plan["validation_source_ref"],
                        "_strong_actual_completed_window_" + str(time.monotonic_ns()))
    def strict_completed_window(row, index):
        # Consume the unmodified native validator before launching another child.
        # No cost fitting, elapsed metric export or holdout decisions occur here.
        R.require(type(index) is int and 0 <= index < 6 and type(row) is dict,
                  "fixed original completed child window")
        descriptor = plan["cells"][0]
        entry = descriptor["entries"][index // 2]
        condition = entry["arm_order"][index % 2]
        arm = "baseline" if condition == "A" else "action"
        output = row["frontend"]["output"]
        rid = row["request_id"]
        R.require(row.get("condition") == condition and row.get("pair_index") == index // 2 and
                  row.get("prompt_token_ids") == entry["prompt_token_ids"] and row.get("seed") == entry["seed"],
                  "same preregistered original completed window input/order")
        R.require(rid == output["request_id"] == row["capture"]["run_id"] and
                  type(output["native_request_id"]) is str and output["native_request_id"] and
                  output["num_cached_tokens"] == CACHED_TOKENS,
                  "original external/native IDs and separate frontend cache stats")
        R.check_ref(root, plan["validation_source_ref"])
        frames = validation.validate_capture(row["capture"], run_id=rid,
            request_id=output["native_request_id"], output_ids=output["output_token_ids"],
            prompt_tokens=descriptor["prompt_tokens"], measured_offset=descriptor["measured_offset"],
            warmup_offsets=descriptor["warmup_offsets"], cached_tokens=descriptor["cached_prompt_tokens"])
        drain = validation.original_post_shutdown_drain(row["native_post_shutdown"],row["native_tail_assertions"],
            run_id=plan["journal_run_id"],native_source_sha256=plan["native_source_ref"]["sha256"])
        validation.validate_io(row["native_journal"],drain,capture=row["capture"],frames=frames,
            run_id=plan["journal_run_id"],native_source_sha256=plan["native_source_ref"]["sha256"],arm=arm,
            measured_offset=descriptor["measured_offset"],
            physical_bytes=descriptor["transfer_quantum_bytes"]*descriptor["operations"],
            operations=descriptor["operations"],independent_payload=row["independent_payload"])
        row["strict_completed_window_validation"] = dict(status="PASS_NATIVE_COMPLETE_WINDOW_CAPTURE_IO",
            window_index=index,actual_frame_count=len(frames),initial_execution_pre_context=descriptor["cached_prompt_tokens"],
            frontend_cached_prompt_tokens=output["num_cached_tokens"],production_qualified=False,
            performance_claim=False,estimator_run=False,heldout_used_to_fit=False)
        return True
    old._strict_completed_window = strict_completed_window
    exec(compile(tree'''
assert text.count(before)==1
text=text.replace(before,after)
with target.open("xb") as stream:
    stream.write(text.encode("utf-8"))
print(target.name,len(target.read_bytes()),hashlib.sha256(target.read_bytes()).hexdigest())
