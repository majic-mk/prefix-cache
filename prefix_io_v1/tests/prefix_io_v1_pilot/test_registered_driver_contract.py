"""Static regression of the optional model-driver facility, before GPU startup."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def trees():
 old=ast.parse((ROOT/"artifacts/prefix_io_v1/server08-p3-14/run_concurrent_pilot-before.py").read_text())
 new=ast.parse((ROOT/"experiments/prefix_io_v1/scripts/run_concurrent_pilot.py").read_text())
 from tests.prefix_io_v1_simple_stage_model.default_ast import strip_simple
 new=strip_simple(new)
 return old,new
def main(tree):
 return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
def text(node):return ast.unparse(node)
def test_default_driver_restores_original_ast():
 old,new=trees();m=main(new);body=[];old_source=next(n for n in main(old).body if 'unexpected published source' in text(n))
 for n in m.body:
  s=text(n)
  if s=="ap.add_argument('--storage-registration', type=Path)" or s=="source_registration = None":continue
  if isinstance(n,ast.If) and s.startswith("if a.storage_registration is None:"):
   assert len(n.body)==1
   assert ast.dump(n.test)==ast.dump(ast.parse("a.storage_registration is None",mode="eval").body)
   assert ast.dump(n.body[0],include_attributes=False)==ast.dump(old_source,include_attributes=False)
   body.append(n.body[0]);continue
  if isinstance(n,ast.If) and s.startswith("if source_registration is not None:"):continue
  body.append(n)
 m.body=body
 assert ast.dump(new,include_attributes=False)==ast.dump(old,include_attributes=False)
def test_source_validation_and_gate_binding_precede_clone_and_runtime():
 _,tree=trees();body=main(tree).body
 source_gate=next(i for i,n in enumerate(body) if isinstance(n,ast.If) and text(n).startswith("if a.storage_registration is None:"))
 clone=next(i for i,n in enumerate(body) if "inventory = clone_private_storage(" in text(n))
 runtime=next(i for i,n in enumerate(body) if "base.configure_runtime_environment()" in text(n))
 assert source_gate < clone < runtime
 block=text(body[source_gate])
 assert 'root / permit' in block and 'reference_plan' in block
 assert 'storage_path' in block and '== source' in block
 assert 'source_registration' in block and "['gpu_uuid']" in block
 assert not any(isinstance(n,ast.Try) for n in ast.walk(body[source_gate]))
def test_native_cache_clone_and_engine_configuration_unchanged():
 old,new=trees()
 for prefix in ["inventory = clone_private_storage(", "config = dict(engine_config", "sampling = dict(base.SAMPLING"]:
  def pick(tree):
   return next(n for n in ast.walk(main(tree)) if isinstance(n,ast.Assign) and text(n).startswith(prefix))
  assert ast.dump(pick(old),include_attributes=False)==ast.dump(pick(new),include_attributes=False)
