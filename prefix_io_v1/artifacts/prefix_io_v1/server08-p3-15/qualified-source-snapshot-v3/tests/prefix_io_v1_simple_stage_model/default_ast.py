"""Restore only the explicitly optional P315 driver blocks for historical checks."""
import ast

def strip_simple(tree):
    class Strip(ast.NodeTransformer):
        def visit_Expr(self, n):
            if isinstance(n.value, ast.Call) and any(isinstance(a,ast.Constant) and a.value=="--simple-stage-config" for a in n.value.args):
                return None
            return self.generic_visit(n)
        def visit_Assign(self,n):
            if len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="simple_extra":
                assert isinstance(n.value,ast.Constant) and n.value.value is None
                return None
            return self.generic_visit(n)
        def visit_If(self,n):
            if (isinstance(n.test,ast.Compare) and
                ((isinstance(n.test.left,ast.Name) and n.test.left.id=="simple_extra") or
                 (isinstance(n.test.left,ast.Attribute) and n.test.left.attr=="simple_stage_config"))):
                assert not n.orelse, "optional policy must not add a default else"
                return None
            return self.generic_visit(n)
    return Strip().visit(tree)
