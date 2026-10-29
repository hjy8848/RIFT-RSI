"""Observe the unmodified reference function; distinguish visits from vetoes."""
from __future__ import annotations

import ast
import inspect
import sys
from collections import Counter
from pathlib import Path

from usami.protocol import PARAMETERS
from usami.world.rules import ReferenceRules, check_invariants


class RuleObserver:
    def __init__(self):
        self.source_path = Path(inspect.getsourcefile(ReferenceRules))
        source = self.source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "decide")
        self.guards = []

        def walk(nodes, scope):
            for node in nodes:
                if not isinstance(node, ast.If):
                    continue
                body_scope = scope
                if (isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
                        and node.test.left.id == "name" and len(node.test.comparators) == 1):
                    value = node.test.comparators[0]
                    if isinstance(node.test.ops[0], ast.Eq) and isinstance(value, ast.Constant):
                        body_scope = scope & {value.value}
                    elif isinstance(node.test.ops[0], ast.In) and isinstance(value, ast.Set):
                        body_scope = scope & {v.value for v in value.elts}
                returns = [n for n in node.body if isinstance(n, ast.Return)]
                for ret in returns:
                    call = ret.value
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "reject":
                        self.guards.append({"id": f"g{len(self.guards) + 1:02d}", "line": node.lineno,
                                            "return_line": ret.lineno, "condition": ast.unparse(node.test),
                                            "error_code": call.args[0].value,
                                            "operations": sorted(body_scope)})
                walk(node.body, body_scope)
                # elif branches are nested in orelse. Only name selectors restrict scope.
                walk(node.orelse, scope - body_scope if body_scope != scope else scope)

        walk(fn.body, set(PARAMETERS))
        self.invariant_codes = sorted({n.args[0].value for n in ast.walk(tree)
                                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                                      and n.func.id == "bad" and isinstance(n.args[0], ast.Constant)})
        self._expressions = {g["id"]: compile(ast.parse(g["condition"], mode="eval"), str(self.source_path), "eval")
                             for g in self.guards}
        self._code = ReferenceRules.decide.__code__
        self._rules = ReferenceRules()

    def decide(self, world, actor, op, event_id):
        lines = set()

        def tracer(frame, event, arg):
            if frame.f_code is self._code:
                if event == "line":
                    lines.add(frame.f_lineno)
                return tracer
            return None

        previous = sys.gettrace()
        try:
            sys.settrace(tracer)
            decision = self._rules.decide(world, actor, op, event_id)
        finally:
            sys.settrace(previous)
        observations = [(g["id"], g["return_line"] in lines) for g in self.guards if g["line"] in lines]
        vetoes = [gid for gid, veto in observations if veto]
        if len(vetoes) != (0 if decision.allowed else 1):
            raise ValueError("reference source no longer matches guard observer")
        if vetoes and next(g["error_code"] for g in self.guards if g["id"] == vetoes[0]) != decision.error_code:
            raise ValueError("observed veto differs from actual reference result")
        return decision, observations

    def potential_vetoes(self, world, actor, op):
        """Derived predicates, including masked ones. Never counts as exercised coverage.

        Evaluate the actual AST predicates with prerequisite locals reconstructed.
        Missing prerequisites yield Unknown; no state changes or policy decisions.
        """
        member = next((m for m in world.memberships.values() if m.user_id == actor
                       and m.org_id == op.org_id and m.status == "active"), None)
        collection = (world.organizations if op.name in {"get_organization", "list_members"} else
                      world.invoices if op.name in {"refund_invoice", "get_invoice"} else
                      world.memberships if op.name == "change_member_role" else world.subscriptions)
        resource = collection.get(op.resource_id)
        local = dict(world=world, actor_id=actor, operation=op, name=op.name, member=member, resource=resource,
                     resource_org=(resource.id if op.name in {"get_organization", "list_members"}
                                   else getattr(resource, "org_id", None)),
                     target=world.rules.plans.get(op.params.get("plan_id")), seats=op.params.get("seats"),
                     active_count=sum(m.org_id == op.org_id and m.status == "active" for m in world.memberships.values()))
        values = {}
        for guard in self.guards:
            if op.name not in guard["operations"]:
                continue
            try:
                # Trusted expressions read from local reference source, never user input.
                values[guard["id"]] = bool(eval(self._expressions[guard["id"]], self._rules.decide.__globals__, local))
            except (AttributeError, TypeError, KeyError):
                values[guard["id"]] = None
        return values


def rule_statistics(observer, records):
    operation_counts = {op: Counter() for op in sorted(PARAMETERS)}
    matrix, errors, visited_sets, potential_sets = {}, Counter(), Counter(), Counter()
    for record in records:
        op = record["operation"]
        outcome = "allowed" if record["allowed"] else "rejected"
        operation_counts[op][outcome] += 1
        operation_counts[op]["policy_allowed" if record["policy_allowed"] else "policy_rejected"] += 1
        if record["error_code"]:
            errors[record["error_code"]] += 1
        for gid, veto in record["guards"]:
            row = matrix.setdefault((op, gid), Counter())
            row[outcome] += 1
            row["veto" if veto else "pass"] += 1
        # Exclude common existence/integrity gates from business combination counts.
        relevant = [gid for gid, _ in record["guards"] if int(gid[1:]) > 7]
        visited_sets["+".join(relevant)] += 1
        false = [gid for gid, value in record["potential"].items() if value is True]
        if len(false) > 1:
            potential_sets["+".join(false)] += 1
    guards = []
    for guard in observer.guards:
        rows = [v for (op, gid), v in matrix.items() if gid == guard["id"]]
        counts = sum(rows, Counter())
        guards.append(dict(guard, visited=sum(v["pass"] + v["veto"] for v in rows),
                           passed=counts["pass"], vetoed=counts["veto"]))
    return {"implemented_guards": guards, "invariant_error_codes": observer.invariant_codes,
            "operations": {op: dict(c) for op, c in operation_counts.items()}, "rejection_codes": dict(sorted(errors.items())),
            "operation_rule_matrix": [dict(operation=op, guard=gid, **c) for (op, gid), c in sorted(matrix.items())],
            "visited_business_guard_sets": dict(sorted(visited_sets.items())),
            "derived_multiple_veto_sets": dict(sorted(potential_sets.items())),
            "never_visited": [g["id"] for g in guards if not g["visited"]],
            "never_vetoed": [g["id"] for g in guards if not g["vetoed"]]}
