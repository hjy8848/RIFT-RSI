"""Explicit offline baselines, never used by real-model execution.

reference-proof is a software competence check with evaluator witness access.
one-step uses evaluator state/domain and takes at most one legal goal-improving
write; it cannot solve prerequisite chains. Neither is fair LLM performance.
"""
import json
from usami.world.rules import ReferenceRules
from usami import generate_world


def reply(content=None, operation=None, index=0):
    message = {"role": "assistant", "content": content}
    if operation:
        message["tool_calls"] = [{"id": f"baseline_{index}", "type": "function", "function": {"name": operation.name, "arguments": json.dumps({"org_id": operation.org_id, "resource_id": operation.resource_id, **operation.params})}}]
    return {"choices": [{"finish_reason": "tool_calls" if operation else "stop", "message": message}]}


class BaselineClient:
    def __init__(self, manifest, instance, mode):
        self.settings = manifest.service_model; self.index = 0
        task = instance.task; self.final = "No legal one-step solution was completed."
        self.operations = []
        if mode == "reference-proof":
            if task.policy_certificate:
                self.final = json.dumps({"disposition": "refused", "policy_code": task.policy_certificate.error_code, "explanation": task.policy_certificate.reason})
            else:
                self.operations = task.witnesses[0].operations; self.final = "Verified reference route completed."
        elif mode == "one-step":
            world = generate_world(instance.seed, instance.generator or manifest.generator)
            candidates = []
            before = sum(p.satisfied(world) for a in task.goal.atoms for p in a.predicates)
            for op in task.capabilities:
                dec = ReferenceRules().decide(world, task.identity, op)
                if dec.allowed:
                    improvement = sum(p.satisfied(dec.expected_state) for a in task.goal.atoms for p in a.predicates) - before
                    if improvement > 0 or task.goal.required_evidence == op: candidates.append((improvement, op))
            if candidates:
                self.operations = [max(candidates, key=lambda item: item[0])[1]]; self.final = "One policy-allowed action was executed; other work may remain."
        else: raise ValueError("unknown offline baseline")

    def complete(self, request):
        index = self.index; self.index += 1
        if index < len(self.operations): return reply(operation=self.operations[index], index=index)
        return reply(self.final)
