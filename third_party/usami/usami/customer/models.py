"""Untrusted choices made by a Customer LLM, never role or identity grants."""
from typing import List, Literal

from pydantic import Field, model_validator

from usami.protocol import Operation, PARAMETERS, QUERIES
from usami.world.models import Model


class CustomerAction(Model):
    type: Literal["send_message", "stop"]
    message: str = ""
    requested_operations: List[Operation] = Field(default_factory=list)
    reason: str = ""

    @model_validator(mode="after")
    def action_shape(self):
        if self.type == "stop":
            if self.message or self.requested_operations:
                raise ValueError("stop cannot send a message or request operations")
        elif not self.message.strip() and not self.requested_operations:
            raise ValueError("send_message needs context or a structured business request")
        if any(op.is_query for op in self.requested_operations):
            raise ValueError("requested_operations describes writes; ask queries in the message")
        return self


def action_schema(tools, rules_version=None):
    """Describe the Customer's nested Operation shape using core parameter models."""
    target_types = {name: "membership ID" for name in ("change_member_role", "activate_member", "deactivate_member", "remove_member")}
    target_types.update(refund_invoice="invoice ID", pay_invoice="invoice ID", invite_member="organization ID")
    operations = []
    for name in tools:
        if name not in PARAMETERS or name in QUERIES:
            continue
        params = PARAMETERS[name].model_json_schema()
        if name == "refund_invoice":
            if rules_version == "saas-rules-0.2.0": params["required"].append("idempotency_key")
            else: params["properties"].pop("idempotency_key", None)
        operations.append({"type": "object", "additionalProperties": False,
            "required": ["name", "org_id", "resource_id", "params"], "properties": {
                "name": {"const": name}, "org_id": {"type": "string", "description": "Observed organization ID"},
                "resource_id": {"type": "string", "description": "Observed " + target_types.get(name, "subscription ID")},
                "params": params}})
    schema = CustomerAction.model_json_schema()
    schema.pop("$defs", None)
    schema["properties"]["requested_operations"]["items"] = {"anyOf": operations} if operations else False
    return schema
