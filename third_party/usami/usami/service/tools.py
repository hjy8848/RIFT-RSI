"""Schemas from the core protocol; projection only, never business decisions."""
from __future__ import annotations

from usami.protocol import PARAMETERS, Operation
from usami.service.models import ServiceError, strict_json

DESCRIPTIONS = {
    "invite_member": "Owner provisions a known directory user in this organization with a pending membership. resource_id is the organization ID. Duplicate pending invitations with the same role are idempotent.",
    "activate_member": "Owner activates a pending or inactive non-owner membership. Requires an active subscription and a free seat. This is administrator provisioning, not acting as the invited user.",
    "deactivate_member": "Owner suspends an active non-owner membership, reducing the active seat requirement.",
    "remove_member": "Owner removes a pending or inactive non-owner membership. Active members must first be deactivated.",
    "pay_invoice": "Simulated full invoice payment. card_declined records failure; other supported methods succeed. Exact invoice/key/method retry returns the original result; changing method needs a fresh key. Never charges real money.",
    "issue_renewal": "Create or retrieve the linked next 30-day renewal invoice for this subscription. Returns invoice ID and period. Active cycles start at current end; expired cycles start at frozen business time.",
    "renew_subscription": "Extend an active subscription with its paid, unrefunded next-cycle invoice. invoice_id must link to this subscription and the correct period.",
    "reactivate_subscription": "Restore an expired subscription with its paid, unrefunded renewal invoice starting at frozen business time.",
    "expire_subscription": "Record expiration only when the service period has ended at frozen business time; disables automatic renewal.",
    "get_customer_context": "Get the fixed authenticated customer's own profile and organization memberships. No identity argument.",
    "get_organization": "Read an organization. On success also discover related invoice IDs; subscription ID is in the entity.",
    "list_members": "Read organization memberships and related public user profiles; resource_id must equal org_id.",
    "get_subscription": "Read the actual subscription, plan, seats, renewal and billing period.",
    "get_invoice": "Read a specific invoice, paid amount, original payment method and refund history. Permissions are enforced.",
    "change_plan": "Change the subscription plan requested by the customer; the current seats must fit the target plan.",
    "change_seats": "Change subscription seats. Respect plan capacity and active member count.",
    "cancel_renewal": "Disable automatic renewal without changing the current service period.",
    "refund_invoice": "Refund an invoice in full to its original payment method, within the reference refund window.",
    "change_member_role": "Change a member's role as requested. Only the owner can manage roles; owner transfer is unsupported.",
}


def tool_schemas(names, rules_version=None):
    tools = []
    for name in names:
        props, required = {}, []
        if name != "get_customer_context":
            params = PARAMETERS[name].model_json_schema()
            props = {"org_id": {"type": "string", "description": "Target organization ID"},
                     "resource_id": {"type": "string", "description": "Target organization, subscription, invoice or membership ID"},
                     **params.get("properties", {})}
            required = ["org_id", "resource_id", *params.get("required", [])]
            if name == "refund_invoice" and rules_version != "saas-rules-0.2.0":
                props.pop("idempotency_key", None)
        description = DESCRIPTIONS[name]
        if name == "refund_invoice" and rules_version == "saas-rules-0.2.0":
            description = "Refund a positive amount no greater than the remaining refundable balance within the refund window to the original method. idempotency_key is required; exact retries do not refund twice."
            required.append("idempotency_key")
        tools.append({"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": required, "additionalProperties": False}}})
    return tools


def parse_operation(name, arguments, rules_version=None):
    try:
        data = strict_json(arguments)
        if not isinstance(data, dict):
            raise ValueError("arguments must be an object")
        if name == "get_customer_context":
            if data:
                raise ValueError("identity is fixed; this tool takes no arguments")
            return None
        shape = PARAMETERS[name].model_json_schema().get("properties", {})
        if name == "refund_invoice" and rules_version != "saas-rules-0.2.0":
            shape.pop("idempotency_key", None)
        if set(data) - {"org_id", "resource_id", *shape}:
            raise ValueError("unexpected argument")
        return Operation(name=name, org_id=data["org_id"], resource_id=data["resource_id"],
                         params={k: v for k, v in data.items() if k not in {"org_id", "resource_id"}})
    except (ValueError, KeyError, TypeError):
        raise ServiceError("INVALID_TOOL_ARGUMENTS", "Arguments must match the advertised tool schema; identity and authorization cannot be supplied") from None


def customer_context(world, identity):
    memberships = [m for m in world.memberships.values() if m.user_id == identity]
    return {"customer": world.users[identity].model_dump(mode="json"),
            "memberships": [m.model_dump(mode="json") for m in memberships],
            "organizations": [{"id": world.organizations[m.org_id].id, "name": world.organizations[m.org_id].name}
                              for m in memberships]}


def discovery(world, op, result):
    # Additional ID/name projections are gated by an actual successful core query.
    if not result["success"]:
        return {}
    if world.rules.version == "saas-rules-0.2.0":
        return {}  # All v0.2 query projections are independently specified core data.
    if op.name == "get_organization":
        return {"invoice_ids": [i.id for i in world.invoices.values() if i.org_id == op.org_id]}
    if op.name == "list_members":
        members = result["data"]["memberships"]
        return {"users": [world.users[m["user_id"]].model_dump(mode="json") for m in members]}
    return {}
