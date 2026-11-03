"""Designer chat prompt: rules of the world, training failures with their hidden goals, examples."""
import json
import random

from rift.designer.format import plan_from_drafts

SYSTEM = """You design training tasks for a customer-service agent that works in a simulated online store.
The agent reads the customer's request, looks up orders with tools and then refunds, exchanges,
reships, cancels, or correctly refuses. You write one store World and the customers' requests in it.
You do NOT say whether a request should be carried out: the environment derives the correct answer
from its business rules and verifies every task. Tasks that cannot be verified are discarded.

Business rules the environment applies:
- refund: only for delivered orders, within the item's return_days of delivery; a refund larger than
  refund_approval_limit_cents is refused (needs approval). Items with return_required=true are refunded
  only after a physical return, which the environment receives one day later.
- exchange: to a different SKU of the same family; the customer pays the price difference plus the
  delivery fee of the fastest warehouse that has stock. Not for return_required items.
- reship: the same SKU again, free, only with valid evidence (evidence_valid) and stock.
- cancel: only unshipped orders, always the whole order.
- Shipped (not yet delivered) orders cannot be refunded, exchanged or reshipped.
- A customer with max_extra_charge_cents or arrive_within_days declines offers outside those limits.
- history lists actions that already happened before the task (for example an earlier partial refund).
- Every requested item must be identifiable from the request alone: refer_by chooses how the customer
  names it (attributes, delivery_age, price, order_date) and that description must match exactly one
  of the customer's order lines.

Good tasks train the agent's weaknesses shown in the failures, mix requests that should be carried out
with requests that should be refused, place values near the rule boundaries, and are not copies of
the failures or of the examples. Answer with exactly one JSON object and nothing else."""


def failure_text(case):
    """One training failure, including its hidden goals (training split only)."""
    goals = '\n'.join(f"  - {goal['item']}: customer asked to {goal['action']} {goal['quantity']}; "
                      f"correct answer: {goal['answer']}" for goal in case['goals'])
    steps = ', '.join(case.get('agent_steps', ())) or 'none'
    return (f"Request: {case['request']}\nHidden goals:\n{goals}\n"
            f"Agent success rate: {case['success_rate']:.2f}\nAgent steps in a failed attempt: {steps}\n"
            f"Agent's last message: {case.get('last_message') or '(none)'}")


def example_plans(seed, count=2):
    """Worked examples from the generator, rotated by ``seed`` so they are not copied verbatim."""
    from rift.benchmark.generator import draft_instance
    rng = random.Random(seed)
    families = rng.sample(['F1', 'F2', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11'], count)
    plans = []
    for family in families:
        data, task = draft_instance(family, rng.randrange(10**6))
        plans.append(plan_from_drafts(data, task, f'example of a generated {family} task'))
    return plans


def designer_messages(failures, *, seed, examples=2, feedback=()):
    """Chat messages for one Designer generation."""
    parts = ['Training failures of the agent:']
    parts += [f'[{index + 1}]\n' + failure_text(case) for index, case in enumerate(failures)] or ['(none yet)']
    if feedback:
        parts.append('Your earlier plans that the environment rejected, with the reason:')
        parts += [f'- {row}' for row in feedback]
    parts.append('Examples of valid plans:')
    parts += [json.dumps(plan.model_dump(mode='json', exclude_defaults=True), ensure_ascii=False)
              for plan in example_plans(seed, examples)]
    parts.append('Write one new plan as a single JSON object.')
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': '\n\n'.join(parts)}]
