"""A per-plan focus, assigned by seed, so Designer coverage does not depend on the model's habits.

Stage 0 showed a frozen Designer writing mostly single-request refunds and exchanges named by
delivery age: no reships, almost no cancellations, no history, and near-identical hypotheses.
Asking for "diverse" tasks in prose did not change that, so each plan is told what to cover.
The focus only steers what the Designer writes; Usami still derives and certifies every answer.
"""
import random

# action -> the rule that decides it, written as the Designer should aim at it
ACTION_RULES = {
    'refund': ('the return window (return_days against how long ago it was delivered)',
               'the approval limit (the refund amount against refund_approval_limit_cents)',
               'a physical return (return_required items are refunded only after the item is received)',
               'a part of a multi-unit line (refund fewer units than were bought)'),
    'exchange': ('the extra charge the customer accepts (max_extra_charge_cents)',
                 'the arrival deadline (arrive_within_days against warehouse delivery days)',
                 'which variant is wanted (attributes such as color or storage, among several in stock)',
                 'stock (the wanted variant is out of stock in every warehouse)'),
    'reship': ('evidence (an item whose evidence_valid is false cannot be reshipped)',
               'stock (the same SKU must be in stock in some warehouse)'),
    'cancel': ('the order status (only an unshipped order can be cancelled; a shipped or delivered one cannot)',),
}
STRUCTURES = (
    'one order with one line',
    'several orders of the same customer, with a similar item in another order as a distractor',
    'one order with several lines, only one of which is requested',
    'two or three requests in one task, on different items',
    'a request on an item that was partly refunded or exchanged before (use history)',
    'two requests on the same line that need disjoint remedies (for example refund one unit, exchange another)',
)
SIDES = ('just inside the rule, so the request should be carried out',
         'just past the rule, so the request should be refused')
NAMING = ('attributes', 'delivery_age', 'price', 'order_date')


def focus_for(seed):
    """The deterministic focus of one plan: action, rule, structure, which side of the rule, naming."""
    rng = random.Random(f'designer-focus:{seed}')
    action = rng.choice(sorted(ACTION_RULES))
    return {'action': action, 'rule': rng.choice(ACTION_RULES[action]), 'structure': rng.choice(STRUCTURES),
            'side': rng.choice(SIDES), 'refer_by': rng.choice(NAMING)}


def focus_text(focus):
    return (f"Focus of this plan (cover it even if the failures above suggest something else):\n"
            f"- the main request is a {focus['action']} request\n"
            f"- the rule it tests: {focus['rule']}\n"
            f"- place the values {focus['side']}\n"
            f"- structure: {focus['structure']}\n"
            f"- the customer names the item by: {focus['refer_by']}")


def focus_hit(focus, plan):
    """Whether a parsed plan has the focus action in some request (a coarse check of obedience)."""
    return any(request.get('action') == focus['action']
               for task in plan.get('tasks', ()) for request in task.get('requests', ()))
