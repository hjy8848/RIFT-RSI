"""Training failures in the form the Designer reads: request, hidden goals and the agent's steps.

Only TRAIN-split tasks may be turned into failure cases; held-out tasks never reach here.
"""


def _describe(world, line_id, intent):
    from usami_v2.scenarios.tasks import _line_description, _referenced_line_description
    if intent is not None:
        return _referenced_line_description(world, line_id, intent.reference)
    return _line_description(world, line_id)


def goal_rows(world, task):
    """The hidden goals of a task with their derived answers."""
    if getattr(task, 'protocol', None) == 'retail-task-v1':
        return retail_goal_rows(world, task)
    intents = {row.goal_id: row for row in getattr(task.request_bundle, 'requests', ())}
    rows = []
    for goal in task.remedy_goals:
        rows.append({'item': _describe(world, goal.line_id, intents.get(goal.goal_id)),
                     'action': '/'.join(goal.allowed_remedies), 'quantity': goal.quantity, 'answer': 'carry it out'})
    for goal in task.cancellation_goals:
        rows.append({'item': _describe(world, goal.line_ids[0], intents.get(goal.goal_id)),
                     'action': 'cancel', 'quantity': len(goal.line_ids), 'answer': 'carry it out'})
    for goal in getattr(task, 'refusal_goals', ()):
        intent = intents.get(goal.goal_id)
        rows.append({'item': _describe(world, goal.line_ids[0], intent), 'action': goal.requested_action,
                     'quantity': intent.quantity if intent is not None else 1,
                     'answer': f'refuse ({goal.reason_code})'})
    return rows


def retail_goal_rows(world, task):
    from usami_v2.scenarios.retail_tasks import _item, _order_phrase, delegated, gold_operations
    current, rows = delegated(world, task.customer_id), []
    for goal in task.goals:
        isolated = task.model_copy(update={'world_hash': current.state_hash, 'goals': (goal,), 'ask_total': False})
        ops, next_world = gold_operations(current, isolated)
        item = (_order_phrase(current, task.customer_id, goal.order_id)
                if goal.order_id else 'default profile address')
        if goal.action == 'change_items':
            item = ' and '.join(_item(current, wish.line_id) for wish in goal.wishes)
        if not ops:
            answer = 'leave unchanged; requested operation is not permitted or feasible'
        elif ops[0].name == 'cancel_order':
            answer = 'cancel the whole order'
        elif ops[0].name in ('modify_order_address', 'update_default_address'):
            p = ops[0].params
            answer = 'change address to %s, %s' % (p['street'], p['city'])
        else:
            op = ops[0]
            chosen = [current.skus[row['target_sku_id']] for row in op.params['changes']]
            names = [' '.join([*sku.attributes.values(), sku.product_family]) for sku in chosen]
            payment = op.params['payment_method']
            if payment == 'original':
                payment = current.orders[goal.order_id].payment_method_id
            method = current.payment_methods.get(payment)
            answer = 'change to ' + ', '.join(names) + '; payment: ' + (method.kind if method else 'original')
        rows.append({'item': item, 'action': goal.action.replace('_', ' '), 'quantity': 1, 'answer': answer})
        current = next_world
    return rows


def agent_steps(episode):
    return [f"{row.operation.get('name')}:{'ok' if row.result.get('success') else row.result.get('error_code')}"
            for row in episode.tool_executions]


def failure_cases(rows, task_registry, capture_store, *, max_cases=8):
    """Turn sampled ``{task_ref, outcome, episode_id}`` rows into Designer failure cases.

    Tasks are ranked by success rate (lowest first); each case shows one failed attempt.
    """
    from rift.schemas import Split
    by_task = {}
    for row in rows:
        by_task.setdefault(row['task_ref'], []).append(row)
    ranked = []
    for task_ref, samples in by_task.items():
        scored = [row for row in samples if row['outcome'] in ('success', 'failure')]
        failed = [row for row in scored if row['outcome'] == 'failure']
        if not failed:
            continue
        ranked.append((sum(row['outcome'] == 'success' for row in scored) / len(scored), task_ref, failed[0]))
    cases = []
    for rate, task_ref, failed in sorted(ranked)[:max_cases]:
        record = task_registry.resolve_record(task_ref)
        if record.identity.split is not Split.TRAIN:
            raise ValueError('only training-split failures may be shown to the Designer')
        episode = capture_store.resolve_episode(failed['episode_id'])
        messages = [row.text for row in episode.service_messages]
        cases.append({'task_ref': task_ref, 'request': record.task.customer_request,
                      'goals': goal_rows(record.world, record.task), 'success_rate': rate,
                      'agent_steps': agent_steps(episode), 'last_message': messages[-1] if messages else None})
    return cases
