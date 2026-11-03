"""Construct and replay registered transaction history through native tools."""
from usami_harness.models import digest
from usami_v2.commerce.models import ExternalEvent
from .outcomes import _temporary_task
from .task_certification import CustomerScenario, _apply_event, _customer_delegation, _execute
from .tasks import OrderCancellationGoal, RemedyGoal, delegated_line_ids


class HistoryStepFailed(ValueError):
    pass


def apply_history(world, aliases, steps):
    state = world.clone()
    for index, step in enumerate(steps):
        try:
            if step.operation == 'return_received':
                line_id = aliases[step.line_alias]
                matches = [row for row in state.returns.values() if row.status == 'scheduled'
                           and state.cases[row.case_id].line_id == line_id
                           and len(state.cases[row.case_id].unit_indices) == step.quantity]
                if len(matches) != 1:
                    raise ValueError('RETURN_NOT_UNIQUE')
                event = ExternalEvent(id='history-event-' + digest([world.state_hash, index])[:20],
                    kind='return_received', target_id=matches[0].case_id, at=state.clock)
                state, _ = _apply_event(state, event)
                continue
            if step.operation == 'cancel_order':
                order = state.orders[aliases[step.order_alias]]
                line_ids = tuple(sorted(line.id for line in state.lines.values() if line.order_id == order.id))
                goal = OrderCancellationGoal(goal_id='history', order_id=order.id, line_ids=line_ids)
                resource, params = order.id, {}
                customer_id = order.customer_id
            else:
                line = state.lines[aliases[step.line_alias]]
                customer_id = state.orders[line.order_id].customer_id
                action = {'return_delivered_order_items': 'refund', 'request_item_return': 'refund',
                          'reship_order_items': 'reship', 'exchange_delivered_order_items': 'exchange'}[step.operation]
                goal = RemedyGoal(goal_id='history', line_id=line.id, quantity=step.quantity,
                                  allowed_remedies=(action,))
                resource, params = line.id, {'quantity': step.quantity}
                if step.operation == 'exchange_delivered_order_items':
                    params['target_sku_id'] = aliases[step.target_sku_alias]
            task = _temporary_task(state, customer_id, goal)
            scenario = CustomerScenario(customer_id=customer_id, delegated_line_ids=delegated_line_ids(task))
            state, _ = _customer_delegation(state, task, scenario)
            state, _, error = _execute(state, customer_id, step.operation, resource, params,
                                       digest([world.state_hash, index]), 1)
            if error:
                raise ValueError(error)
            # Delegation is a transient historical authorization, never part of
            # the next Customer's starting authority.
            state.delegations.clear()
            state.consents.clear()
        except (KeyError, ValueError) as exc:
            raise HistoryStepFailed(f'HISTORY_STEP_FAILED:{index}:{exc}') from exc
    if steps:
        state.observation_protocol = 'commerce-observation-v2'
    return state
