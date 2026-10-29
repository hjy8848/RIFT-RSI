"""Role and temporal cost accounting; time order is not a causal blame claim."""


def _tools(call):
    try: return call.response['choices'][0]['message'].get('tool_calls') or []
    except (KeyError, TypeError, IndexError): return []


def _sum(calls, tools=0):
    result = dict(llm_calls=len(calls), prompt_tokens=0, completion_tokens=0, total_tokens=0,
                  missing_usage=0, tool_calls=tools)
    for call in calls:
        usage = call.response.get('usage') if isinstance(call.response, dict) else None
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            value = usage.get(key) if isinstance(usage, dict) else None
            if type(value) is not int or value < 0:
                result[key] = None
            elif result[key] is not None: result[key] += value
        if not isinstance(usage, dict) or any(type(usage.get(key)) is not int or usage[key] < 0
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')):
            result['missing_usage'] += 1
    return result


def breakdown(evidence, execution, direct_event_ids=()):
    customer = [call for call in evidence.model_calls if call.role == 'customer']
    service = [call for call in evidence.model_calls if call.role == 'service']
    ordering = {}
    for index, call in enumerate(customer): ordering[call.id] = (index, 0, 0)
    tool_to_call = {}
    for index, call in enumerate(service):
        turn = sum(msg.get('role') == 'user' for msg in call.base_request.get('messages', [])) - 1
        ordering[call.id] = (turn, 1, index)
        for tool in _tools(call): tool_to_call[tool['id']] = call
    rows = {row['event_id']: row for row in execution.oracle['operations']}
    first_business, first_direct = None, None
    business_index, direct_index = None, None
    for index, record in enumerate(execution.tool_records):
        row = rows.get(record.event_id)
        if not row or row['classification'] != 'successful_business_violation': continue
        if first_business is None: first_business, business_index = tool_to_call.get(record.call_id), index
        if record.event_id in direct_event_ids and first_direct is None:
            first_direct, direct_index = tool_to_call.get(record.call_id), index
    calls = customer + service
    def phase(boundary, before, tool_boundary):
        if boundary is None: return None
        chosen = [call for call in calls if (ordering[call.id] <= ordering[boundary.id] if before
                                             else ordering[call.id] > ordering[boundary.id])]
        tools = tool_boundary + 1 if before else len(execution.tool_records) - tool_boundary - 1
        return {'customer': _sum([call for call in chosen if call.role == 'customer']),
                'service': _sum([call for call in chosen if call.role == 'service'], tools)}
    return dict(customer=_sum(customer), service=_sum(service, len(execution.tool_records)),
        through_first_direct_discovery=phase(first_direct, True, direct_index),
        after_first_business_violation=phase(first_business, False, business_index),
        first_direct_call_id=first_direct.id if first_direct else None,
        first_business_violation_call_id=first_business.id if first_business else None,
        attribution='temporal_only_not_causal; full completion usage belongs to the call containing discovery',
        termination=evidence.termination)
