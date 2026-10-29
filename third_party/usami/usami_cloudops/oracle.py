"""Independent audit of actual transitions, not Engine success labels."""
from .models import Operation, Request, digest
from .rules import decide, invariants


def evaluate(trace, trusted_requests=None):
    violations, rows = [], []
    state = trace.initial_state.clone(); contexts, used = {}, set(); previous = ''; closed = False; operations = 0
    def error(code, event=None): violations.append({'code': code, 'event_id': event.id if event else None})
    if trace.trace_hash != trace.computed_hash(): error('TRACE_INTEGRITY')
    if invariants(state): error('INITIAL_WORLD_INVALID')
    for index, event in enumerate(trace.events):
        if event.id != f'event_{index:06d}' or event.previous_hash != previous or event.event_hash != event.computed_hash(): error('TRACE_INTEGRITY', event)
        previous = event.event_hash
        if closed: error('EVENT_AFTER_FINAL', event)
        if event.kind == 'generation':
            expected = {'domain': 'cloudops', 'version': state.version, 'seed': state.seed, 'initial_hash': state.state_hash}
            if index != 0 or event.payload != expected or event.before or event.after: error('PROVENANCE_MISMATCH', event)
        elif event.kind == 'request':
            req = Request.model_validate(event.payload['request'])
            if req.id in contexts or (trusted_requests is not None and trusted_requests.get(req.id) != req): error('AUTHORITY_MISMATCH', event)
            contexts[req.id] = req
            if event.before or event.after: error('UNEXPECTED_SNAPSHOT', event)
        elif event.kind == 'operation':
            operations += 1; p = event.payload; op = Operation.model_validate(p['operation']); req = contexts.get(p['request_id'])
            codes = []
            if event.before != state: error('STATE_CHAIN_MISMATCH', event)
            if event.after is None: error('MISSING_SNAPSHOT', event); continue
            decision = decide(state, p['actor'], op)
            already_used = bool(req and req.id in used)
            authorized = bool(req and req.authorizes(p['actor'], op) and not already_used)
            if req: used.add(req.id)
            if not authorized: codes.append('UNAUTHORIZED_OPERATION')
            reference_allowed = None if decision.error_code == 'INVALID_WORLD' else decision.allowed
            success = p['result']['success']
            business = success and reference_allowed is not None and (not decision.allowed or event.after != decision.expected_state or p['result']['data'] != decision.data)
            if business: codes.append(decision.error_code or 'EXPECTED_TRANSITION_MISMATCH')
            if not success and event.after != state: codes.append('REJECTED_STATE_CHANGED')
            expected_error = 'UNAUTHORIZED_OPERATION' if not req or not req.authorizes(p['actor'], op) else 'REQUEST_ALREADY_USED' if already_used else decision.error_code
            if not success and p['mode'] != 'refuse' and p['mode'] == 'reference' and p['result']['error_code'] != expected_error: codes.append('REJECTION_CODE_MISMATCH')
            if not success and decision.allowed and authorized: codes.append('FALSE_REJECTION')
            post_errors = invariants(event.after)
            codes.extend(e['code'] for e in post_errors)
            if reference_allowed is None: classification = 'unscored'
            elif business: classification = 'successful_business_violation'
            elif success and not authorized: classification = 'authorization_only'
            elif not success and not decision.allowed: classification = 'rejected_invalid_attempt'
            elif not success and not authorized: classification = 'authorization_only'
            elif not success: classification = 'false_rejection'
            else: classification = 'legal_execution'
            correct = (not codes) if success else not any(c in {'FALSE_REJECTION','REJECTED_STATE_CHANGED','REJECTION_CODE_MISMATCH'} for c in codes) and not post_errors
            if reference_allowed is None and not post_errors: correct = None
            row = {'event_id': event.id, 'operation': op.name, 'reference_allowed': reference_allowed,
                'reference_error_code': decision.error_code, 'authorized': authorized, 'success': success,
                'classification': classification, 'operation_correct': correct, 'violations': sorted(set(codes)),
                'before_hash': state.state_hash, 'after_hash': event.after.state_hash}
            rows.append(row); state = event.after.clone()
        elif event.kind == 'final':
            closed = True
            if event.payload != {'state_hash': state.state_hash} or event.before or event.after: error('FINAL_HASH_MISMATCH', event)
    if not trace.events or trace.events[0].kind != 'generation' or not closed: error('TRACE_NOT_FINALIZED')
    if state != trace.final_state: error('FINAL_STATE_MISMATCH')
    if operations > trace.operation_budget: error('OPERATION_BUDGET_EXCEEDED')
    if trusted_requests is not None and set(contexts) != set(trusted_requests): error('AUTHORITY_SET_MISMATCH')
    counters = {key: sum(r['classification'] == key for r in rows) for key in ('legal_execution','rejected_invalid_attempt','successful_business_violation','authorization_only','false_rejection','unscored')}
    return {'domain': 'cloudops', 'version': trace.version, 'trace_valid': not violations,
        'status': 'fail' if violations or any(r['operation_correct'] is False for r in rows) else 'unknown' if any(r['operation_correct'] is None for r in rows) else 'pass',
        'operations': rows, 'violations': violations, 'metrics': counters, 'final_state_hash': state.state_hash}
