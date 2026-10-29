"""Independent transition audit; never calls actual mutations or Engine."""
from .models import Request


def evaluate(adapter, trace, trusted_requests=None):
    state = adapter.World.model_validate(trace.initial_state)
    errors, rows, contexts, used = [], [], {}, set()
    previous, closed, operations = '', False, 0
    def error(code, event=None):
        errors.append({'code': code, 'event_id': event.id if event else None})
    if trace.domain != adapter.name or trace.version != state.version:
        error('DOMAIN_PROVENANCE_MISMATCH')
    if trace.trace_hash != trace.computed_hash():
        error('TRACE_INTEGRITY')
    if adapter.invariants(state):
        error('INITIAL_WORLD_INVALID')
    for index, event in enumerate(trace.events):
        if event.id != f'event_{index:06d}' or event.previous_hash != previous or event.event_hash != event.computed_hash():
            error('TRACE_INTEGRITY', event)
        previous = event.event_hash
        if closed:
            error('EVENT_AFTER_FINAL', event)
        if event.kind == 'generation':
            expected = {'domain': state.domain, 'version': state.version, 'seed': state.seed, 'initial_hash': state.state_hash}
            if index != 0 or event.payload != expected or event.before is not None or event.after is not None:
                error('PROVENANCE_MISMATCH', event)
        elif event.kind == 'request':
            request = Request.model_validate(event.payload['request'])
            if request.id in contexts or (trusted_requests is not None and trusted_requests.get(request.id) != request):
                error('AUTHORITY_MISMATCH', event)
            contexts[request.id] = request
            if event.before is not None or event.after is not None:
                error('UNEXPECTED_SNAPSHOT', event)
        elif event.kind == 'operation':
            operations += 1
            p, codes = event.payload, []
            operation = adapter.Operation.model_validate(p['operation'])
            request = contexts.get(p['request_id'])
            if event.before != state.model_dump(mode='json'):
                error('STATE_CHAIN_MISMATCH', event)
            if event.after is None:
                error('MISSING_SNAPSHOT', event)
                continue
            after = adapter.World.model_validate(event.after)
            decision = adapter.decide(state, p['actor'], operation)
            reused = bool(request and request.id in used)
            authorized = bool(request and request.authorizes(p['actor'], operation) and not reused)
            if request:
                used.add(request.id)
            if not authorized:
                codes.append('UNAUTHORIZED_OPERATION')
            allowed = None if decision.error_code == 'INVALID_WORLD' else decision.allowed
            success = p['result']['success']
            business = success and allowed is not None and (not decision.allowed or after != decision.expected_state or p['result']['data'] != decision.data)
            if business:
                codes.append(decision.error_code or 'EXPECTED_TRANSITION_MISMATCH')
            if not success and after != state:
                codes.append('REJECTED_STATE_CHANGED')
            expected_code = ('UNAUTHORIZED_OPERATION' if not request or not request.authorizes(p['actor'], operation)
                             else 'REQUEST_ALREADY_USED' if reused else decision.error_code)
            if not success and p['mode'] == 'reference' and p['result']['error_code'] != expected_code:
                codes.append('REJECTION_CODE_MISMATCH')
            if not success and decision.allowed and authorized:
                codes.append('FALSE_REJECTION')
            post = adapter.invariants(after)
            codes.extend(e['code'] for e in post)
            classification = ('unscored' if allowed is None else 'successful_business_violation' if business
                              else 'authorization_only' if not authorized else 'rejected_invalid_attempt' if not success and not decision.allowed
                              else 'false_rejection' if not success else 'legal_execution')
            correct = not codes if success else not post and not any(c in {'FALSE_REJECTION', 'REJECTED_STATE_CHANGED', 'REJECTION_CODE_MISMATCH'} for c in codes)
            if allowed is None and not post:
                correct = None
            rows.append({'event_id': event.id, 'operation': operation.name, 'authorized': authorized,
                         'reference_allowed': allowed, 'reference_error_code': decision.error_code,
                         'success': success, 'classification': classification, 'operation_correct': correct,
                         'violations': sorted(set(codes)), 'before_hash': state.state_hash, 'after_hash': after.state_hash})
            state = after
        elif event.kind == 'final':
            closed = True
            if event.payload != {'state_hash': state.state_hash} or event.before is not None or event.after is not None:
                error('FINAL_HASH_MISMATCH', event)
    if not trace.events or trace.events[0].kind != 'generation' or not closed:
        error('TRACE_NOT_FINALIZED')
    if trace.final_state != state.model_dump(mode='json'):
        error('FINAL_STATE_MISMATCH')
    if operations > trace.operation_budget:
        error('OPERATION_BUDGET_EXCEEDED')
    if trusted_requests is not None and contexts != trusted_requests:
        error('AUTHORITY_SET_MISMATCH')
    classes = ('legal_execution', 'rejected_invalid_attempt', 'successful_business_violation', 'authorization_only', 'false_rejection', 'unscored')
    return {'domain': adapter.name, 'version': trace.version, 'trace_valid': not errors,
            'status': 'fail' if errors or any(r['operation_correct'] is False for r in rows) else 'unknown' if any(r['operation_correct'] is None for r in rows) else 'pass',
            'operations': rows, 'violations': errors, 'metrics': {c: sum(r['classification'] == c for r in rows) for c in classes},
            'final_state_hash': state.state_hash}
