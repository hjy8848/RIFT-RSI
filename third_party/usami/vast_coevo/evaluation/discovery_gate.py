"""Paired, preregistered Quality-Gated Discovery. No business judging here."""
from collections import Counter
import random
from .customer import CustomerGateManifest, CustomerQualityResult, ProbeEpisode


def _detected(row):
    return any(finding.get('root_identity_status', 'rule_level') == 'rule_level'
               for finding in row.direct_discoveries)


def summarize(rows):
    probes = [probe for row in rows for probe in row.probes]
    valid = [probe for probe in probes if probe.probe_validity == 'valid']
    assessable = sum(probe.probe_validity in {'valid', 'invalid'} for probe in probes)
    roots, coverage = {}, {}
    for row in rows:
        world = row.config.initial_world_hash
        for finding in row.direct_discoveries:
            if finding.get('root_identity_status', 'rule_level') == 'rule_level':
                roots.setdefault(finding['discovery_fingerprint'], set()).add(world)
        for probe in row.probes:
            if probe.probe_validity == 'valid' and not probe.duplicate:
                for key in probe.coverage: coverage.setdefault(key, set()).add(world)
    requests = len(probes)
    result = dict(episodes=len(rows), requests=requests, valid_probes=len(valid), assessable=assessable,
        invalid_probes=sum(probe.probe_validity == 'invalid' for probe in probes),
        unresolved_probes=sum(probe.probe_validity == 'unresolved' for probe in probes),
        censored_probes=sum(probe.probe_validity == 'censored' for probe in probes),
        valid_probe_rate=len(valid) / requests if requests else 0,
        assessable_rate=assessable / requests if requests else 0,
        duplicate_rate=sum(probe.duplicate for probe in probes) / requests if requests else 0,
        protocol_error_rate=sum(row.protocol_errors for row in rows) / max(1, sum(row.customer_calls for row in rows)),
        direct_business_violations=sum(len(row.direct_discoveries) for row in rows),
        incidental_business_violations=sum(len(row.incidental_discoveries) for row in rows),
        authorization_anomalies=sum(len(row.authorization_anomalies) for row in rows),
        independent_rule_failures=len(roots), root_world_support={key: len(value) for key, value in sorted(roots.items())},
        valid_coverage=len(coverage), coverage_world_support={key: len(value) for key, value in sorted(coverage.items())},
        detection_rate=sum(_detected(row) for row in rows) / max(1, len(rows)),
        unresolved_root_discoveries=sum(finding.get('root_identity_status') == 'unresolved'
            for row in rows for finding in row.direct_discoveries),
        customer_calls=sum(row.customer_calls for row in rows), service_calls=sum(row.service_calls for row in rows),
        tool_calls=sum(row.tool_calls for row in rows),
        total_tokens=sum(row.total_tokens for row in rows) if all(row.total_tokens is not None for row in rows) else None,
        query_failures=sum(outcome.get('is_query') is True and outcome.get('success') is False
            for probe in probes for outcome in probe.observed_service_outcomes),
        terminations=dict(Counter(row.termination for row in rows)))
    if any(row.cost_breakdown is not None for row in rows):
        result['role_costs'] = {}
        for role in ('customer', 'service'):
            values = [row.cost_breakdown[role]['total_tokens'] if row.cost_breakdown else None for row in rows]
            result['role_costs'][role + '_tokens'] = sum(values) if all(value is not None for value in values) else None
        result['temporal_costs'] = [dict(episode_id=row.episode_id, **(row.cost_breakdown or {})) for row in rows]
    return result


def _paired_intervals(old, new, policy):
    """Resample whole paired worlds; preserve multiple requests per episode."""
    rng = random.Random(policy.bootstrap_seed)
    v2 = policy.protocol == 'quality-gated-discovery-v2'
    keys = ('valid_probe_rate', 'protocol_error_rate', 'duplicate_rate', 'detection_rate', 'token_ratio', 'tool_ratio', 'service_token_ratio')
    values = {key: [] for key in keys}
    # Cache sufficient per-world counts; bootstrapping does not reparse traces.
    def row_counts(row):
        return (len(row.probes), sum(p.probe_validity == 'valid' for p in row.probes),
                row.protocol_errors, row.customer_calls, sum(p.duplicate for p in row.probes),
                _detected(row), row.cost_breakdown['customer']['total_tokens'] if v2 else row.total_tokens, row.tool_calls,
                row.cost_breakdown['service']['total_tokens'] if v2 else None)
    a, b = list(map(row_counts, old)), list(map(row_counts, new))
    for _ in range(policy.bootstrap_samples):
        indexes = [rng.randrange(len(old)) for _ in old]
        sums = []
        for side in (a, b):
            sums.append([None if any(side[i][col] is None for i in indexes) else sum(side[i][col] for i in indexes)
                         for col in range(9)])
        left, right = sums
        values['valid_probe_rate'].append(right[1] / max(1, right[0]) - left[1] / max(1, left[0]))
        values['protocol_error_rate'].append(right[2] / max(1, right[3]) - left[2] / max(1, left[3]))
        values['duplicate_rate'].append(right[4] / max(1, right[0]) - left[4] / max(1, left[0]))
        values['detection_rate'].append((right[5] - left[5]) / len(old))
        for name, column in (('token_ratio', 6), ('tool_ratio', 7), ('service_token_ratio', 8)):
            if left[column] is not None and right[column] is not None:
                # Zero-baseline/positive-candidate cost is an unbounded increase.
                values[name].append(right[column] / left[column] if left[column] else (1.0 if not right[column] else float('inf')))
    tail = (1 - policy.confidence) / 2
    intervals = {}
    for key, data in values.items():
        if data:
            data.sort()
            # JSON evidence must remain finite. A missing cost interval means
            # unknown/unbounded, which cannot satisfy the cost constraint.
            low, high = data[int(tail * (len(data)-1))], data[int((1-tail) * (len(data)-1))]
            if high != float('inf'): intervals[key] = (low, high)
    if v2:
        for old_key, new_key in (('token_ratio', 'customer_token_ratio'), ('tool_ratio', 'service_tool_ratio')):
            if old_key in intervals: intervals[new_key] = intervals.pop(old_key)
    return intervals


def compare_customer_probes(manifest, baseline, candidate):
    manifest = CustomerGateManifest.model_validate(manifest.model_dump(mode='json'))
    baseline = tuple(ProbeEpisode.model_validate(row.model_dump(mode='json')) for row in baseline)
    candidate = tuple(ProbeEpisode.model_validate(row.model_dump(mode='json')) for row in candidate)
    old, new = summarize(baseline), summarize(candidate)
    def result(decision, reasons, intervals=None, improvements=()):
        return CustomerQualityResult(protocol=manifest.policy.protocol, decision=decision, reasons=tuple(reasons), manifest_hash=manifest.manifest_hash,
            baseline=old, candidate=new, intervals=intervals or {}, improvements=tuple(improvements),
            baseline_reports=tuple(row.report_hash for row in baseline), candidate_reports=tuple(row.report_hash for row in candidate))
    if len(baseline) != len(candidate) or len(baseline) != len(manifest.configs):
        return result('INCONCLUSIVE', ['missing_paired_evidence'])
    for expected, left, right in zip(manifest.configs, baseline, candidate):
        if (left.config != expected or right.config != expected or
                left.customer_snapshot_hash != manifest.baseline_customer_hash or
                right.customer_snapshot_hash != manifest.candidate_customer_hash or
                left.service_snapshot_hash != manifest.service_hash or right.service_snapshot_hash != manifest.service_hash):
            return result('REJECT', ['pinned_pair_or_partition_mismatch'])
    if any(not row.replay_verified or not row.injection_verified for row in baseline + candidate):
        return result('REJECT', ['replay_or_skill_injection_not_verified'])
    if manifest.policy.protocol == 'quality-gated-discovery-v2' and any(
            row.evaluator != 'cloudops-probes-v2' or row.cost_breakdown is None for row in baseline + candidate):
        return result('INCONCLUSIVE', ['v2_read_and_role_cost_evidence_required'])
    injected = {key for row in candidate for key in row.injected_skill_hashes}
    if set(manifest.required_candidate_skill_hashes) - injected:
        return result('INCONCLUSIVE', ['candidate_skill_not_exercised_in_actual_requests'])
    policy = manifest.policy
    if len(baseline) < policy.min_pairs or min(old['requests'], new['requests']) < policy.min_requests:
        return result('INCONCLUSIVE', ['insufficient_preregistered_samples'])
    if min(old['assessable_rate'], new['assessable_rate']) < policy.min_assessable_rate:
        return result('INCONCLUSIVE', ['insufficient_independently_assessable_probes'])
    if any(any(word in row.termination.lower() for word in ('provider', 'interrupted')) for row in baseline + candidate):
        return result('INCONCLUSIVE', ['external_failure_censors_comparison'])
    intervals = _paired_intervals(baseline, candidate, policy)
    reject, uncertain = [], []
    if new['valid_probe_rate'] < policy.min_valid_probe_rate:
        # Censored and unresolved requests are never blamed on the Customer.
        (uncertain if new['unresolved_probes'] or new['censored_probes'] else reject).append('valid_probe_quality_floor_not_met')
    for key, margin, direction in (('valid_probe_rate', policy.quality_margin, -1),
                                    ('protocol_error_rate', policy.reliability_margin, 1),
                                    ('duplicate_rate', policy.duplicate_margin, 1)):
        low, high = intervals[key]
        if (high < -margin if direction == -1 else low > margin): reject.append(key + '_material_regression')
        elif (low < -margin if direction == -1 else high > margin): uncertain.append(key + '_noninferiority_unproven')
    v2 = policy.protocol == 'quality-gated-discovery-v2'
    token_key, tool_key = ('customer_token_ratio', 'service_tool_ratio') if v2 else ('token_ratio', 'tool_ratio')
    for key in ((token_key, tool_key, 'service_token_ratio') if v2 else (token_key, tool_key)):
        if key not in intervals: uncertain.append(key + '_unknown_or_unbounded')
        elif intervals[key][0] > policy.max_cost_ratio: reject.append(key + '_material_regression')
        elif intervals[key][1] > policy.max_cost_ratio: uncertain.append(key + '_cost_bound_unproven')
    if reject: return result('REJECT', reject, intervals)
    if uncertain: return result('INCONCLUSIVE', uncertain, intervals)
    improvements = []
    old_roots, new_roots = set(old['root_world_support']), set(new['root_world_support'])
    stable_roots = {key for key, count in new['root_world_support'].items() if count >= policy.min_root_worlds}
    no_detection_loss = intervals['detection_rate'][0] >= 0 and old_roots <= new_roots
    if no_detection_loss and stable_roots - old_roots:
        improvements.append('replicated_new_independent_rule_failure')
    if old_roots and old_roots <= new_roots and intervals['detection_rate'][0] > 0:
        improvements.append('paired_detection_rate_improved')
    old_cover, new_cover = set(old['coverage_world_support']), set(new['coverage_world_support'])
    stable_cover = {key for key, count in new['coverage_world_support'].items() if count >= policy.min_root_worlds}
    if no_detection_loss and old_cover <= new_cover and stable_cover - old_cover:
        improvements.append('replicated_valid_boundary_coverage_expanded')
    comparable_exposure = not v2 or all(
        len(right.probes) >= len(left.probes) and all(not row.protocol_errors and
            not row.direct_discoveries and not row.incidental_discoveries and
            not any(p.probe_validity == 'censored' for p in row.probes) and
            not any(word in row.termination.lower() for word in ('error', 'invalid', 'interrupted', 'llm_budget', 'tool_budget', 'service_llm', 'service_tool'))
            for row in (left, right)) for left, right in zip(baseline, candidate))
    if (comparable_exposure and no_detection_loss and old_cover <= new_cover and old_cover and
            intervals[token_key][1] <= 1 - policy.min_cost_reduction and intervals[tool_key][1] <= 1):
        improvements.append('paired_cost_reduced_with_preserved_discovery_and_coverage')
    if not improvements:
        return result('INCONCLUSIVE', ['no_robust_preregistered_improvement'], intervals)
    return result('ACCEPT', ['quality_constraints_and_discovery_gain_supported'], intervals, improvements)
