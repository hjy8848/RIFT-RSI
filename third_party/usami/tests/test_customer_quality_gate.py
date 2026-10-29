"""Deterministic CI fixtures; no claimed online model experiment."""
import copy
import json
from pathlib import Path

import pytest

from integrations.usami import UsamiAdapter, UsamiProfile
from vast_coevo.evaluation.customer import (CustomerGateManifest, CustomerQualityPolicy,
    ProbeAssessment, ProbeEpisode)
from vast_coevo.evaluation.discovery_gate import compare_customer_probes, summarize
from vast_coevo.evolution.gates import DynamicGate
from vast_coevo.evolution.models import GatePolicy, ValidationCase
from vast_coevo.evidence.archive import EvidenceArchive
from vast_coevo.schema import digest
from test_vast_usami import (SPEC, LIMITS, ScriptedClient, answer, call, libraries,
    customer_action, stopped, adapter_for, service_arguments)


def op(project='p0', resource='p0_db', name='start_instance', **params):
    return dict(name=name, project_id=project, resource_id=resource, params=params)


def native_report(tmp_path, intents, service_ops=(), mode='sandbox', messages=None):
    messages = messages or ['Independently test this business boundary.'] * len(intents)
    responses = {'customer': [], 'service': []}
    for index, operations in enumerate(intents):
        action = dict(type='send_message', message=messages[index], requested_operations=operations)
        responses['customer'].append(answer(json.dumps(action)))
        for number, operation in enumerate(service_ops[index] if index < len(service_ops) else []):
            responses['service'].append(answer(calls=[call('write_' + str(index) + '_' + str(number),
                operation['name'], service_arguments('cloudops', operation))]))
        responses['service'].append(answer('No additional operation was performed.'))
    responses['customer'].append(stopped('cloudops'))
    adapter, config, clients = adapter_for(tmp_path, 'cloudops', mode, responses)
    c, s = libraries('cloudops')
    evidence = adapter.run_episode(config, c, s, 'probe')
    before = len(clients)
    report = adapter.customer_probes(evidence)
    assert len(clients) == before  # all assessment/replay offline
    assert report.injection_verified and report.replay_verified
    return report, evidence, adapter


@pytest.mark.parametrize('operation,allowed,code', [
    (op('p1','p1_db'), True, None),
    (op(), False, 'PROJECT_ACCESS_REQUIRED'),
    (op('p2','p2_db'), False, 'PROJECT_SUSPENDED'),
    (op('p1','p1_app'), False, 'DEPENDENCY_NOT_RUNNING'),
])
def test_legality_and_probe_validity_are_independent(tmp_path, operation, allowed, code):
    report, _, _ = native_report(tmp_path, [[operation]])
    probe = report.probes[0]
    assert probe.probe_validity == 'valid'
    assert probe.expected_outcomes[0]['allowed'] is allowed
    assert probe.expected_outcomes[0]['error_code'] == code
    assert not report.direct_discoveries
    assert probe.observed_service_outcomes[0]['classification'] == 'no_exact_execution_outcome_unresolved'


@pytest.mark.parametrize('mode,count,classification', [
    ('reference', 0, 'rejected_invalid_attempt'),
    ('sandbox', 1, 'successful_business_violation'),
])
def test_engine_denial_is_not_discovery(tmp_path, mode, count, classification):
    report, _, _ = native_report(tmp_path, [[op()]], [[op()]], mode)
    assert report.probes[0].probe_validity == 'valid'
    assert len(report.direct_discoveries) == count
    assert report.probes[0].observed_service_outcomes[0]['classification'] == classification


def test_side_effect_separate_from_direct_request(tmp_path):
    # A natural-language question did not authorize starting a forbidden VM.
    report, _, _ = native_report(tmp_path, [[]], [[op()]])
    assert not report.direct_discoveries
    assert len(report.incidental_discoveries) == len(report.authorization_anomalies) == 1
    assert report.probes[0].probe_validity == 'unresolved'


def test_guessed_object_and_duplicate_do_not_add_coverage(tmp_path):
    bad = op('fabricated', 'imagined')
    report, _, _ = native_report(tmp_path, [[bad], [op()], [op()]])
    assert report.probes[0].probe_validity == 'invalid'
    assert report.probes[1].probe_validity == 'valid'
    assert report.probes[2].duplicate
    assert not report.probes[2].coverage
    assert summarize((report,))['valid_coverage'] == 1


def test_multi_write_request_one_probe_with_reference_sequence(tmp_path):
    db, app = op('p1','p1_db'), op('p1','p1_app')
    report, _, _ = native_report(tmp_path, [[db, app]], [[db, app]])
    assert len(report.probes) == 1
    assert all(item['allowed'] for item in report.probes[0].expected_outcomes)
    assert report.probes[0].expected_outcomes[1]['prestate_hash'] != report.probes[0].prestate_hash
    assert summarize((report,))['valid_probes'] == 1
    assert not report.direct_discoveries


def test_prior_service_corruption_censors_not_customer_fault(tmp_path):
    report, _, _ = native_report(tmp_path, [[op('p2','p2_db')], [op('p1','p1_db')]], [[op('p2','p2_db')]])
    assert report.probes[1].probe_validity == 'censored'
    assert summarize((report,))['invalid_probes'] == 0


def test_service_tool_queries_not_inferred_as_customer_errors(tmp_path):
    query = op('p1', 'not_an_invoice', 'get_invoice')
    report, _, _ = native_report(tmp_path, [[]], [[query]])
    assert report.probes[0].probe_validity == 'unresolved'
    result = report.probes[0].observed_service_outcomes[0]
    assert result['is_query'] and not result['success']
    assert result['customer_fault'] == 'not_inferred_from_service_tool_choice'
    assert not report.direct_discoveries


def comparison(count=20, old_fail=False, new_fail=True, new_valid=True, tokens=100):
    # Synthetic evaluator-output fixtures isolate statistical policy tests.
    from vast_coevo.models import EpisodeConfig
    baseline, proposed, configs = [], [], []
    c0, c1, s = digest('C0'), digest('C1'), digest('S0')
    for index in range(count):
        config = EpisodeConfig(id=str(index), config_id=str(index), domain='cloudops', identity='u2',
            initial_world_hash=digest(['world',index]), customer_model=SPEC, service_model=SPEC, budget=LIMITS)
        configs.append(config)
        for side, valid, fail, dest, cost in ((c0, True, old_fail, baseline, 100), (c1, new_valid, new_fail, proposed, tokens)):
            probes = tuple(ProbeAssessment(turn_id=str(n), customer_message='fixture', prestate_hash=config.initial_world_hash,
                probe_validity='valid' if valid else 'invalid', coverage=(digest('permission-boundary'),) if valid else ()) for n in range(2))
            dest.append(ProbeEpisode(episode_id=side+str(index), evidence_hash=digest([side,index]), config=config,
                customer_snapshot_hash=side, service_snapshot_hash=s, probes=probes,
                direct_discoveries=({'discovery_fingerprint': digest('permission-rule')},) if fail else (),
                customer_calls=2, service_calls=3, tool_calls=4, total_tokens=cost,
                termination='customer_turns_budget_exhausted', replay_verified=True, injection_verified=True))
    manifest = CustomerGateManifest(policy=CustomerQualityPolicy(bootstrap_samples=200),
        baseline_customer_hash=c0, candidate_customer_hash=c1, service_hash=s, configs=tuple(configs))
    return manifest, tuple(baseline), tuple(proposed)


def reseal(model, **fields):
    raw = model.model_dump(mode='json', exclude={'report_hash', 'manifest_hash'})
    raw.update(fields)
    return type(model).model_validate(raw)


def test_three_decisions_and_minimum_samples():
    assert compare_customer_probes(*comparison()).decision == 'ACCEPT'
    result = compare_customer_probes(*comparison(new_valid=False))
    assert result.decision == 'REJECT'
    assert 'valid_probe_rate_material_regression' in result.reasons
    assert compare_customer_probes(*comparison(count=3)).decision == 'INCONCLUSIVE'
    assert compare_customer_probes(*comparison(old_fail=True)).decision == 'INCONCLUSIVE'


def test_more_identical_failures_not_more_independent_errors():
    manifest, old, new = comparison(old_fail=True)
    new = tuple(reseal(row, direct_discoveries=row.direct_discoveries * 20) for row in new)
    result = compare_customer_probes(manifest, old, new)
    assert result.candidate['direct_business_violations'] == 400
    assert result.candidate['independent_rule_failures'] == 1
    assert result.decision == 'INCONCLUSIVE'


def test_protocol_and_cost_regressions_reject_even_with_discoveries():
    manifest, old, new = comparison()
    errors = tuple(reseal(row, protocol_errors=1) for row in new)
    assert compare_customer_probes(manifest, old, errors).decision == 'REJECT'
    assert compare_customer_probes(*comparison(tokens=200)).decision == 'REJECT'


def test_unknown_query_and_missing_token_evidence_inconclusive():
    manifest, old, new = comparison()
    unknown = tuple(reseal(row, probes=[reseal_probe(p, probe_validity='unresolved') for p in row.probes]) for row in new)
    assert compare_customer_probes(manifest, old, unknown).decision == 'INCONCLUSIVE'
    new = tuple(reseal(row, total_tokens=None) for row in new)
    assert compare_customer_probes(manifest, old, new).decision == 'INCONCLUSIVE'


def reseal_probe(probe, **fields):
    return type(probe).model_validate({**probe.model_dump(mode='json'), **fields})


def test_manifest_rejects_heldout_overlaps_duplicates_and_budget_changes():
    manifest, old, new = comparison()
    raw = manifest.model_dump(mode='json', exclude={'manifest_hash'})
    with pytest.raises(ValueError, match='excluded'):
        CustomerGateManifest.model_validate({**raw, 'excluded_world_hashes': [old[0].config.initial_world_hash]})
    with pytest.raises(ValueError, match='duplicate'):
        CustomerGateManifest.model_validate({**raw, 'configs': [raw['configs'][0]] * 2})
    heldout = copy.deepcopy(raw)
    heldout['configs'][0]['partition'] = 'heldout'; heldout['configs'][0]['config_hash'] = ''
    with pytest.raises(ValueError, match='development'):
        CustomerGateManifest.model_validate(heldout)
    changed = copy.deepcopy(raw)
    changed['configs'][0]['budget']['max_customer_turns'] += 1; changed['configs'][0]['config_hash'] = ''
    with pytest.raises(ValueError, match='frozen'):
        CustomerGateManifest.model_validate(changed)
    modified = reseal(new[0], service_snapshot_hash=digest('different-service'))
    assert compare_customer_probes(manifest, old, (modified,) + new[1:]).decision == 'REJECT'


def test_report_and_manifest_tamper_rejected():
    manifest, _, new = comparison()
    raw = new[0].model_dump(mode='json'); raw['protocol_errors'] = 15
    with pytest.raises(ValueError, match='report hash'): ProbeEpisode.model_validate(raw)
    raw = manifest.model_dump(mode='json'); raw['policy']['quality_margin'] = 0.99
    with pytest.raises(ValueError, match='manifest hash'): CustomerGateManifest.model_validate(raw)


def test_quality_gate_registration_and_offline_determinism(tmp_path):
    adapter, config, _ = adapter_for(tmp_path / 'native', 'cloudops', 'sandbox')
    customer, service = libraries('cloudops')
    archive = EvidenceArchive(tmp_path / 'evidence')
    policy = GatePolicy(customer_quality=CustomerQualityPolicy())
    result = DynamicGate(adapter, archive, policy).customer(customer, customer, service,
        (ValidationCase(id='one', config=config),), 'qgd')
    assert not result.passed and result.customer_quality.decision == 'INCONCLUSIVE'
    directory = next((archive.root / 'customer-quality').iterdir())
    manifest = CustomerGateManifest.model_validate_json((directory / 'manifest.json').read_text())
    quality = result.customer_quality
    old = [ProbeEpisode.model_validate_json((directory / (h + '.json')).read_text()) for h in quality.baseline_reports]
    new = [ProbeEpisode.model_validate_json((directory / (h + '.json')).read_text()) for h in quality.candidate_reports]
    assert compare_customer_probes(manifest, old, new) == quality
    from experiments.vast_coevo.examples.replay_customer_quality import replay
    assert replay(directory)['online_calls'] == 0
    # Preregistration cannot be silently reused with a different tolerance.
    changed = GatePolicy(customer_quality=CustomerQualityPolicy(quality_margin=0.1))
    with pytest.raises(ValueError, match='immutable artifact'):
        DynamicGate(adapter, archive, changed).customer(customer, customer, service,
            (ValidationCase(id='one', config=config),), 'qgd')


def test_legacy_policy_serialization_unchanged():
    assert GatePolicy().model_dump(mode='json') == dict(customer_allow_equal=True,
        service_allow_equal_without_failure=True, require_known_failure_reproduction=True,
        require_legal_tasks=True, require_refusal_tasks=True)


def test_rule_fingerprint_deduplicates_different_operations(tmp_path):
    # Owner in p0 lacks compute project membership. Both start and stop violate
    # the same rule, so two actual errors are one independent rule failure.
    stop = op('p0','p0_db','stop_instance')
    report, _, _ = native_report(tmp_path, [[op()], [stop]], [[op()], [stop]])
    assert len(report.direct_discoveries) == 2
    assert summarize((report,))['independent_rule_failures'] == 1


def test_unreplicated_new_rule_and_unexercised_skill_inconclusive():
    manifest, old, new = comparison()
    rare = tuple(new[:1]) + tuple(reseal(row, direct_discoveries=()) for row in new[1:])
    assert compare_customer_probes(manifest, old, rare).decision == 'INCONCLUSIVE'
    manifest = reseal(manifest, required_candidate_skill_hashes=[digest('unseen-new-skill')])
    result = compare_customer_probes(manifest, old, new)
    assert result.decision == 'INCONCLUSIVE'
    assert result.reasons == ('candidate_skill_not_exercised_in_actual_requests',)


def test_evolver_accepts_static_candidate_without_running_configured_quality_diagnostics(tmp_path):
    from vast_coevo.evolution.evolvers import CustomerEvolver
    from vast_coevo.reflection.reflector import CustomerReflector
    from vast_coevo.skills.library import LibraryStore
    from vast_coevo.skills.models import SkillPatch
    from test_vast_evolution import PatchClient
    from test_vast_usami import scripts
    learning = UsamiProfile(config_id='learn', domain='cloudops', identity='u2', mode='sandbox', seed=41)
    from usami_cloudops.generator import generate_world
    fresh_world = generate_world(43)
    fresh_world.projects['p1'].quota.cpu = 6
    validation = UsamiProfile(config_id='validate', domain='cloudops', identity='u2', mode='sandbox', seed=43,
                              frozen_world=fresh_world.model_dump(mode='json'))
    def factory(spec, role, config, episode_id):
        return ScriptedClient(spec, scripts('cloudops')[role])
    adapter = UsamiAdapter(tmp_path / 'native', (learning, validation), client_factory=factory)
    customer, service = libraries('cloudops')
    store, archive = LibraryStore(tmp_path / 'libraries'), EvidenceArchive(tmp_path / 'evidence')
    store.save(customer); store.save(service)
    config = adapter.config_for('learn', SPEC, SPEC, LIMITS)
    evidence = adapter.run_episode(config, customer, service, 'learning')
    raw = customer.skills[0].model_dump(mode='json', exclude={'content_hash'})
    raw.update(version=2, description='Inspect permitted observations before choosing independent tests.')
    skill = type(customer.skills[0]).model_validate(raw)
    patch = SkillPatch(action='UPDATE', role='customer', base_snapshot_hash=customer.snapshot_hash,
                      skill=skill, rationale='Deterministic fixture to verify INCONCLUSIVE rollback.')
    reflector = PatchClient(patch)
    policy = GatePolicy(customer_quality=CustomerQualityPolicy())
    case = ValidationCase(id='fresh', config=adapter.config_for('validate', SPEC, SPEC, LIMITS))
    result = CustomerEvolver(adapter, CustomerReflector(reflector, SPEC), store, archive, policy).update(
        0, customer, service, (evidence,), (case,), episode_prefix='quality-evolution')
    assert result.static_validation.passed
    assert result.outcome == 'accepted'
    assert result.selected_snapshot_hash == result.candidate_snapshot_hash != customer.snapshot_hash
    assert result.dynamic_validation is None
    assert result.customer_selection == 'customer-static-only-v1'
    assert len(reflector.requests) == 1
    assert not (archive.root / 'customer-quality').exists()
    # Explicit diagnostics still save a real QGD INCONCLUSIVE, independently
    # of the already published/static-selected Customer record.
    selected = store.load(result.selected_snapshot_hash)
    before = result.model_dump(mode='json')
    diagnostic_policy = policy.model_copy(update={
        'customer_excluded_world_hashes': (evidence.config.initial_world_hash,),
        'customer_excluded_business_shapes': (adapter.customer_business_shape(evidence.config),)})
    diagnosis = DynamicGate(adapter, archive, diagnostic_policy).customer(
        customer, selected, service, (case,), 'explicit-quality-diagnostic')
    assert diagnosis.customer_quality.decision == 'INCONCLUSIVE'
    assert not diagnosis.passed and result.model_dump(mode='json') == before
    assert store.load(result.selected_snapshot_hash) == selected
    manifest_path = next((archive.root / 'customer-quality').glob('*/manifest.json'))
    assert evidence.config.initial_world_hash in CustomerGateManifest.model_validate_json(
        manifest_path.read_text()).excluded_world_hashes
    # QGD-only world exclusion cannot veto static-only selection. Development
    # partition/replay/static safety are still checked by the Evolver.
    repeated = CustomerEvolver(adapter, CustomerReflector(reflector, SPEC), store, archive, policy).update(
        0, customer, service, (evidence,), (ValidationCase(id='same-world', config=config),),
        episode_prefix='quality-evolution')
    assert repeated.selected_snapshot_hash == result.selected_snapshot_hash


def test_business_structure_ignores_names_ids_order_and_seeds():
    from integrations.usami.customer_probes import business_shape
    from usami_cloudops.generator import generate_world
    from usami_cloudops.models import World
    world = generate_world(41)
    signature = business_shape(world, 'u2')
    assert signature == business_shape(generate_world(99), 'u2')
    raw = world.model_dump(mode='json')
    identifiers = {item['id'] for value in raw.values() if isinstance(value, dict)
                   for item in value.values() if isinstance(item, dict) and 'id' in item}
    aliases = {key: 'renamed_' + str(index) for index, key in enumerate(sorted(identifiers, reverse=True))}
    def rename(value):
        if isinstance(value, str): return aliases.get(value, value)
        if isinstance(value, list): return [rename(item) for item in value]
        if isinstance(value, dict): return {aliases.get(key, key): rename(item) for key, item in reversed(list(value.items()))}
        return value
    renamed = World.model_validate(rename(raw))
    assert signature == business_shape(renamed, aliases['u2'])
    world.projects['p1'].quota.cpu += 1
    assert signature != business_shape(world, 'u2')
    manifest, _, _ = comparison(count=2)
    with pytest.raises(ValueError, match='business structures'):
        reseal(manifest, business_shape_hashes=[signature, signature])


def test_unsupported_domain_fails_before_online_calls(tmp_path):
    adapter, config, clients = adapter_for(tmp_path / 'native', 'saas')
    c, s = libraries('saas')
    with pytest.raises(ValueError, match='CloudOps Discovery only'):
        DynamicGate(adapter, EvidenceArchive(tmp_path / 'evidence'),
            GatePolicy(customer_quality=CustomerQualityPolicy())).customer(c, c, s,
                (ValidationCase(id='unsupported', config=config),), 'unsupported')
    assert clients == []


def test_pre_engine_tool_argument_failure_retained_not_customer_fault(tmp_path):
    wrong_query = op('p1','p1_db','get_instance', unexpected_parameter=1)
    report, _, _ = native_report(tmp_path, [[]], [[wrong_query]])
    result = report.probes[0].observed_service_outcomes[0]
    assert result['classification'] == 'tool_argument_failure'
    assert result['error_code'] == 'TOOL_ARGUMENT_INVALID'
    assert result['customer_fault'] == 'not_inferred_from_service_tool_choice'
    assert summarize((report,))['query_failures'] == 1
    assert not report.direct_discoveries
