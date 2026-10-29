"""CPU only: native Usami fixtures exercise actual certification and engine/replay."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'third_party' / 'usami' / 'tests'))
from test_usami_v2_task_certification import frozen_world, make_task, customer_scenario
from test_usami_v2_runtime import frozen, refund_task, refund_profile, ScriptedCustomer, ScriptedRefundService
from usami_v2.scenarios.task_certification import ProofBudget
from usami_v2.runtime.runner import EpisodeBudget
from rift.benchmark.usami import certify, run_episode, public_episode
from rift.schemas import Outcome
from rift.tasks import CertificationStatus

def test_native_certification():
    world, aliases = frozen_world()
    task = make_task(world, aliases, attributes=({'name':'interface','value':'USB_C'},))
    result = certify('proposal', world, task, customer_scenario(aliases), ProofBudget(max_plan_candidates=16, max_operation_steps=16, max_witnesses=2))
    assert result.status is CertificationStatus.CERTIFIED and result.task_ref == task.record_hash
    assert set(result.__dict__) == {'proposal_id', 'status', 'task_ref'}

def test_native_episode_execution_replay_and_projection():
    world, aliases = frozen()
    episode = run_episode(world, refund_task(world, aliases), refund_profile(aliases), ScriptedCustomer(), ScriptedRefundService(world, aliases), budget=EpisodeBudget(max_customer_turns=3,max_service_decisions=10,max_service_tool_calls=8,max_model_calls=0))
    public = public_episode(episode)
    assert public.outcome is Outcome.SUCCESS
    assert any(row.kind == 'tool_result' for row in public.events)
    assert 'oracle' not in repr(public).lower()

def test_native_grounding():
    import json
    from test_usami_v2_builder import scene
    from usami_v2.scenarios.builder import build_world
    from rift.tasks import Proposal
    from rift.benchmark.usami import ground
    spec = scene(); build = build_world(spec, 13)
    data = {'task_id':'grounded-task', 'customer_alias':'customer', 'hypothesis':'preconditions',
        'remedy_goals':[{'goal_id':'g','line_alias':'item','quantity':1,'allowed_remedies':['refund']}],
        'line_selectors':[{'alias':'item','family':'device','issue':'damaged'}]}
    proposal = Proposal('p', 'preconditions', 'Please help', json.dumps(data))
    result = ground(proposal, [('world', spec, build)])
    assert result.status == 'GROUNDED' and result.candidates
    assert result.candidates[0].world_hash == build.world_hash

def test_native_ground_compile_certify_chain():
    import json
    from test_usami_v2_designer import distribution_and_slot, scenario_draft
    from usami_v2.scenarios.builder import build_world
    from usami_v2.scenarios.task_certification import CustomerScenario
    from rift.tasks import Proposal
    from rift.benchmark.usami import ground, compile_candidate
    distribution, _ = distribution_and_slot(13)
    spec = scenario_draft().compile(distribution); build = build_world(spec, 13)
    data = {'task_id':'grounded-task','customer_alias':'customer','hypothesis':'missing precondition',
        'remedy_goals':[{'goal_id':'g','line_alias':'item','quantity':1,'allowed_remedies':['refund']}],
        'line_selectors':[{'alias':'item','family':'device','issue':'damaged'}]}
    proposal = Proposal('p', 'missing precondition', 'Please refund', json.dumps(data))
    candidates = ground(proposal, [('world', spec, build)]).candidates
    task = compile_candidate(candidates[0], spec, build, distribution)
    scenario = CustomerScenario(customer_id=task.customer_id, delegated_line_ids=tuple(g.line_id for g in task.remedy_goals), maximum_extra_charge_cents=0)
    result = certify(proposal.proposal_id, build.world, task, scenario, ProofBudget())
    assert result.status is CertificationStatus.CERTIFIED and result.task_ref == task.record_hash

def test_scripted_native_episode_cannot_be_replay_demonstration():
    from rift.replay.trajectory import extract_demonstrations, DecisionEvidenceVerifier
    from rift.replay.splits import SplitRegistry
    from rift.schemas import Identity, Split, Checkpoint
    import pytest
    world, aliases = frozen()
    record = run_episode(world, refund_task(world, aliases), refund_profile(aliases), ScriptedCustomer(), ScriptedRefundService(world, aliases), budget=EpisodeBudget(max_customer_turns=3,max_service_decisions=10,max_service_tool_calls=8,max_model_calls=0))
    identity = Identity(world.state_hash, record.task.record_hash, 'f', 'source', Split.REPLAY)
    class Verifier(DecisionEvidenceVerifier):
        def verify_source(self, episode, checkpoint): return True
    with pytest.raises(ValueError, match='real Service model'):
        extract_demonstrations(record, identity, Checkpoint('service','r','t','w'), SplitRegistry([identity]), Verifier())

def test_scripted_episode_cannot_generate_service_training_data():
    from rift.training.service_data import prepare_service_update
    from rift.schemas import Checkpoint
    import pytest
    world, aliases = frozen()
    record = run_episode(world, refund_task(world, aliases), refund_profile(aliases), ScriptedCustomer(), ScriptedRefundService(world, aliases), budget=EpisodeBudget(max_customer_turns=3,max_service_decisions=10,max_service_tool_calls=8,max_model_calls=0))
    class Recorder:
        def verify_source(self, episode, checkpoint): return True
    with pytest.raises(ValueError, match='real Service model'):
        prepare_service_update('package', {record.task.record_hash}, [record], Checkpoint('service','r','t','w'), Recorder())

def test_native_commerce_host_certification():
    import json
    from test_usami_v2_designer import distribution_and_slot, scenario_draft
    from usami_v2.scenarios.builder import build_world
    from usami_v2.scenarios.task_certification import CustomerScenario
    from rift.tasks import Proposal
    from rift.benchmark.usami import CommerceHost
    distribution, _ = distribution_and_slot(13)
    spec = scenario_draft().compile(distribution); build = build_world(spec, 13)
    data = {'task_id':'grounded-task','customer_alias':'customer','hypothesis':'missing precondition',
        'remedy_goals':[{'goal_id':'g','line_alias':'item','quantity':1,'allowed_remedies':['refund']}],
        'line_selectors':[{'alias':'item','family':'device','issue':'damaged'}]}
    def scenario(task, spec, build):
        return CustomerScenario(customer_id=task.customer_id, delegated_line_ids=tuple(g.line_id for g in task.remedy_goals), maximum_extra_charge_cents=0)
    host = CommerceHost([('world', spec, build)], distribution, scenario, None, None, proof_budget=ProofBudget(), episode_budget=EpisodeBudget())
    cert, task = host.certify_proposal(Proposal('p','missing precondition','refund',json.dumps(data)))
    assert cert.status is CertificationStatus.CERTIFIED and cert.task_ref == task.record_hash
