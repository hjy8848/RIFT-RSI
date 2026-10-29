"""Deterministic CloudOps compiler. No model-generated world or witness accepted."""
from collections import deque
import re
import json
from typing import Any, Dict, Literal, Optional, Tuple
from pydantic import Field
from vast_coevo.schema import FrozenModel
from vast_coevo.designer.storage import digest
from vast_coevo.skills.models import Hash
from vast_coevo.designer.models import Branch
from usami_cloudops.models import World, Operation, TOOLS
from usami_cloudops.rules import decide, invariants, POLICY
from usami_cloudops.observations import observation, visible_ids
from usami_harness.goals import JointGoal
from integrations.usami.examiner import ExamConfig, Proposal, certify, validate_world

FIELDS = {'projects': {'status', 'quota.cpu', 'quota.memory', 'quota.instances'},
          'access': {'role', 'user_id', 'project_id'},
          'instances': {'status', 'cpu', 'memory', 'dependencies', 'project_id'},
          'invoices': {'status', 'payments'}}


class SceneEdit(FrozenModel):
    collection: Literal['projects', 'access', 'instances', 'invoices']
    entity: str = Field(min_length=1)
    field: str = Field(min_length=1)
    expected_before: Any
    value: Any


class ChallengeProposal(FrozenModel):
    anchor_hash: Hash
    branch: Branch
    purpose: Literal['isolate', 'expand', 'boundary']
    hypothesis: str = Field(min_length=1, max_length=3000)
    edits: Tuple[SceneEdit, ...] = Field(default=(), max_length=12)
    identity: str
    customer_need: str = Field(min_length=1, max_length=3000)
    constraints: Tuple[str, ...] = Field(default=(), max_length=8)
    target: Operation
    capabilities: Tuple[Operation, ...] = Field(min_length=1, max_length=10)
    goal: Optional[JointGoal] = None


class Anchor(FrozenModel):
    evidence_hash: Hash
    world: dict
    identity: str
    target: Operation
    branch: Branch
    service_hash: Hash
    facts: dict
    anchor_hash: str = ''

    def checked(self):
        expected = digest(self.model_dump(mode='json', exclude={'anchor_hash'}))
        if self.anchor_hash and self.anchor_hash != expected:
            raise ValueError('anchor hash mismatch')
        return self.model_copy(update={'anchor_hash': expected})


class CompiledScene(FrozenModel):
    protocol: Literal['designer-service-v1'] = 'designer-service-v1'
    proposal: ChallengeProposal
    anchor: Anchor
    world: dict
    diff: Tuple[dict, ...]
    reference_before: dict
    reference_after: dict
    certificate: dict
    witness: Tuple[dict, ...]
    brief: dict
    search_nodes: int
    scene_hash: str = ''

    def checked(self):
        expected = digest(self.model_dump(mode='json', exclude={'scene_hash'}))
        if self.scene_hash and self.scene_hash != expected:
            raise ValueError('scene hash mismatch')
        return self.model_copy(update={'scene_hash': expected})


def expectation(world, identity, operation):
    result = decide(world, identity, operation)
    return dict(allowed=result.allowed, code=result.error_code)


class CloudOpsCompiler:
    domain = 'cloudops'

    def __init__(self, budget, max_nodes=512, max_depth=6):
        self.budget, self.max_nodes, self.max_depth = budget, max_nodes, max_depth

    def compile(self, anchor, proposal):
        anchor = Anchor.model_validate(anchor.model_dump(mode='json')).checked()
        proposal = ChallengeProposal.model_validate(proposal.model_dump(mode='json'))
        if proposal.anchor_hash != anchor.anchor_hash or proposal.branch != anchor.branch:
            raise ValueError('proposal source or bidirectional branch differs from verified anchor')
        if proposal.branch == 'error_amplification' and proposal.purpose == 'boundary':
            raise ValueError('error amplification must explicitly isolate or expand')
        old = World.model_validate(anchor.world)
        if invariants(old): raise ValueError('anchor prestate is not a valid business world')
        raw = old.model_dump(mode='json'); diff = []; touched = set()
        for edit in proposal.edits:
            key = (edit.collection, edit.entity, edit.field)
            if edit.field not in FIELDS[edit.collection] or key in touched:
                raise ValueError('field outside compiler whitelist or duplicate edit')
            touched.add(key)
            obj = raw[edit.collection].get(edit.entity)
            if obj is None: raise ValueError('edit refers to missing entity')
            parts = edit.field.split('.')
            for part in parts[:-1]: obj = obj[part]
            before = obj[parts[-1]]
            if digest(before) != digest(edit.expected_before): raise ValueError('edit precondition differs from anchor')
            if digest(before) == digest(edit.value): raise ValueError('no-op scene edit')
            obj[parts[-1]] = edit.value
            diff.append(dict(collection=edit.collection, entity=edit.entity, field=edit.field, before=before, after=edit.value))
        # Canonical mapping order before the FIRST query, not only on disk.
        # Native observations contain lists derived from mappings.
        world = validate_world('cloudops', json.loads(json.dumps(raw,sort_keys=True)), old)
        if proposal.target.name != anchor.target.name:
            raise ValueError('local search must preserve the anchor business operation type')
        if proposal.target not in proposal.capabilities or any(op.is_query for op in proposal.capabilities):
            raise ValueError('target must be an explicit write capability')
        view = observation(world, proposal.identity); visible = visible_ids(view)
        for op in proposal.capabilities:
            if op.project_id not in visible or op.resource_id not in visible:
                raise ValueError('business need refers to a Customer-invisible object')
        text = proposal.customer_need + ' '.join(proposal.constraints)
        from vast_coevo.skills.validation import _unsafe_instruction
        if _unsafe_instruction(text): raise ValueError('business brief attempts to change execution authority')
        all_ids={key for name in ('projects','instances','invoices','users','organizations','access') for key in getattr(world,name)}
        if any(re.search(r'(?<!\w)'+re.escape(key)+r'(?!\w)',text) for key in all_ids-visible):
            raise ValueError('business brief mentions Customer-invisible entities')
        if any(code in text for code in POLICY) or any(term in text.casefold() for term in ('oracle', 'witness', 'hypothesis', 'error_amplification', 'correct_to_failure', '<vast_')):
            raise ValueError('business brief contains evaluator or designer-only material')
        if proposal.goal is None:
            first=decide(world,proposal.identity,proposal.target)
            if first.allowed:
                proposal=proposal.model_copy(update={'goal':goal_from_transition(world,first.expected_state)})
        witness, nodes = [], 0
        if proposal.goal is not None:
            from integrations.usami.examiner import goal_adapter
            proposal.goal.validate_for(goal_adapter('cloudops', world), world)
            if proposal.goal.satisfied(world): raise ValueError('goal already satisfied')
            queue = deque([(world, [], frozenset())]); seen = set()
            while queue and nodes < self.max_nodes:
                state, path, used = queue.popleft(); nodes += 1
                if proposal.goal.satisfied(state) and proposal.goal.frame_preserved(world, state):
                    witness = path; break
                if len(path) >= min(self.max_depth, self.budget.max_service_tool_calls): continue
                for i, op in enumerate(proposal.capabilities):
                    if i in used: continue
                    decision = decide(state, proposal.identity, op)
                    if not decision.allowed: continue
                    next_used = used | {i}; key = (decision.expected_state.state_hash, next_used)
                    if key in seen: continue
                    seen.add(key); queue.append((decision.expected_state, path+[op], next_used))
            if not witness: raise ValueError('bounded Reference search found no reliable legal proof')
        kind = 'completion' if proposal.goal is not None else 'policy_refusal'
        native = Proposal(world=world.model_dump(mode='json'), identity=proposal.identity, message=proposal.customer_need,
            kind=kind, capabilities=[op.model_dump(mode='json') for op in proposal.capabilities], goal=proposal.goal,
            witness=[op.model_dump(mode='json') for op in witness], refusal_operation=proposal.target.model_dump(mode='json') if kind=='policy_refusal' else None,
            rationale=proposal.hypothesis)
        config = ExamConfig(domain='cloudops', template=old.model_dump(mode='json'), profile={},
                            episode={'tools': list(TOOLS), 'budget': self.budget.model_dump(mode='json')})
        _, certificate = certify(config, native)  # Real native Engine + Oracle + Replay for the chosen proof.
        brief = {'protocol': 'fixed-customer-v1', 'need': proposal.customer_need, 'constraints': list(proposal.constraints),
                 'capabilities': [op.model_dump(mode='json') for op in proposal.capabilities]}
        return CompiledScene(proposal=proposal, anchor=anchor, world=world.model_dump(mode='json'), diff=tuple(diff),
            reference_before=expectation(old, anchor.identity, anchor.target), reference_after=expectation(world, proposal.identity, proposal.target),
            certificate=certificate, witness=tuple(op.model_dump(mode='json') for op in witness), brief=brief, search_nodes=nodes).checked()


COMPILERS = {'cloudops': CloudOpsCompiler}  # Other domains opt in with their own original native bindings.


def source_group(anchor):
    """Cluster name/ID/seed-equivalent states, not independent repeated calls."""
    from integrations.usami.customer_probes import business_shape
    return digest({'business_shape':business_shape(World.model_validate(anchor.world),anchor.identity),
                   'operation':anchor.target.name})


def goal_from_transition(world, final):
    """Common deterministic single-write goal adapter, also available to Designer."""
    from integrations.usami.examiner import goal_adapter
    adapter=goal_adapter('cloudops',world);before=world.model_dump(mode='json');after=final.model_dump(mode='json')
    atoms=[];frame={}
    for collection,fields in adapter.GOAL_FIELDS.items():
        for entity in sorted(set(before[collection])|set(after[collection])):
            left=before[collection].get(entity);right=after[collection].get(entity)
            if left==right:continue
            if right is None:
                atoms.append(dict(collection=collection,entity=entity,comparison='absent'));frame[collection+'/'+entity]=['*'];continue
            changes=[f for f in fields if left is None or left.get(f)!=right.get(f)]
            if not changes:raise ValueError('transition has no supported observable goal')
            for field in changes:atoms.append(dict(collection=collection,entity=entity,field=field,value=right[field]))
            frame[collection+'/'+entity]=['*'] if left is None else changes
    if not atoms:raise ValueError('reference no-op is not a nontrivial completion challenge')
    return JointGoal.model_validate({'atoms':atoms,'allowed_changes':frame})
