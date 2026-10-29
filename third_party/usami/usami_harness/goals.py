"""Declarative goals, bounded legal search and omission-based causal evidence."""
from collections import deque
from typing import Any, Dict, List, Literal, Optional
from pydantic import Field
from .models import Model, Request, digest
from .engine import Engine, replay_trace
from .oracle import evaluate


def field_value(value, path):
    if hasattr(value, 'model_dump'):
        value = value.model_dump(mode='json')
    for part in path.split('.') if path else []:
        if isinstance(value, dict):
            if part not in value:
                raise ValueError('unknown predicate field')
            value = value[part]
        elif isinstance(value, list) and part.isdecimal() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise ValueError('invalid predicate field path')
    return value


class AtomicGoal(Model):
    collection: str
    entity: str
    field: str = ''
    comparison: Literal['eq', 'ge', 'le', 'absent'] = 'eq'
    value: Any = None

    def satisfied(self, world):
        value = getattr(world, self.collection).get(self.entity)
        if self.comparison == 'absent':
            return value is None
        if value is None:
            return False
        try:
            actual = field_value(value, self.field)
            if self.comparison == 'eq':
                return actual == self.value
            return actual >= self.value if self.comparison == 'ge' else actual <= self.value
        except (ValueError, TypeError):
            return False


class JointGoal(Model):
    atoms: List[AtomicGoal] = Field(min_length=1)
    allowed_changes: Dict[str, List[str]]

    def validate_for(self, adapter, world):
        assignments, bounds = {}, {}
        for atom in self.atoms:
            if atom.collection not in adapter.GOAL_FIELDS:
                raise ValueError('predicate collection not in domain allowlist')
            if atom.comparison != 'absent' and (not atom.field or atom.field.split('.')[0] not in adapter.GOAL_FIELDS[atom.collection]):
                raise ValueError('predicate field not in domain allowlist')
            value = getattr(world, atom.collection).get(atom.entity)
            if value is not None and atom.comparison != 'absent':
                field_value(value, atom.field)
            key = (atom.collection, atom.entity, atom.field)
            if atom.comparison == 'eq':
                if key in assignments and assignments[key] != atom.value:
                    raise ValueError('incompatible atomic goal assignments')
                assignments[key] = atom.value
            elif atom.comparison in {'ge','le'}:
                if isinstance(atom.value,bool) or not isinstance(atom.value,int):
                    raise ValueError('ordered goal comparisons require integer bounds')
                lower,upper=bounds.get(key,(None,None))
                if atom.comparison=='ge':lower=max(lower,atom.value) if lower is not None else atom.value
                else:upper=min(upper,atom.value) if upper is not None else atom.value
                bounds[key]=(lower,upper)
        for key,(lower,upper) in bounds.items():
            if lower is not None and upper is not None and lower>upper:
                raise ValueError('incompatible atomic goal bounds')
            if key in assignments:
                value=assignments[key]
                if isinstance(value,bool) or not isinstance(value,int) or (lower is not None and value<lower) or (upper is not None and value>upper):
                    raise ValueError('exact atomic goal conflicts with its bounds')
        for key, fields in self.allowed_changes.items():
            collection, entity = key.split('/')
            if collection not in adapter.FRAME_FIELDS or not fields or len(fields) != len(set(fields)):
                raise ValueError('frame collection or fields not allowed')
            if '*' in fields:
                if fields != ['*'] or (entity in getattr(world, collection) and collection not in adapter.DELETABLE):
                    raise ValueError('whole-entity frame exemption allowed only for new entities or domain deletions')
            elif set(fields) - set(adapter.FRAME_FIELDS[collection]):
                raise ValueError('immutable or unknown frame field')
        # An absent predicate conflicts with another terminal property of the same entity.
        absent = {(a.collection, a.entity) for a in self.atoms if a.comparison == 'absent'}
        if any((a.collection, a.entity) in absent and a.comparison != 'absent' for a in self.atoms):
            raise ValueError('absent entity cannot also have terminal fields')
        return self

    def satisfied(self, world):
        return all(a.satisfied(world) for a in self.atoms)

    def frame_preserved(self, initial, final):
        left, right = initial.model_dump(mode='json'), final.model_dump(mode='json')
        for key, fields in self.allowed_changes.items():
            collection, entity = key.split('/')
            for value in (left, right):
                if fields == ['*']:
                    value[collection].pop(entity, None)
                elif entity in value[collection]:
                    for field in fields:
                        value[collection][entity].pop(field, None)
        return left == right


class Witness(Model):
    operations: List[Dict[str, Any]]
    direct_edges: List[Dict[str, Any]]
    mediated_failures: List[Dict[str, Any]]
    omissions: List[Dict[str, Any]]
    causal_depth: int
    final_state_hash: str
    trace_hash: str


class Task(Model):
    domain: str
    version: str
    id: str = 'pending'
    family: str
    identity: str
    kind: Literal['completion', 'policy_refusal'] = 'completion'
    message: str
    initial_state_hash: str
    goal: JointGoal
    capabilities: List[Dict[str, Any]]
    witnesses: List[Witness] = Field(default_factory=list)
    refusal_code: Optional[str] = None
    search_nodes: int = 0
    search_exhausted: bool = False
    max_search_depth: int = 18
    max_search_nodes: int = 8192
    structure_signature: str = ''
    semantic_signature: str = ''


def execute(adapter, world, identity, path):
    engine = Engine(adapter, world, max(1, len(path)))
    for index, operation in enumerate(path):
        op = adapter.Operation.model_validate(operation.model_dump() if hasattr(operation, 'model_dump') else operation)
        request = Request(id=f'proof_{index:03d}', identity=identity, operation=op.model_dump(mode='json'))
        engine.register(request)
        event = engine.execute(identity, op, request.id)
        if not event.payload['result']['success']:
            raise ValueError('reference witness rejected: ' + str(event.payload['result']))
    trace = engine.finish()
    report = evaluate(adapter, trace, engine.requests)
    if report['status'] != 'pass' or replay_trace(adapter, trace, engine.requests).state != engine.state:
        raise ValueError('Reference/actual Engine/independent Oracle/replay disagreement')
    return trace


def reachability(adapter, world, task, max_paths=4):
    task.goal.validate_for(adapter, world)
    operations = [adapter.Operation.model_validate(op) for op in task.capabilities]
    if any(op.is_query for op in operations):
        raise ValueError('queries cannot constitute a business witness')
    queue, seen, paths, nodes = deque([(world.clone(), [], frozenset())]), {}, [], 0
    while queue and nodes < task.max_search_nodes and len(paths) < max_paths:
        state, path, used = queue.popleft()
        nodes += 1
        if task.goal.satisfied(state) and task.goal.frame_preserved(world, state):
            # Exclude legal no-op padding and a step that can be omitted while all
            # subsequent steps still execute and meet the complete terminal goal.
            irreducible = True
            for omitted in range(len(path)):
                alternate = world.clone()
                for i, operation in enumerate(path):
                    if i == omitted:
                        continue
                    decision = adapter.decide(alternate, task.identity, operation)
                    if not decision.allowed:
                        break
                    alternate = decision.expected_state
                else:
                    if task.goal.satisfied(alternate) and task.goal.frame_preserved(world, alternate):
                        irreducible = False
                        break
            if irreducible and path:
                paths.append(path)
            continue
        if len(path) >= task.max_search_depth:
            continue
        for index, operation in enumerate(operations):
            if index in used:
                continue
            decision = adapter.decide(state, task.identity, operation)
            if not decision.allowed or decision.expected_state == state:
                continue
            following = decision.expected_state
            next_used = used | {index}
            key = (following.state_hash, next_used)
            if seen.get(key, 0) >= max_paths:
                continue
            seen[key] = seen.get(key, 0) + 1
            queue.append((following, path + [operation], next_used))
    return paths, nodes, not queue


def certify(adapter, world, task, path):
    trace = execute(adapter, world, task.identity, path)
    final = adapter.World.model_validate(trace.final_state)
    if not task.goal.satisfied(final) or not task.goal.frame_preserved(world, final):
        raise ValueError('witness does not satisfy goal and frame')
    direct, mediated, omissions, prefixes = [], [], [], []
    state = world.clone()
    for operation in path:
        prefixes.append(state)
        state = adapter.decide(state, task.identity, operation).expected_state
    for omitted in range(len(path)):
        alternate, failed = prefixes[omitted].clone(), False
        first_failure = None
        for later in range(omitted + 1, len(path)):
            decision = adapter.decide(alternate, task.identity, path[later])
            if not decision.allowed:
                edge = {'from': omitted, 'to': later, 'error_code': decision.error_code}
                if not failed:
                    direct.append(edge)
                    first_failure = edge
                else:
                    mediated.append(edge)
                failed = True
            else:
                alternate = decision.expected_state
        omissions.append({'omitted_step': omitted, 'first_failure': first_failure,
            'goal_satisfied_after_remaining_legal_steps': task.goal.satisfied(alternate),
            'frame_preserved_after_remaining_legal_steps': task.goal.frame_preserved(world, alternate)})
    levels = [0] * len(path)
    for edge in sorted(direct, key=lambda e: (e['to'], e['from'])):
        levels[edge['to']] = max(levels[edge['to']], levels[edge['from']] + 1)
    return Witness(operations=[o.model_dump(mode='json') for o in path], direct_edges=direct,
        mediated_failures=mediated, omissions=omissions, causal_depth=max(levels, default=0),
        final_state_hash=final.state_hash, trace_hash=trace.trace_hash)


def freeze_task(adapter, world, task):
    if task.domain != adapter.name or task.version != adapter.version or adapter.invariants(world):
        raise ValueError('invalid conditioned domain world')
    if task.initial_state_hash != world.state_hash:
        raise ValueError('task initial state pin differs')
    task.goal.validate_for(adapter, world)
    if task.kind == 'policy_refusal':
        if len(task.capabilities) != 1 or task.goal.allowed_changes:
            raise ValueError('refusal proof requires one exact probe and total frame preservation')
        operation = adapter.Operation.model_validate(task.capabilities[0])
        decision = adapter.decide(world, task.identity, operation)
        if decision.allowed or decision.error_code != task.refusal_code:
            raise ValueError('independent refusal certificate does not match')
        engine = Engine(adapter, world, 1)
        request = Request(id='refusal', identity=task.identity, operation=operation.model_dump(mode='json'))
        engine.register(request)
        event = engine.execute(task.identity, operation, request.id)
        trace = engine.finish()
        if (event.payload['result']['success'] or engine.state != world
                or evaluate(adapter, trace, engine.requests)['status'] != 'pass'
                or replay_trace(adapter, trace, engine.requests).state != world):
            raise ValueError('refusal Engine/Oracle/replay mismatch')
        task.search_exhausted = True
    else:
        paths, task.search_nodes, task.search_exhausted = reachability(adapter, world, task)
        if not paths:
            raise ValueError('no certified reachable goal inside declared finite capabilities and bounds')
        task.witnesses = [certify(adapter, world, task, p) for p in paths]
    task.structure_signature, task.semantic_signature = adapter.task_signatures(world, task)
    task.id = adapter.name + '_task_' + digest(task.model_dump(mode='json', exclude={'id'}))[:24]
    return task
