"""Explicit, lazy bindings to pinned Usami commerce v2 APIs; host-only."""
from dataclasses import dataclass
from hashlib import sha256
import time
from rift.benchmark.private_evaluation import project_events, public_certification


@dataclass(frozen=True)
class DesignerTrainingWorld:
    proposal_id: str
    distribution_hash: str
    world_ref: str
    spec: object
    build: object
    build_verification: dict
    challenge_results: tuple[dict, ...]
    certifications: tuple[object, ...]
    tasks: tuple[object, ...]

def certify_native(proposal_id, world, task, customer_scenario, budget):
    from usami_v2.scenarios.task_certification import certify_task, verify_task_certificate
    certificate = certify_task(world, task, customer_scenario, budget)
    verification = verify_task_certificate(world, task, customer_scenario, certificate)
    return (public_certification(proposal_id, certificate, verified=verification["verified"]),
            certificate if verification["verified"] else None)

def certify(proposal_id, world, task, customer_scenario, budget):
    return certify_native(proposal_id, world, task, customer_scenario, budget)[0]

def run_episode(*args, **kwargs):
    from usami_v2.runtime.runner import run_episode as native_run
    return native_run(*args, **kwargs)

def public_episode(record):
    from usami_v2.runtime.runner import replay_episode
    verification = replay_episode(record)
    return project_events(record, verification.verified)


def ground(proposal, worlds, budget=None):
    """Host-only candidates: never feed grounding bindings or witnesses to a model."""
    import json
    from usami_v2.scenarios.grounding import AbstractChallenge, ground_challenge
    challenge = AbstractChallenge.model_validate(json.loads(proposal.abstract_spec))
    if challenge.hypothesis != proposal.hypothesis:
        raise ValueError("proposal hypothesis differs from native challenge")
    return ground_challenge(challenge, worlds, budget)

def compile_candidate(candidate, spec, build, distribution, allowed_training_hashes=()):
    from usami_v2.scenarios.grounding import compile_grounded
    return compile_grounded(candidate, spec, build, distribution, allowed_training_hashes)

def _customer_limits(task, build):
    """Build Customer bounds from structured remedy goals, if any."""
    if task.protocol != 'commerce-task-v3':
        raise ValueError('default Customer requires a structured commerce-task-v3 task')
    if not task.remedy_goals:
        # Information-only tasks need no transaction delegation or consent.
        return (), 0, None, None
    allowed = {remedy for goal in task.remedy_goals for remedy in goal.allowed_remedies}
    capped_charges = [goal.maximum_extra_charge_cents for goal in task.remedy_goals
                      if goal.maximum_extra_charge_cents is not None]
    maximum = (max(capped_charges) if len(capped_charges) == len(task.remedy_goals)
               else 100_000_000)
    deadlines = [goal.latest_arrival for goal in task.remedy_goals if goal.latest_arrival is not None]
    latest = (None if len(deadlines) != len(task.remedy_goals) else max(deadlines))
    days = None if latest is None else (latest - build.world.clock).days
    if days is not None and days < 0:
        raise ValueError('Customer arrival deadline has already passed')
    return tuple(sorted(allowed)), maximum, latest, days

def default_customer_scenario(task, spec, build):
    from usami_v2.scenarios.task_certification import CustomerScenario
    allowed, maximum, latest, _ = _customer_limits(task, build)
    return CustomerScenario(customer_id=task.customer_id,
        delegated_line_ids=tuple(g.line_id for g in task.remedy_goals),
        accepted_remedies=allowed, maximum_extra_charge_cents=maximum, latest_arrival=latest)

def default_customer_profile(task, spec, build):
    from usami_v2.runtime.customer import CustomerProfile, bind_customer_profile
    allowed, maximum, _, days = _customer_limits(task, build)
    profile = CustomerProfile(customer_id=task.customer_id,
        accepted_remedies=allowed, maximum_extra_charge_cents=maximum, latest_arrival_days=days)
    return bind_customer_profile(profile, task)

class CommerceHost:
    """Trusted native host with bounded grounding and certified task storage."""
    def __init__(self, worlds, distribution, scenario_factory, profile_factory, customer_agent, *, registry, identity_factory, proof_budget, episode_budget, grounding_budget=None, task_registry=None):
        self.worlds = tuple(worlds)
        if distribution.split != "train" or any(spec.distribution_hash != distribution.record_hash for _, spec, _ in self.worlds):
            raise ValueError("Host requires a registered training distribution")
        self.distribution = distribution
        self.scenario_factory = scenario_factory or default_customer_scenario
        self.profile_factory = profile_factory or default_customer_profile
        self.customer_agent = customer_agent
        self.registry = registry
        self.identity_factory = identity_factory
        self.proof_budget = proof_budget
        self.episode_budget = episode_budget
        self.grounding_budget = grounding_budget
        from rift.verl.task_registry import CertifiedTaskRegistry
        self.task_registry = task_registry or CertifiedTaskRegistry(registry)
        if self.task_registry.split_registry is not registry:
            raise ValueError('certified task registry must use the Host split registry')
    def certify_proposal(self, proposal):
        from rift.tasks import Certification, CertificationStatus
        from usami_v2.scenarios.designer import TaskCapabilityUnsupported
        result = ground(proposal, self.worlds, self.grounding_budget)
        if not result.candidates:
            return Certification(proposal.proposal_id, CertificationStatus.UNKNOWN), None
        # Native Grounding has already sorted and capped candidates by its
        # deterministic budget. A non-certified binding must not mask the next.
        last_rejection = unresolved = None
        for candidate in result.candidates:
            matches = [(spec, build) for ref, spec, build in self.worlds
                       if ref == candidate.world_ref and build.world_hash == candidate.world_hash]
            if len(matches) != 1:
                raise ValueError('grounded world binding is ambiguous')
            spec, build = matches[0]
            try:
                task = compile_candidate(candidate, spec, build, self.distribution)
            except TaskCapabilityUnsupported:
                unresolved = Certification(proposal.proposal_id, CertificationStatus.UNKNOWN)
                continue
            scenario = self.scenario_factory(task, spec, build)
            certification, native_certificate = certify_native(
                proposal.proposal_id, build.world, task, scenario, self.proof_budget)
            if certification.status is not CertificationStatus.CERTIFIED:
                if certification.status is CertificationStatus.UNKNOWN:
                    unresolved = certification
                else:
                    last_rejection = certification
                continue
            from rift.schemas import Identity, Split
            identity = self.identity_factory(task, spec, build)
            if (not isinstance(identity, Identity) or identity.split is not Split.TRAIN
                    or identity.task != task.record_hash or identity.world != build.world_hash):
                raise ValueError('training identity must match the certified task and world')
            profile = self.profile_factory(task, spec, build)
            from rift.verl.task_registry import CertifiedNativeTask
            self.registry.register(identity)
            self.task_registry.register(CertifiedNativeTask(
                build.world, task, profile, scenario, native_certificate, identity))
            return certification, task
        return unresolved or last_rejection or Certification(
            proposal.proposal_id, CertificationStatus.UNKNOWN), None

    def certify_training_plan(self, proposal, *, world_seed, world_build_budget=None,
                              allowed_training_hashes=(), max_candidate_challenges=32,
                              metrics=None):
        """Build one native World, then ground and certify each native challenge in it."""
        from rift.tasks import CertificationStatus
        if not getattr(proposal, 'is_training_plan', False):
            raise ValueError('a package-level Designer World and challenge plan is required')
        if (type(world_seed) is not int or world_seed < 0
                or type(max_candidate_challenges) is not int or max_candidate_challenges < 1
                or len(proposal.challenges) > max_candidate_challenges):
            raise ValueError('bounded World construction seed and challenge budget required')

        from usami_v2.scenarios.builder import build_world, verify_build
        from usami_v2.scenarios.designer import ScenarioDraft, TaskCapabilityUnsupported
        from usami_v2.scenarios.grounding import AbstractChallenge, ground_challenge

        draft = ScenarioDraft.model_validate(proposal.world_draft).checked()
        spec = draft.compile(self.distribution)
        started = time.perf_counter()
        build = build_world(spec, world_seed, world_build_budget)
        if metrics is not None:
            metrics('world_build_seconds', time.perf_counter() - started,
                    proposal_id=proposal.proposal_id)
        build_verification = verify_build(spec, build)
        world_ref = sha256(f"rift-designer-world-v1:{proposal.proposal_id}:{spec.record_hash}:{world_seed}:{build.record_hash}".encode()).hexdigest()[:32]
        if build.status != 'SAT' or build_verification.get('verified') is not True:
            return DesignerTrainingWorld(proposal.proposal_id, self.distribution.record_hash,
                world_ref, spec, build, build_verification, (), (), ())

        worlds = ((world_ref, spec, build),)
        challenge_results, certifications, tasks = [], [], []
        for raw_challenge in proposal.challenges:
            certification_started = time.perf_counter()
            challenge = AbstractChallenge.model_validate(raw_challenge).checked()
            grounding = ground_challenge(challenge, worlds, self.grounding_budget)
            attempts, selected_task = [], None
            if grounding.candidates:
                for index, candidate in enumerate(grounding.candidates):
                    try:
                        task = compile_candidate(candidate, spec, build, self.distribution,
                                                 allowed_training_hashes)
                    except TaskCapabilityUnsupported as exc:
                        attempts.append({'candidate_index':index, 'task_ref':None,
                            'status':'UNKNOWN', 'native_status':'UNSUPPORTED',
                            'reason':str(exc), 'reason_code':exc.code})
                        continue
                    scenario = self.scenario_factory(task, spec, build)
                    public_result, native_certificate = certify_native(
                        proposal.proposal_id, build.world, task, scenario, self.proof_budget)
                    native_status = 'UNKNOWN' if native_certificate is None else native_certificate.status
                    reasons = () if native_certificate is None else tuple(native_certificate.blockers)
                    attempts.append({'candidate_index':index, 'task_ref':task.record_hash,
                        'status':public_result.status.value, 'native_status':native_status,
                        'reason':list(reasons)})
                    if public_result.status is not CertificationStatus.CERTIFIED:
                        continue
                    identity = self.identity_factory(task, spec, build)
                    from rift.schemas import DataPurpose, Identity, Split
                    if (not isinstance(identity, Identity) or identity.split is not Split.TRAIN
                            or identity.purpose is not DataPurpose.TRAIN
                            or identity.task != task.record_hash or identity.world != build.world_hash
                            or task.partition != 'train'):
                        raise ValueError('Designer task identity must bind this verified TRAIN World')
                    profile = self.profile_factory(task, spec, build)
                    from rift.verl.task_registry import CertifiedNativeTask
                    self.registry.register(identity)
                    self.task_registry.register(CertifiedNativeTask(
                        build.world, task, profile, scenario, native_certificate, identity))
                    certifications.append(public_result)
                    tasks.append(task)
                    selected_task = task
                    break
            if selected_task is not None:
                status = 'CERTIFIED'
                reasons = []
            elif not grounding.candidates:
                status = grounding.status
                reasons = [grounding.status]
            else:
                native_statuses = {row['native_status'] for row in attempts}
                status = ('UNKNOWN' if 'UNKNOWN' in native_statuses or 'UNSUPPORTED' in native_statuses
                          else 'UNSAT')
                reasons = [row.get('reason_code') or row.get('reason') or row['native_status']
                           for row in attempts]
            challenge_results.append({
                'challenge_id':challenge.task_id,
                'grounding_status':grounding.status,
                'search_truncated':grounding.search_truncated,
                'candidate_count':len(grounding.candidates),
                'attempts':attempts,
                'status':status,
                'reasons':reasons,
                'selected_task_ref':None if selected_task is None else selected_task.record_hash,
            })
            if metrics is not None:
                metrics('task_certification_seconds', time.perf_counter() - certification_started,
                        proposal_id=proposal.proposal_id, challenge_id=challenge.task_id,
                        candidate_count=len(attempts))
        return DesignerTrainingWorld(proposal.proposal_id, self.distribution.record_hash,
            world_ref, spec, build, build_verification, tuple(challenge_results),
            tuple(certifications), tuple(tasks))

    def rollout(self, task, service):
        record = self.task_registry.resolve_record(task.record_hash)
        if record.task != task or not self.task_registry.verify_task(
                task.record_hash, record.world, task, record.profile):
            raise ValueError('task was not certified by this host')
        return run_episode(record.world, task, record.profile, self.customer_agent, service,
                           budget=self.episode_budget, external_events=record.external_events)
