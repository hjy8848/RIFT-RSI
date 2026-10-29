"""Explicit, lazy bindings to pinned Usami commerce v2 APIs; host-only."""
from rift.benchmark.private_evaluation import project_events, public_certification

def certify(proposal_id, world, task, customer_scenario, budget):
    from usami_v2.scenarios.task_certification import certify_task, verify_task_certificate
    certificate = certify_task(world, task, customer_scenario, budget)
    verification = verify_task_certificate(world, task, customer_scenario, certificate)
    return public_certification(proposal_id, certificate, verified=verification["verified"])

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
    """Conservative common policy for native structured remedy tasks."""
    if task.protocol != 'commerce-task-v3' or not task.remedy_goals or task.information_goals:
        raise ValueError('default Customer supports structured remedy-only tasks')
    allowed = set(task.remedy_goals[0].allowed_remedies)
    for goal in task.remedy_goals[1:]:
        allowed.intersection_update(goal.allowed_remedies)
    if not allowed:
        raise ValueError('task has no common Customer remedy policy')
    maximum = min(goal.maximum_extra_charge_cents if goal.maximum_extra_charge_cents is not None else 0
                  for goal in task.remedy_goals)
    deadlines = [goal.latest_arrival for goal in task.remedy_goals if goal.latest_arrival is not None]
    latest = min(deadlines) if deadlines else None
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
    """Trusted native host with deterministic first grounding and supplied scenario/profile factories."""
    def __init__(self, worlds, distribution, scenario_factory, profile_factory, customer_agent, *, registry, identity_factory, proof_budget, episode_budget, grounding_budget=None):
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
        self._tasks = {}
    def certify_proposal(self, proposal):
        from rift.tasks import Certification, CertificationStatus
        result = ground(proposal, self.worlds, self.grounding_budget)
        if not result.candidates:
            return Certification(proposal.proposal_id, CertificationStatus.UNKNOWN), None
        candidate = result.candidates[0]
        matches = [(spec, build) for ref, spec, build in self.worlds if ref == candidate.world_ref and build.world_hash == candidate.world_hash]
        if len(matches) != 1: raise ValueError('grounded world binding is ambiguous')
        spec, build = matches[0]
        task = compile_candidate(candidate, spec, build, self.distribution)
        scenario = self.scenario_factory(task, spec, build)
        result = certify(proposal.proposal_id, build.world, task, scenario, self.proof_budget)
        if result.status is not CertificationStatus.CERTIFIED: return result, None
        from rift.schemas import Identity, Split
        identity = self.identity_factory(task, spec, build)
        if (not isinstance(identity, Identity) or identity.split is not Split.TRAIN
                or identity.task != task.record_hash or identity.world != build.world_hash):
            raise ValueError('training identity must match the certified task and world')
        self.registry.register(identity)
        self._tasks[task.record_hash] = (task, spec, build, identity)
        return result, task
    def rollout(self, task, service):
        binding = self._tasks.get(task.record_hash)
        if binding is None or binding[0] != task: raise ValueError('task was not certified by this host')
        _, spec, build, identity = binding
        if not self.registry.contains(identity): raise ValueError('training identity is no longer registered')
        profile = self.profile_factory(task, spec, build)
        return run_episode(build.world, task, profile, self.customer_agent, service, budget=self.episode_budget)
