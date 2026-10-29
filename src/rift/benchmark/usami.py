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

class CommerceHost:
    """Trusted native host with deterministic first grounding and supplied scenario/profile factories."""
    def __init__(self, worlds, distribution, scenario_factory, profile_factory, customer_agent, *, proof_budget, episode_budget, grounding_budget=None):
        self.worlds = tuple(worlds)
        if distribution.split != "train" or any(spec.distribution_hash != distribution.record_hash for _, spec, _ in self.worlds):
            raise ValueError("Host requires a registered training distribution")
        self.distribution = distribution
        self.scenario_factory = scenario_factory
        self.profile_factory = profile_factory
        self.customer_agent = customer_agent
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
        self._tasks[task.record_hash] = (task, spec, build)
        return result, task
    def rollout(self, task, service):
        binding = self._tasks.get(task.record_hash)
        if binding is None or binding[0] != task: raise ValueError('task was not certified by this host')
        _, spec, build = binding
        profile = self.profile_factory(task, spec, build)
        return run_episode(build.world, task, profile, self.customer_agent, service, budget=self.episode_budget)
