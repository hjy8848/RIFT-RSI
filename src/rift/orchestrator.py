"""Host-driven preparation; explicit injected actors, no trainer invocation."""
from typing import Protocol
from rift.agents import propose
from rift.loop import LoopState, Stage
from rift.tasks import Certification, CertificationStatus
from rift.training.service_data import prepare_service_update

class BenchmarkHost(Protocol):
    def certify_proposal(self, proposal) -> tuple[Certification, object | None]: ...
    def rollout(self, task, service) -> object: ...

def prepare_round(package_id, experiences, registry, designer, host, service, checkpoint, recorder, *, algorithm='policy_gradient', initial_state=None):
    if getattr(host, 'registry', None) is not registry:
        raise ValueError('Designer and benchmark host must share the split registry')
    if initial_state is not None and (initial_state.stage is not Stage.PROPOSED or initial_state.package_id != package_id):
        raise ValueError('resumed round is not ready for this package')
    if initial_state is not None and initial_state.expected_service is not None and checkpoint != initial_state.expected_service:
        raise ValueError('Service policy does not continue the prior checkpoint')
    proposals = propose(designer, experiences, registry)
    state = initial_state or LoopState(package_id, tuple(p.proposal_id for p in proposals))
    if state.stage is not Stage.PROPOSED or state.package_id != package_id or state.proposal_ids != tuple(p.proposal_id for p in proposals):
        raise ValueError('resumed round does not match the proposed package')
    certs, tasks = [], []
    for proposal in proposals:
        cert, task = host.certify_proposal(proposal)
        if not isinstance(cert, Certification) or cert.proposal_id != proposal.proposal_id: raise ValueError('certificate does not match proposal')
        if cert.status is not CertificationStatus.CERTIFIED or task is None or task.record_hash != cert.task_ref: raise ValueError('proposal lacks a bound certified task')
        certs.append(cert); tasks.append(task)
    state = state.certify(certs)
    episodes = [host.rollout(task, service) for task in tasks]
    request = prepare_service_update(package_id, set(state.task_refs), episodes, checkpoint, recorder, algorithm=algorithm)
    return state.ready(request)
