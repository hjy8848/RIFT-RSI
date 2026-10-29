"""Host-driven preparation; explicit injected actors, no trainer invocation."""
from typing import Protocol
from rift.agents import propose
from rift.loop import LoopState
from rift.tasks import Certification, CertificationStatus
from rift.training.service_data import prepare_service_update

class BenchmarkHost(Protocol):
    def certify_proposal(self, proposal) -> tuple[Certification, object | None]: ...
    def rollout(self, task, service) -> object: ...

def prepare_round(package_id, experiences, registry, designer, host, service, checkpoint, recorder, *, algorithm='policy_gradient'):
    proposals = propose(designer, experiences, registry)
    state = LoopState(package_id, tuple(p.proposal_id for p in proposals))
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
