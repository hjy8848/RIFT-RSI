"""Bind RSI task registrations to unchanged native Usami certificates.

Business reachability certificates do not assert complete dialogue success.
Their actual coverage is retained, never upgraded by the package registry.
"""
import inspect

from rsi_core.packages import TaskProof
from rsi_core.protocol import digest
from usami_v2.scenarios.task_certification import (
    CustomerScenario, TaskCertification, verify_task_certificate,
)

from .adapter import UsamiAdapter


def _binding(native):
    factory = native.customer_factory
    return {"world_hash": native.world.state_hash, "task_hash": native.task.checked().record_hash,
            "customer_profile_hash": native.customer_profile.checked().record_hash,
            "budget_hash": native.budget.checked().record_hash, "mode": native.mode,
            "external_events": [event.model_dump(mode="json") for event in native.external_events],
            "customer_factory_ref": digest({"module": factory.__module__, "name": factory.__qualname__,
                                             "source": inspect.getsource(factory)})}


class NativePackageVerifier:
    def __init__(self, tasks, archive):
        self.tasks, self.archive = dict(tasks), archive

    def register(self, task_ref, native, origin_ref, scenario, certificate):
        """Atomically add a frozen native task only after its Reference proof checks."""
        if not isinstance(task_ref, str) or not task_ref or task_ref in self.tasks:
            raise ValueError("native task reference is empty or already registered")
        self.tasks[task_ref] = native
        try:
            return self.issue(task_ref, origin_ref, scenario, certificate)
        except Exception:
            del self.tasks[task_ref]
            raise

    def issue(self, task_ref, origin_ref, scenario, certificate):
        native = self.tasks[task_ref]
        if scenario.customer_id != native.customer_profile.customer_id:
            raise ValueError("certificate Customer differs from registered runtime Customer")
        if set(scenario.delegated_line_ids) != set(native.customer_profile.delegated_line_ids):
            raise ValueError("certificate delegation differs from registered runtime delegation")
        if certificate.status != "CERTIFIED" or not certificate.business_reachability_certified:
            raise ValueError("native task lacks a certified business reachability path")
        verify_task_certificate(native.world, native.task, scenario, certificate)
        ref = self.archive.save({"kind": "rsi_usami_task_certificate_v1", "task_ref": task_ref,
                                 "origin_ref": origin_ref, "binding": _binding(native),
                                 "scenario": scenario.model_dump(mode="json"),
                                 "certificate": certificate.model_dump(mode="json"),
                                 "coverage": {"business": certificate.business_reachability_certified,
                                              "full_task": certificate.full_task_reachability_certified,
                                              "interaction": certificate.interaction_certified}})
        return TaskProof(task_ref, origin_ref, ref)

    def __call__(self, domain_id, proof):
        if domain_id != UsamiAdapter.domain_id:
            raise ValueError("task certificate belongs to another native domain")
        saved = self.archive.load(proof.certificate_ref)
        if (saved["kind"] != "rsi_usami_task_certificate_v1" or saved["task_ref"] != proof.task_ref
                or saved["origin_ref"] != proof.origin_ref):
            raise ValueError("native certificate belongs to another task or original source")
        native = self.tasks[proof.task_ref]
        if saved["binding"] != _binding(native):
            raise ValueError("native task/runtime changed after certificate registration")
        scenario = CustomerScenario.model_validate(saved["scenario"])
        certificate = TaskCertification.model_validate(saved["certificate"])
        if certificate.status != "CERTIFIED" or not certificate.business_reachability_certified:
            raise ValueError("unavailable native reachability certificate")
        return verify_task_certificate(native.world, native.task, scenario, certificate)["verified"]
