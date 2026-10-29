"""Registered native proofs retain actual certificate coverage and bindings."""
from dataclasses import replace

import pytest

from integrations.rsi_usami.adapter import NativeTask, UsamiAdapter
from integrations.rsi_usami.packages import NativePackageVerifier
from rsi_core.archive import Archive
from rsi_core.packages import PackageProvider, PackageRegistration
from rsi_core.protocol import AdaptationPackage
from usami_v2.runtime.runner import EpisodeBudget
from usami_v2.scenarios.task_certification import CustomerScenario, certify_task
from test_usami_v2_runtime import frozen, refund_profile, refund_task, ScriptedCustomer


def native(seed):
    world, aliases = frozen(seed)
    task, profile = refund_task(world, aliases), refund_profile(aliases)
    scenario = CustomerScenario(customer_id=task.customer_id, delegated_line_ids=profile.delegated_line_ids)
    certificate = certify_task(world, task, scenario)
    return NativeTask(world, task, profile, ScriptedCustomer, EpisodeBudget()), scenario, certificate


def test_actual_native_certificates_register_and_keep_private_witnesses_out_of_public_view(tmp_path):
    first, scenario1, cert1 = native(17)
    second, scenario2, cert2 = native(18)
    archive = Archive(tmp_path)
    verifier = NativePackageVerifier({"native-probe": first, "native-query": second}, archive)
    proofs = (verifier.issue("native-probe", "fixture-origin-17", scenario1, cert1),
              verifier.issue("native-query", "fixture-origin-18", scenario2, cert2))
    package = AdaptationPackage("native-package", UsamiAdapter.domain_id, "fixture-family",
        "fixture-lineage", "meta-train", "C_reward", "simple-P0", (),
        ("native-probe",), ("native-query",), {"score": "native strict task success"})
    provider = PackageProvider((PackageRegistration(package, (), proofs),), archive, verifier)
    sampled = provider.sample("meta-train", "C_reward", 1)
    assert sampled.packages[0] == package
    public = provider.public_view(package.package_id)
    assert "native-query" not in str(public)
    assert "witness" not in str(public)
    saved = archive.load(proofs[0].certificate_ref)
    assert saved["coverage"]["full_task"] == cert1.full_task_reachability_certified
    assert saved["coverage"]["interaction"] == cert1.interaction_certified
    verifier.tasks["native-probe"] = replace(first, mode="sandbox")
    with pytest.raises(ValueError, match="runtime changed"):
        verifier(UsamiAdapter.domain_id, proofs[0])


def test_certificate_cannot_be_renamed_or_used_with_different_customer_delegation(tmp_path):
    task, scenario, certificate = native(17)
    verifier = NativePackageVerifier({"one": task, "alias": task}, Archive(tmp_path))
    proof = verifier.issue("one", "original-one", scenario, certificate)
    with pytest.raises(ValueError, match="another task or original source"):
        verifier(UsamiAdapter.domain_id, replace(proof, task_ref="alias"))
    with pytest.raises(ValueError, match="delegation differs"):
        verifier.issue("one", "original-one", CustomerScenario(customer_id=scenario.customer_id,
                                                               delegated_line_ids=()), certificate)
    # A model_copy retaining a stale record_hash cannot pass the binding check.
    corrupted = task.customer_profile.model_copy(update={"delegation_request_spans": {}})
    verifier.tasks["one"] = replace(task, customer_profile=corrupted)
    with pytest.raises(ValueError, match="content hash differs"):
        verifier(UsamiAdapter.domain_id, proof)
