"""Source/role separation and deterministic sampling; fixture proofs are explicit."""
from dataclasses import replace
import json

import pytest

from rsi_core.archive import Archive
from rsi_core.packages import PackagePoolExhausted, PackageProvider, PackageRegistration, TaskProof
from rsi_core.protocol import AdaptationPackage


def bank(archive, size=3):
    rows = []
    for purpose in ("C_probe", "C_select", "C_reward"):
        for i in range(size):
            prefix = purpose + str(i)
            package = AdaptationPackage(prefix, "protocol-fixture", purpose + "-family",
                purpose + "-lineage", "meta-train", purpose, "fixture-P0",
                ({"text": "public support " + prefix},), (prefix + "-probe",),
                (prefix + "-PRIVATE-query",), {"label": "public capability"}, seed=i)
            proofs = []
            for ref in package.probe_refs + package.query_refs:
                origin = "original-" + ref
                certificate = archive.save({"kind": "fixture-proof", "task_ref": ref,
                                             "origin_ref": origin})
                proofs.append(TaskProof(ref, origin, certificate))
            rows.append(PackageRegistration(package, (prefix + "-support-origin",), tuple(proofs)))
    return tuple(rows)


def fixture_verifier(archive):
    def verify(domain, proof):
        saved = archive.load(proof.certificate_ref)
        return domain == "protocol-fixture" and saved == {
            "kind": "fixture-proof", "task_ref": proof.task_ref, "origin_ref": proof.origin_ref}
    return verify


def test_reward_draws_rotate_without_replacement_and_resume_exact_history(tmp_path):
    archive = Archive(tmp_path)
    rows = bank(archive)
    provider = PackageProvider(rows, archive, fixture_verifier(archive))
    counts = {purpose: 1 for purpose in ("C_probe", "C_select", "C_reward")}
    first = provider.sample_training_context(7, counts=counts)
    saved = archive.load(first.archive_ref)
    for ref in saved["draw_refs"].values():
        assert provider.verify_draw(ref)["verified"]
    restored = PackageProvider(rows, archive, fixture_verifier(archive), state_ref=first.state_ref)
    second = restored.sample_training_context(7, counts=counts)
    matching = provider.sample_training_context(7, counts=counts)
    assert second.archive_ref == matching.archive_ref
    assert first.reward[0].package_id != second.reward[0].package_id
    third = restored.sample_training_context(7, counts=counts)
    assert len({first.reward[0].package_id, second.reward[0].package_id, third.reward[0].package_id}) == 3
    state = restored.state()
    with pytest.raises(PackagePoolExhausted):
        restored.sample_training_context(8, counts=counts)
    assert restored.state() == state


def test_failed_context_preflight_does_not_consume_other_pools(tmp_path):
    archive = Archive(tmp_path)
    rows = bank(archive, size=2)
    provider = PackageProvider(rows, archive, fixture_verifier(archive))
    provider.sample("meta-train", "C_reward", 0, count=2)
    state = provider.state()
    with pytest.raises(PackagePoolExhausted):
        provider.sample_training_context(1, counts={"C_probe": 1, "C_select": 1, "C_reward": 1})
    assert provider.state() == state
    assert len(provider.sample("meta-train", "C_probe", 2, count=2).packages) == 2


def test_same_original_data_cannot_be_renamed_across_roles_or_purposes(tmp_path):
    archive = Archive(tmp_path)
    rows = list(bank(archive, size=1))
    # A query exposed as public support under another purpose remains the
    # same original source even though all outward IDs differ.
    rows[1] = replace(rows[1], support_origins=(rows[0].task_proofs[-1].origin_ref,))
    with pytest.raises(ValueError, match="original source crosses"):
        PackageProvider(rows, archive, fixture_verifier(archive))


def test_manifest_snapshot_public_projection_and_certificate_failures(tmp_path):
    archive = Archive(tmp_path)
    rows = bank(archive, size=1)
    with pytest.raises(ValueError, match="independently verified"):
        PackageProvider(rows, archive, lambda domain, proof: False)
    provider = PackageProvider(rows, archive, fixture_verifier(archive))
    package = rows[0].package
    package.capabilities["mutated-after-registration"] = True
    package.support[0]["text"] = "mutated after registration"
    public = provider.public_view(package.package_id)
    assert "mutated" not in json.dumps(public)
    assert "PRIVATE-query" not in json.dumps(public)
    assert "certificate" not in json.dumps(public)
    public["support"][0]["text"] = "caller mutation"
    assert "caller mutation" not in json.dumps(provider.public_view(package.package_id))


def test_restoration_checks_actual_draw_and_bank_not_only_state_label(tmp_path):
    archive = Archive(tmp_path)
    rows = bank(archive, size=2)
    provider = PackageProvider(rows, archive, fixture_verifier(archive))
    draw = provider.sample("meta-train", "C_reward", 0)
    bad_draw = archive.save({**archive.load(draw.archive_ref), "package_ids": ["different-package"]})
    bad_state = archive.save({**archive.load(draw.state_ref), "draw_refs": [bad_draw]})
    with pytest.raises(ValueError, match="deterministic sampling"):
        PackageProvider(rows, archive, fixture_verifier(archive), state_ref=bad_state)
    with pytest.raises(ValueError, match="different package bank"):
        PackageProvider(rows[:-1], archive, fixture_verifier(archive), state_ref=draw.state_ref)


def test_duplicate_registration_and_test_reward_roles_are_rejected(tmp_path):
    archive = Archive(tmp_path)
    rows = bank(archive, size=1)
    with pytest.raises(ValueError, match="duplicate registered"):
        PackageProvider(rows + rows[:1], archive, fixture_verifier(archive))
    with pytest.raises(ValueError, match="validation/test"):
        replace(rows[0], package=replace(rows[0].package, partition="meta-test"))
    renamed = replace(rows[0], package=replace(rows[0].package, package_id="renamed-copy"))
    with pytest.raises(ValueError, match="query source is repeated"):
        PackageProvider(rows + (renamed,), archive, fixture_verifier(archive))
