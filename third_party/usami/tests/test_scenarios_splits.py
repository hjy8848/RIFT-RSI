from itertools import islice

import pytest

from usami import GeneratorConfig, generate_scenarios, generate_world, evaluate_trace, replay_trace
from usami.evaluation.scenario import derive_scenario, simulate_scenario, validate_scenario
from usami.evaluation.splits import (assert_fair_comparison, business_fingerprint, create_manifest,
                                     load_manifest, sample_train, validate_manifest, validate_splits)
from usami.world.generator import composition_signature
from usami.world.models import digest
from test_world import operation


def test_scenario_labels_and_traces_across_generated_worlds():
    categories, labels = set(), set()
    for seed in range(6):
        world = generate_world(seed)
        scenarios = generate_scenarios(world)
        assert scenarios == generate_scenarios(world)
        for scenario in scenarios:
            categories.add(scenario.category)
            labels.update(s.expected_allowed for s in scenario.steps)
            validate_scenario(world, scenario)
            trace = simulate_scenario(world, scenario)
            assert trace.final_state_hash == scenario.steps[-1].expected_state_hash
            report = evaluate_trace(trace)
            assert report.status == "pass", report.model_dump_json()
            assert replay_trace(trace).state == trace.final_state
    assert labels == {True, False}
    assert {"multi_step_plan", "multi_step_refund", "multi_entity", "cross_organization", "permission", "authorization", "query"} <= categories


def test_multistep_dependency_changes_outcome(world):
    sequence = [("user_000", operation("change_plan", plan_id="team")),
                ("user_000", operation("change_seats", seats=8)),
                ("user_000", operation("change_plan", plan_id="starter")),
                ("user_000", operation("change_seats", seats=4)),
                ("user_000", operation("change_plan", plan_id="starter"))]
    scenario = derive_scenario(world, "dependency", sequence, "multi_step")
    assert [s.expected_allowed for s in scenario.steps] == [True, True, False, True, True]
    trace = simulate_scenario(world, scenario)
    assert trace.final_state.subscriptions["sub_000"].plan_id == "starter"
    assert evaluate_trace(trace).status == "pass"


def test_role_then_billing_dependency(world):
    scenario = derive_scenario(world, "roles", [
        ("user_003", operation("cancel_renewal")),
        ("user_000", operation("change_member_role", resource="membership_000_user_003", role="billing_admin")),
        ("user_003", operation("cancel_renewal"))], "multi_entity")
    assert [s.expected_allowed for s in scenario.steps] == [False, True, True]
    assert evaluate_trace(simulate_scenario(world, scenario)).status == "pass"


def test_scenario_label_tampering_and_world_mismatch(world):
    scenario = derive_scenario(world, "r", [("user_000", operation("cancel_renewal"))], "single")
    scenario.steps[0].expected_allowed = False
    with pytest.raises(ValueError, match="derived"):
        simulate_scenario(world, scenario)
    with pytest.raises(ValueError, match="mismatch"):
        validate_scenario(generate_world(99), scenario)


def test_manifest_freezing_reproduction_and_private_loader(tmp_path):
    manifest = create_manifest("validation", [1000000, 1000001])
    assert manifest == create_manifest("validation", [1000000, 1000001])
    target = tmp_path / "validation.json"
    manifest.save(target)
    assert load_manifest(target) == manifest
    assert_fair_comparison(manifest, load_manifest(target))
    private = create_manifest("test", [991009, 991010])
    private.save(tmp_path / "private" / "final-test.json")
    assert load_manifest(tmp_path / "private" / "final-test.json").visibility == "private"
    validate_splits(manifest, private)


def test_compositional_holdout_is_business_state_not_identifiers():
    cfg_train, cfg_test = GeneratorConfig(profile="train_single"), GeneratorConfig(profile="test_composed")
    for seed in range(100):
        train, test = generate_world(seed, cfg_train), generate_world(seed, cfg_test)
        assert not composition_signature(train)
        assert composition_signature(test)
        assert set(train.users) == set(test.users)
        assert set(train.organizations) == set(test.organizations)
        assert business_fingerprint(train) != business_fingerprint(test)
    train = create_manifest("train", [1, 2, 3])
    validation = create_manifest("validation", [1000000, 1000001])
    test = create_manifest("test", [2000000, 2000001])
    validate_splits(train, validation, test)
    # Each standalone permission/subscription/invoice rule is represented in train;
    # only the cross-organization conjunction is reserved for composed test.
    assert not any(composition_signature(w) for w in islice(sample_train(), 10))


def test_duplicate_instances_and_cross_split_overlap_detected():
    with pytest.raises(ValueError, match="duplicate"):
        create_manifest("validation", [42, 42])
    a = create_manifest("validation", [42])
    b = create_manifest("validation", [42])
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(a, b)
    with pytest.raises(ValueError, match="positive|empty"):
        create_manifest("validation", [])


def test_manifest_tampering_and_fairness():
    original = create_manifest("validation", [42])
    modified = original.model_copy(deep=True)
    modified.operation_budget = 21
    with pytest.raises(ValueError, match="hash"):
        validate_manifest(modified)
    other = create_manifest("validation", [43])
    with pytest.raises(ValueError, match="same frozen"):
        assert_fair_comparison(original, other)
    modified = original.model_copy(deep=True)
    modified.entries[0].scenarios[0].steps[0].expected_allowed = not modified.entries[0].scenarios[0].steps[0].expected_allowed
    modified.manifest_hash = modified.computed_hash()
    with pytest.raises(ValueError, match="derived"):
        validate_manifest(modified)


def test_no_published_blind_test_claims():
    with pytest.raises(ValueError, match="private"):
        create_manifest("test", [4], visibility="public")
    assert create_manifest("test", [4], visibility="demo").visibility == "demo"


def test_generator_from_sparse_custom_world(world):
    world.invoices = {}
    scenarios = generate_scenarios(world)
    assert scenarios
    for scenario in scenarios:
        validate_scenario(world, scenario)
