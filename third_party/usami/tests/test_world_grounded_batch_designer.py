import copy
import json

import pytest

from usami.service.client import ModelSettings
from usami_benchmark.abstract_challenges import (
    BATCH_REVISION_PROTOCOL, PROTOCOL, AbstractChallenge, challenge_fingerprints, validate_challenge,
)
from usami_benchmark.designer_batch import (
    BatchDesignerConfig, TOOL_NAME, replay_run, run_batch,
)
from usami_benchmark.grounder import GroundingBudget, ground_challenge
from usami_benchmark.registry import default_spec
from usami_benchmark.world_index import build_world_index, public_domain_catalogue
from usami_benchmark.worlds import WorldPool, freeze
from usami_harness.models import digest


def _mixed_challenge(challenge_id="mixed-1"):
    return {
        "challenge_id": challenge_id,
        "hypothesis": "Can the service complete an eligible case while refusing an ineligible one?",
        "domain": "aftersales",
        "customer_request": "Please handle both issues using each purchase's actual details.",
        "entity_variables": [
            {"alias": "buyer_a", "type": "profile"}, {"alias": "buyer_b", "type": "profile"},
            {"alias": "line_a", "type": "line"}, {"alias": "line_b", "type": "line"},
        ],
        "entity_constraints": [
            {"alias": "line_a", "predicate": "eligible_for_case", "value": True},
            {"alias": "line_a", "predicate": "has_valid_evidence", "value": True},
            {"alias": "line_b", "predicate": "eligible_for_case", "value": False},
        ],
        "relationship_constraints": [
            {"type": "profile_for_line", "left": "buyer_a", "right": "line_a"},
            {"type": "profile_for_line", "left": "buyer_b", "right": "line_b"},
            {"type": "same_identity", "left": "buyer_a", "right": "buyer_b"},
            {"type": "distinct", "left": "line_a", "right": "line_b"},
        ],
        "objectives": [
            {"kind": "created_object_completion", "target": "line_a", "collection": "cases",
             "field": "status", "accepted_values": ["evidence_verified"]},
            {"kind": "correct_refusal", "target": "line_b", "operation": "open_case"},
        ],
        "required_customer_information": [
            {"profile_alias": "buyer_a", "fact_keys": ["line_id", "issue", "evidence_id"]},
            {"profile_alias": "buyer_b", "fact_keys": ["line_id", "issue"]},
        ],
    }


def _response(challenges, *, usage=True):
    row = {"protocol": PROTOCOL, "challenges": challenges}
    result = {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
        "tool_calls": [{"id": "batch-call-1", "type": "function", "function": {
            "name": TOOL_NAME, "arguments": json.dumps(row, sort_keys=True)}}]}}]}
    if usage:
        result["usage"] = {"prompt_tokens": 321, "completion_tokens": 97, "total_tokens": 418}
    return result


class ScriptedClient:
    def __init__(self, responses):
        self.settings = ModelSettings(model="offline-scripted", base_url="https://fixture.invalid/v1", options={})
        self.responses = list(responses)
        self.requests = []

    def complete(self, request):
        self.requests.append(copy.deepcopy(request))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return copy.deepcopy(response)


@pytest.fixture(scope="module")
def world_and_index():
    frozen = freeze(default_spec("aftersales", 1, "train"))
    return frozen, build_world_index(WorldPool([frozen]))


def test_registered_grammar_rejects_unknown_fields_code_and_ids(world_and_index):
    frozen, index = world_and_index
    valid = AbstractChallenge.model_validate(_mixed_challenge())
    assert validate_challenge(valid).challenge_hash == valid.challenge_hash
    assert challenge_fingerprints(valid)["structural"]

    bad = _mixed_challenge()
    bad["entity_constraints"][0] = {"alias": "line_a", "field": "__import__", "op": "eq", "value": "x"}
    with pytest.raises(ValueError, match="unknown field"):
        validate_challenge(bad)
    bad = _mixed_challenge()
    bad["entity_constraints"][0] = {"alias": "line_a", "predicate": "eval", "value": True}
    with pytest.raises(Exception):
        AbstractChallenge.model_validate(bad)
    bad = _mixed_challenge()
    bad["hypothesis"] += " profile_u1_order0_line0"
    with pytest.raises(ValueError, match="concrete generated-world identifiers"):
        validate_challenge(bad)

    catalogue = public_domain_catalogue(index, "aftersales", partitions=("train",))
    assert frozen.world["profiles"]
    assert not any(profile_id in json.dumps(catalogue) for profile_id in frozen.world["profiles"])


def test_grounder_binds_same_identity_mixed_goals_and_native_certificate_replays(world_and_index):
    frozen, index = world_and_index
    before = frozen.model_dump(mode="json")
    result = ground_challenge(_mixed_challenge(), index, GroundingBudget(max_matches=1, partition="train"))
    assert result.status == "GROUNDED"
    match = result.matches[0]
    assert len(match.profile_ids) == 2
    assert match.identity == frozen.world["profiles"][match.profile_ids[0]]["identity"]
    assert match.binding_hash == digest({"registration_hash": frozen.registration_hash, "binding": match.binding})
    assert frozen.model_dump(mode="json") == before

    from usami_benchmark.tasks import ConstructionBudget, compile_task, verify_task
    task = compile_task(frozen, match.selection, ConstructionBudget(max_nodes=4096, max_depth=16, max_paths=1))
    assert verify_task(task)
    witness = task.dialogue_witnesses[0]
    assert witness.certificate.evaluation["created_completion_subgoals"] == [True]
    assert witness.certificate.evaluation["refusal_subgoals"] == [True]

    again = ground_challenge(_mixed_challenge(), index, GroundingBudget(max_matches=1, partition="train"))
    assert [m.binding_hash for m in again.matches] == [m.binding_hash for m in result.matches]


def test_no_match_and_tampered_world_index_are_explicit(world_and_index):
    frozen, index = world_and_index
    out_of_scope = ground_challenge(_mixed_challenge(), index,
        GroundingBudget(max_matches=1, partition="validation"))
    assert out_of_scope.status == "OUT_OF_SCOPE"
    with pytest.raises(ValueError, match="train/validation"):
        public_domain_catalogue(index, "aftersales", partitions=("heldout",))

    no_match = _mixed_challenge("absent"); no_match["objectives"] = [no_match["objectives"][0]]
    no_match["entity_variables"] = [v for v in no_match["entity_variables"] if v["alias"] in {"buyer_a", "line_a"}]
    no_match["entity_variables"].append({"alias": "order_a", "type": "order"})
    no_match["relationship_constraints"] = [
        {"type": "profile_for_line", "left": "buyer_a", "right": "line_a"},
        {"type": "references", "left": "line_a", "right": "order_a"},
    ]
    no_match["required_customer_information"] = []
    no_match["entity_constraints"] = [
        {"alias": "line_a", "predicate": "within_case_window", "value": True},
        {"alias": "order_a", "field": "status", "op": "eq", "value": "processing"},
    ]
    row = ground_challenge(no_match, index, GroundingBudget(max_matches=1, partition="train"))
    assert row.status == "NO_MATCH"
    assert row.matches == []

    raw = index.model_dump(mode="json")
    raw["worlds"][0]["frozen"]["world"]["clock"] = "tampered"
    with pytest.raises(ValueError):
        type(index).model_validate(raw).checked()


def test_single_shot_calls_once_replays_grounding_and_detects_request_tampering(world_and_index):
    _, index = world_and_index
    client = ScriptedClient([_response([_mixed_challenge()])])
    client.settings = client.settings.model_copy(update={"options": {"max_tokens": 4096}})
    run = run_batch(index, client, BatchDesignerConfig(num_challenges=1,
        grounding=GroundingBudget(max_matches=1, partition="train")))
    assert len(client.requests) == 1
    assert "max_tokens" not in client.requests[0]
    assert run.request_revisions == [0] and run.response_revisions == [0]
    assert run.metrics["total_llm_calls"] == 1
    assert run.cost["total_tokens"] == 418 and run.cost["usage_complete"] is True
    assert run.candidates[0].final_status == "CERTIFIED"
    assert run.candidates[0].model_request_hash == digest(client.requests[0])
    replay = replay_run(run, index)
    assert replay["verified"] and replay["native_certificates_replayed"] == 1
    assert replay["service_executions"] == 0

    tampered = run.model_copy(deep=True)
    tampered.requests[0]["messages"][0]["content"] += " changed"
    tampered.run_hash = tampered.computed_hash()
    with pytest.raises(ValueError, match="recorded model request differs"):
        replay_run(tampered, index)


def test_single_shot_does_not_retry_schema_failure_and_records_missing_usage(world_and_index):
    _, index = world_and_index
    invalid = _mixed_challenge(); invalid["entity_constraints"][0]["predicate"] = "unknown_predicate"
    client = ScriptedClient([_response([invalid], usage=False)])
    run = run_batch(index, client, BatchDesignerConfig(num_challenges=1))
    assert len(client.requests) == 1
    assert run.metrics["schema_errors"] == 1
    assert run.candidates[0].abstract_status == "SCHEMA_INVALID"
    assert run.cost["usage_complete"] is False
    assert replay_run(run, index)["verified"]


def test_one_revision_is_bounded_and_replayed(world_and_index):
    _, index = world_and_index
    invalid = _mixed_challenge()
    invalid["entity_constraints"][0]["predicate"] = "unknown_predicate"
    client = ScriptedClient([_response([invalid]), _response([_mixed_challenge()])])
    run = run_batch(index, client, BatchDesignerConfig(protocol=BATCH_REVISION_PROTOCOL, num_challenges=1,
        grounding=GroundingBudget(max_matches=1, partition="train")))
    assert len(client.requests) == 2
    assert run.candidate_history[0].revision == 0 and run.candidate_history[1].revision == 1
    assert run.candidates[0].final_status == "CERTIFIED"
    assert replay_run(run, index)["verified"]


def test_unconfigured_client_is_not_counted_as_a_real_llm_call(world_and_index):
    _, index = world_and_index

    class Unavailable:
        settings = ModelSettings(model="unavailable", base_url="https://fixture.invalid/v1", options={})
        available = False
        unavailable_reason = "test credential unavailable"

    run = run_batch(index, Unavailable(), BatchDesignerConfig(num_challenges=1))
    assert run.status == "BLOCKED_EXTERNAL_MODEL"
    assert run.metrics["request_attempts"] == 1
    assert run.metrics["total_llm_calls"] == 0
    assert run.metrics["preflight_errors"] == 1 and run.metrics["transport_errors"] == 0
    assert run.cost["designer_api_calls"] == 0
    replay = replay_run(run, index)
    assert replay["verified"] and replay["model_calls_replayed_offline"] == 0
    assert replay["request_records_replayed_offline"] == 1


def test_offline_abc_comparison_uses_same_frozen_world_and_one_call_each(world_and_index, tmp_path):
    from experiments.vast_coevo.examples.compare_world_grounded_designer import run_comparison

    frozen, index = world_and_index
    grounded = ground_challenge(_mixed_challenge(), index, GroundingBudget(max_matches=1, partition="train"))
    assert grounded.status == "GROUNDED"
    proposal = grounded.matches[0].selection.composition.model_dump(mode="json")
    proposal.pop("protocol", None); proposal.pop("frames", None)
    v4_response = {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
        "tool_calls": [{"id": "v4-submit", "type": "function", "function": {
            "name": "submit_frozen_world_task", "arguments": json.dumps(proposal)}}]}}],
        "usage": {"prompt_tokens": 250, "completion_tokens": 70, "total_tokens": 320}}
    batch_response = _response([_mixed_challenge()])
    settings = {"model": "offline-paired", "base_url": "https://fixture.invalid/v1", "options": {}}
    v4_file = tmp_path / "v4.json"; v4_file.write_text(json.dumps({"model": settings, "responses": [v4_response]}))
    batch_file = tmp_path / "batch.json"; batch_file.write_text(json.dumps({"model": settings, "responses": [batch_response]}))
    report = run_comparison(index, tmp_path / "comparison", batch_count=1,
        v4_response=str(v4_file), batch_response=str(batch_file))
    assert report["A_v4"]["status"] == "CERTIFIED"
    assert report["B_world_grounded_batch"]["status"] == "COMPLETED"
    assert report["budget_control"]["api_call_matched"] is True
    assert report["budget_control"]["token_budget_matched"] is False
    assert report["C_rule_combination_sampling"]["cost"]["designer_api_calls"] == 0


def _single_challenge():
    from usami_benchmark.designer_batch import _rule_challenges
    return _rule_challenges(1)[0]


def test_catalogue_and_wire_schema_only_advertise_executable_grammar(world_and_index):
    import jsonschema
    from typing import get_args
    from usami_benchmark.abstract_challenges import Relation, ChallengeObjective, submission_schema
    from usami_benchmark.designer_batch import _rule_challenges

    _, index = world_and_index
    catalogue = public_domain_catalogue(index, "aftersales")
    assert set(catalogue["relationships"]) == set(get_args(Relation))
    assert set(catalogue["objective_types"]) == set(get_args(ChallengeObjective.model_fields["kind"].annotation))
    assert "profile_for_line" in catalogue["relationships"]
    assert not {"authenticated_customer", "authenticated_identity"} & set(catalogue["relationships"])
    assert not {"mixed", "multi_object"} & set(catalogue["objective_types"])
    assert catalogue["field_contracts"]["evidence"]["issue"]["value_schema"]["enum"] == ["damaged", "wrong_item", "missing_parts"]
    for count in range(1, 6):
        schema = submission_schema(count)
        jsonschema.Draft202012Validator.check_schema(schema)
        assert "$ref" not in json.dumps(schema) and "$defs" not in schema
        challenges = _rule_challenges(count)
        jsonschema.validate({"protocol": PROTOCOL, "challenges": challenges}, schema)
        for challenge in challenges:
            validate_challenge(challenge)
    schema = submission_schema(1)
    for key in ("challenge_id", "hypothesis", "domain", "customer_request", "entity_variables"):
        bad = _single_challenge(); bad.pop(key)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"protocol": PROTOCOL, "challenges": [bad]}, schema)
    bad = _single_challenge(); bad["objectives"][0]["kind"] = "multi_object"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"protocol": PROTOCOL, "challenges": [bad]}, schema)


@pytest.mark.parametrize("kind,field,op,value", [
    ("evidence", "issue", "eq", "damage"),
    ("external_event", "kind", "eq", "return_pickup"),
    ("product", "stock", "eq", True),
    ("order", "payment_channel", "eq", 123),
    ("product", "returnable", "gt", False),
    ("order", "status", "lt", "delivered"),
    ("order", "delivered_at", "gte", "2026-01-01T00:00:00"),
    ("external_event", "at", "in", []),
])
def test_invalid_field_values_rejected_before_world_search(kind, field, op, value):
    from usami_benchmark.abstract_challenges import validate_field_value
    with pytest.raises(ValueError):
        validate_field_value(kind, field, value, op)


def test_time_comparisons_use_instants_and_nullable_delivery_dates(world_and_index):
    from usami_benchmark.grounder import _aware_time, _compare
    assert _compare(_aware_time("2026-01-01T00:00:00Z"), "eq", _aware_time("2026-01-01T08:00:00+08:00"))
    assert not _compare(None, "gte", _aware_time("2026-01-01T00:00:00Z"))
    _, index = world_and_index
    challenge = _single_challenge()
    challenge["entity_variables"].append({"alias": "purchase", "type": "order"})
    challenge["relationship_constraints"].append({"type": "references", "left": "line_a", "right": "purchase"})
    challenge["entity_constraints"].append({"alias": "purchase", "field": "delivered_at", "op": "gte", "value": "2000-01-01T00:00:00Z"})
    result = ground_challenge(challenge, index, GroundingBudget(max_matches=1))
    assert result.status == "GROUNDED"


def test_all_bound_entities_share_customer_scope_and_facts_use_real_collections(world_and_index):
    _, index = world_and_index
    challenge = _single_challenge()
    challenge["entity_variables"].extend([{"alias": "who", "type": "customer"},
        {"alias": "photo", "type": "evidence"}, {"alias": "notice", "type": "external_event"}])
    challenge["relationship_constraints"].extend([
        {"type": "references", "left": "photo", "right": "line_a"},
        {"type": "event_for_line", "left": "notice", "right": "line_a"}])
    result = ground_challenge(challenge, index, GroundingBudget(max_matches=2))
    assert result.status == "GROUNDED"
    for match in result.matches:
        assert match.binding["who"] == match.identity
        assert not match.selection.composition.external_schedule  # Inspection does not schedule events.
        for source in match.fact_sources:
            collection, key = source["source"].split("/")
            assert key in index.worlds[0].frozen.world[collection.removeprefix("world.")]


def test_grounding_budget_never_exceeds_declared_limit(world_and_index):
    _, index = world_and_index
    result = ground_challenge(_mixed_challenge(), index, GroundingBudget(max_bindings=1))
    assert result.status == "BUDGET_EXHAUSTED"
    assert result.bindings_explored <= 1 and result.search_truncated


@pytest.mark.parametrize("malformed", [["not-a-string"], {"bad": "id"}, None, 17])
def test_malformed_candidate_ids_do_not_crash_or_collapse_records(world_and_index, malformed):
    _, index = world_and_index
    bad = _single_challenge(); bad["challenge_id"] = malformed
    run = run_batch(index, ScriptedClient([_response([bad, copy.deepcopy(bad)])]), BatchDesignerConfig(num_challenges=2))
    assert len(run.candidates) == 2
    assert len({c.challenge_id for c in run.candidates}) == 2
    assert run.metrics["schema_errors"] == 2
    assert run.cost["reference_compile_attempts"] == 0
    assert replay_run(run, index)["verified"]


@pytest.mark.parametrize("mode", ["duplicate", "overfull", "underfull"])
def test_bad_batches_are_not_silently_truncated_or_compiled(world_and_index, mode):
    _, index = world_and_index
    first, second = _single_challenge(), _single_challenge()
    second["challenge_id"] = "second"
    rows, count = {"duplicate": ([first, first], 2), "overfull": ([first, second], 1), "underfull": ([first], 2)}[mode]
    run = run_batch(index, ScriptedClient([_response(rows)]), BatchDesignerConfig(num_challenges=count))
    assert run.metrics["raw_generated_challenges"] == len(rows)
    assert len(run.candidate_history) == max(count, len(rows))
    assert run.metrics["certified_challenges"] == 0
    assert run.metrics["certification_status"] == "NO_CERTIFIED_CHALLENGES"
    assert run.cost["reference_compile_attempts"] == 0
    assert replay_run(run, index)["verified"]


def test_one_revision_recovers_entire_malformed_envelope_with_multiple_candidates(world_and_index):
    _, index = world_and_index
    a, b = _single_challenge(), _single_challenge()
    a.pop("domain"); b.pop("domain"); b["challenge_id"] = "second"
    client = ScriptedClient([{"choices": []}, _response([a, b])])
    run = run_batch(index, client, BatchDesignerConfig(protocol=BATCH_REVISION_PROTOCOL, num_challenges=2))
    assert len(client.requests) == 2 and len(run.candidates) == 2
    assert run.metrics["revision_protocol_errors"] == 0
    assert replay_run(run, index)["verified"]


@pytest.mark.parametrize("choices", [None, 1, {}, "bad", [None]])
def test_malformed_response_envelopes_are_recorded_without_crashing(world_and_index, choices):
    _, index = world_and_index
    run = run_batch(index, ScriptedClient([{"choices": choices}]), BatchDesignerConfig(num_challenges=1))
    assert run.metrics["response_errors"] == 1 and run.metrics["total_llm_calls"] == 1
    assert replay_run(run, index)["verified"]


def test_malformed_provider_usage_is_retained_but_not_treated_as_measured_cost(world_and_index):
    _, index = world_and_index
    response = _response([{}]); response["usage"] = {"prompt_tokens": "321", "completion_tokens": True, "total_tokens": -3}
    run = run_batch(index, ScriptedClient([response]), BatchDesignerConfig(num_challenges=1))
    assert run.responses[0]["usage"] == response["usage"]
    assert not run.cost["usage_complete"] and run.cost["total_tokens"] == 0
    assert replay_run(run, index)["verified"]


def test_replay_checks_diagnostic_metrics_and_exported_files(world_and_index, tmp_path):
    from usami_benchmark.designer_batch import save_run, replay_directory
    _, index = world_and_index
    run = run_batch(index, ScriptedClient([_response([{}])]), BatchDesignerConfig(num_challenges=1))
    for key, value in (("actual_service_executions", 1), ("certified_challenges", 12), ("schema_errors", 0)):
        bad = run.model_copy(deep=True); bad.metrics[key] = value; bad.run_hash = bad.computed_hash()
        with pytest.raises(ValueError, match="metrics"):
            replay_run(bad, index)
    root = save_run(run, index, tmp_path / "archive")
    assert replay_directory(root)["verified"]
    (root / "model-request.json").write_text("[]")
    with pytest.raises(ValueError, match="artifact hash"):
        replay_directory(root)


def test_old_archives_are_explicitly_bound_to_original_source_version(world_and_index):
    _, index = world_and_index
    run = run_batch(index, ScriptedClient([_response([{}])]), BatchDesignerConfig(num_challenges=1))
    old = run.model_dump(mode="json"); old.pop("contract_version")
    with pytest.raises(ValueError, match="original source commit cbb5da4"):
        replay_run(old, index)


def test_fingerprints_preserve_objective_entity_links_and_ignore_alias_spelling():
    original = _mixed_challenge()
    renamed = json.loads(json.dumps(original).replace("buyer_a", "renamed_profile").replace("line_a", "renamed_line"))
    assert challenge_fingerprints(original) == challenge_fingerprints(renamed)
    changed = copy.deepcopy(original)
    changed["objectives"][0]["target"] = "line_b"
    changed["objectives"][1]["target"] = "line_a"
    assert challenge_fingerprints(original)["semantic"] != challenge_fingerprints(changed)["semantic"]
    reversed_relations = copy.deepcopy(original)
    for relation in reversed_relations["relationship_constraints"]:
        relation["left"], relation["right"] = relation["right"], relation["left"]
    assert challenge_fingerprints(original) == challenge_fingerprints(reversed_relations)


def test_cli_invalid_submission_fails_and_never_reuses_output_directory(world_and_index, tmp_path, monkeypatch):
    import usami_benchmark.designer_batch as module
    _, index = world_and_index
    index_file = tmp_path / "index.json"; index_file.write_text(index.model_dump_json())
    response_file = tmp_path / "response.json"; response_file.write_text(json.dumps(_response([{}])))
    root = tmp_path / "attempt"
    args = ["run", "--index", str(index_file), "--num-challenges", "1", "--scripted-response", str(response_file), "--output-dir", str(root)]
    assert module.main(args) == 2
    assert module.replay_directory(root)["verified"]
    monkeypatch.setattr(module, "run_batch", lambda *a, **k: pytest.fail("must not dispatch into an existing archive"))
    with pytest.raises(SystemExit) as exc:
        module.main(args)
    assert exc.value.code == 2


def test_oversized_envelope_preserves_actual_raw_candidate_count(world_and_index):
    _, index = world_and_index
    run = run_batch(index, ScriptedClient([_response([{}] * 6)]), BatchDesignerConfig(num_challenges=1))
    assert run.metrics["raw_generated_challenges"] == 6
    assert run.metrics["certified_challenges"] == 0 and run.metrics["response_errors"] == 1
    assert replay_run(run, index)["verified"]


@pytest.mark.parametrize("layer", ["grounder", "compiler", "replay", "client"])
def test_code_failures_are_not_reported_as_bad_proposals_or_retried(world_and_index, monkeypatch, layer):
    import usami_benchmark.designer_batch as module
    _, index = world_and_index

    def broken(*args, **kwargs):
        raise RuntimeError("deliberate offline code-failure fixture")

    response = _response([_single_challenge()])
    if layer != "client":
        monkeypatch.setattr(module, {"grounder": "ground_challenge", "compiler": "compile_task", "replay": "verify_task"}[layer], broken)
    client = ScriptedClient([RuntimeError("do not leak exception text") if layer == "client" else response])
    run = run_batch(index, client, BatchDesignerConfig(protocol=BATCH_REVISION_PROTOCOL, num_challenges=1,
        grounding=GroundingBudget(max_matches=1)))
    assert len(client.requests) == 1
    if layer == "client":
        assert run.status == "INTERNAL_ERROR"
        assert run.transport_errors[0]["message"] == "completion client raised RuntimeError"
    else:
        assert run.candidates[0].final_status == ("REPLAY_REJECTED" if layer == "replay" else "INTERNAL_ERROR")
    assert run.metrics["certified_challenges"] == 0
    assert replay_run(run, index)["verified"]


def test_duplicate_constraints_cannot_inflate_coverage_signatures():
    row = _mixed_challenge()
    row["entity_constraints"].append(copy.deepcopy(row["entity_constraints"][0]))
    with pytest.raises(ValueError, match="must not repeat"):
        validate_challenge(row)
    row = _mixed_challenge()
    row["relationship_constraints"].append({"type": "same_identity", "left": "buyer_b", "right": "buyer_a"})
    with pytest.raises(ValueError, match="symmetric reversals"):
        validate_challenge(row)
