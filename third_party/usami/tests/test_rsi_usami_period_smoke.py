"""End-to-end RSI period over the real native Usami commerce runtime."""
from rsi_core.archive import Archive
from rsi_core.programs import DesignerSnapshot, ProgramArtifact, ProgramStore

from experiments.rsi.examples.smoke_v02_usami_period import IMAGE, run


def test_v02_period_uses_native_usami_and_replays_every_episode(tmp_path):
    output = tmp_path / "native-period"
    report = run(output, image=IMAGE)
    period = report["period"]

    assert period["status"] == "READY_FOR_TRAINING"
    assert period["training_status"] == "NOT_RUN"
    assert report["online_model_calls"] == 0
    assert report["verl_training_runs"] == 0
    assert report["parameter_updates"] == 0
    assert report["native_domain"] == "usami-commerce-v2"

    assert period["service_rounds"][0]["group_status"] == "USABLE"
    assert period["output_service_ref"] == period["input_service_ref"]
    assert report["native_episode_count"] == 36
    assert report["native_replays_verified"] == report["native_episode_count"]
    assert all(row["replay_verified"] is True for row in report["native_episodes"])
    assert any(row["strict_task_success"] is True for row in report["native_episodes"])
    assert any(row["strict_task_success"] is False and row["safety"] == "PASS"
               for row in report["native_episodes"])

    archive = Archive(output / "archive")
    service_group = archive.load(period["service_rounds"][0]["group_ref"])
    assert service_group["status"] == "USABLE"
    assert period["service_rounds"][0]["candidate_selected_ref"] == period["input_service_ref"]
    service_roots = [archive.load(ref) for ref in service_group["root_refs"]]
    assert service_roots and all(row["status"] == "SUBMITTED" for row in service_roots)
    service_selection = next(iter(service_group["selections"].values()))
    assert service_selection["decision"] == "KEEP_PARENT"
    assert service_selection["reasons"] == ["NO_PAIRED_GAIN"]

    group = archive.load(period["self_edit_group_ref"])
    assert group["status"] == "USABLE"
    assert group["selected_ref"] == period["selected_m_ref"]
    assert group["selections"][period["selected_m_ref"]]["decision"] == "ACCEPT"
    store = ProgramStore(output / "programs")
    initial_designer = store.load(period["input_designer_ref"])
    assert isinstance(initial_designer, DesignerSnapshot)
    selected_m = store.load(period["selected_m_ref"])
    assert isinstance(selected_m, ProgramArtifact) and selected_m.kind == "M"
    assert selected_m.ref != initial_designer.research_program.ref
    designer = store.load(period["output_designer_ref"])
    assert isinstance(designer, DesignerSnapshot)
    assert designer.parent_ref == period["input_designer_ref"]
    assert designer.research_program.ref == period["selected_m_ref"]
