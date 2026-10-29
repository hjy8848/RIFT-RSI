"""Production evidence verification and run-directory initialization."""
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from threading import Lock

from rift.banks.records import EvidenceLevel
from rift.benchmark.usami import public_episode
from rift.banks.records import TaskEvidenceRecord
from rift.schemas import DataPurpose, Outcome, Split


class JsonlMetricSink:
    """Append one timestamped, run-local measurement per JSONL row."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def __call__(self, name, value, *, round_index=None, **fields):
        if not isinstance(name, str) or not name:
            raise ValueError('metric name required')
        row = {'timestamp': datetime.now(timezone.utc).isoformat(),
            'round_index': round_index, 'metric': name, 'value': value, **fields}
        encoded = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
        with self._lock, self.path.open('a', encoding='utf-8') as stream:
            stream.write(encoded + '\n')
            stream.flush()


class NativeProbeTrialVerifier:
    """Verify Probe output from its persisted native Episode and token capture."""
    def __init__(self, task_registry, capture_store):
        if not callable(getattr(task_registry, "resolve_record", None)):
            raise TypeError("Host certified task registry required")
        if not callable(getattr(capture_store, "resolve_episode", None)):
            raise TypeError("native Episode and capture resolver required")
        self.task_registry, self.capture_store = task_registry, capture_store

    def verify_probe_trial(self, trial, probe, checkpoint):
        try:
            record = self.task_registry.resolve_record(trial.task_ref)
            episode = self.capture_store.resolve_episode(trial.trial_id)
            from rift.verl.sampling import rollout_seed
            expected_rollout_id = sha256(
                f"rift-probe:{trial.task_ref}:{checkpoint.weights_digest}:{trial.seed}:{probe.protocol_ref}".encode()
            ).hexdigest()[:32]
            sampling_scope = sha256(
                f"rift-probe-sampling-v1:{trial.task_ref}:{trial.seed}:{probe.protocol_ref}".encode()
            ).hexdigest()
            calls = [call for call in episode.model_calls if call.role == "service"]
            public = public_episode(episode)
            if (trial.evidence_level is not EvidenceLevel.REAL_SERVICE
                    or trial.public_summary is not None or trial.task_ref != probe.task_ref
                    or trial.checkpoint != checkpoint or trial.protocol_digest != probe.protocol_digest
                    or record.identity != probe.identity or record.identity.purpose is not DataPurpose.BANK_CANDIDATE
                    or episode.record_hash != trial.trial_id or episode.task.record_hash != trial.task_ref
                    or public.outcome is not trial.outcome
                    or not self.capture_store.verify_trajectory(episode, checkpoint)
                    or not calls):
                return False
            if trial.evidence_ref is not None and trial.evidence_ref != self.capture_store.evidence_ref(episode):
                return False
            if trial.outcome is not Outcome.UNKNOWN and (not trial.replay_verified or not trial.evidence_ref):
                return False
            stored = self.capture_store._load(episode, checkpoint)
            if stored.get("rollout_id") != expected_rollout_id:
                return False
            for turn, call in enumerate(calls):
                request = call.request
                if (request.get("rift_mode") != "probe" or request.get("task_ref") != trial.task_ref
                        or request.get("sampling_seed") != trial.seed
                        or request.get("protocol_ref") != probe.protocol_ref
                        or request.get("protocol_digest") != probe.protocol_digest
                        or request.get("rollout_id") != expected_rollout_id):
                    return False
                sampling = request.get("sampling_params", {})
                expected_seed = rollout_seed(trial.seed, sampling_scope, turn)
                if sampling.get("seed") != expected_seed:
                    return False
            return True
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return False


class ProductionBankUpdater:
    """Persist verified post-update Probe observations into the existing banks."""
    def __init__(self, task_evidence_store, task_registry, trial_verifier):
        self.task_evidence_store, self.task_registry = task_evidence_store, task_registry
        self.trial_verifier = trial_verifier

    def __call__(self, state, banks):
        if (state.stage.value != "DESIGNER_TRAINING_COMPLETED"
                or state.service_update is None or state.recovery_after is None
                or state.retention_after is None):
            raise ValueError("complete real Service update and post-update Probe evidence required")
        checkpoint = state.service_update.policy_after
        snapshots = (state.recovery_snapshot, state.retention_snapshot)
        by_purpose = {snapshot.purpose: snapshot for snapshot in snapshots}
        for result_set in (state.recovery_after, state.retention_after):
            snapshot = by_purpose[result_set.purpose]
            probes = {probe.task_ref: probe for probe in snapshot.probes}
            for result in result_set.results:
                probe = probes.get(result.task_ref)
                if (probe is None or probe.identity.split is Split.TEST
                        or result.identity.split is Split.TEST
                        or result.checkpoint != checkpoint
                        or result.identity != probe.identity
                        or result.phase != "after"):
                    raise ValueError("post-update Probe result differs from the frozen snapshot")
                if any(self.trial_verifier.verify_probe_trial(trial, probe, checkpoint) is not True
                       for trial in result.trials):
                    raise ValueError("post-update Probe trial evidence failed verification")
                record = TaskEvidenceRecord(result.task_ref, result.identity, checkpoint,
                    result.protocol_digest, result.trials,
                    datetime.now(timezone.utc).isoformat())
                try:
                    stored = self.task_evidence_store.load(result.task_ref, checkpoint)
                except FileNotFoundError:
                    self.task_evidence_store.save(record)
                    stored = record
                if (stored.task_ref != record.task_ref or stored.identity != record.identity
                        or stored.checkpoint != record.checkpoint
                        or stored.protocol_digest != record.protocol_digest
                        or stored.trials != record.trials):
                    raise ValueError("post-update evidence conflicts with the immutable task record")
                event_id = f"round-{state.round_index:06d}:{result_set.purpose.value}:{result.task_ref}:{checkpoint.revision}"
                banks.observe(stored, event_id=event_id)


def prepare_run_directory(root, *, config_snapshot, manifest=None):
    """Create the durable RIFT run layout and immutable config/environment records."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    directories = (
        "bootstrap", "task_registry", "worlds", "episodes", "designer_captures",
        "probe_evidence", "checkpoints/service", "checkpoints/designer", "round_states",
    )
    for relative in directories:
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / 'metrics.jsonl').touch(exist_ok=True)
    config_path = root / "config_snapshot.yaml"
    config_text = str(config_snapshot)
    if config_path.exists():
        if config_path.read_text(encoding="utf-8") != config_text:
            raise ValueError("run directory is already bound to a different config snapshot")
    else:
        config_path.write_text(config_text, encoding="utf-8")
    packages = {}
    for name in ("numpy", "torch", "torch-npu", "transformers", "verl", "vllm", "vllm-ascend", "triton-ascend"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    environment = {"python": sys.version.split()[0], "platform": platform.platform(),
                   "machine": platform.machine(), "packages": packages,
                   "deployment": dict(manifest or {})}
    manifest_path = root / "environment_manifest.json"
    encoded = json.dumps(environment, sort_keys=True, ensure_ascii=False, indent=2)
    if manifest_path.exists():
        if manifest_path.read_text(encoding="utf-8") != encoded:
            raise ValueError("run directory environment manifest changed across resume")
    else:
        manifest_path.write_text(encoded, encoding="utf-8")
    return root
