"""VERL Agent Loop Manager adapters for real Service and Probe episodes."""
import asyncio
from dataclasses import asdict
import inspect
import json
from hashlib import sha256
import time

from rift.banks.records import EvidenceLevel, ObservedTrial
from rift.schemas import Checkpoint, Outcome
from rift.training.service_data import prepare_package_service_update


def _sync_await(value):
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    raise RuntimeError("synchronous RIFT runtime cannot wait inside an active asyncio loop")


def _object_array(values):
    import numpy as np
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


def checkpoint_json(checkpoint):
    if not isinstance(checkpoint, Checkpoint):
        raise TypeError("Checkpoint required")
    return json.dumps(asdict(checkpoint), sort_keys=True, separators=(",", ":"))


class VerlAgentLoopBatch:
    """Build one pinned VERL DataProto row per RIFT episode request.

    AgentLoopManager's standard worker already executes rows concurrently. RIFT
    expands Service GRPO groups into explicit rows so it can bind each sample to
    a TaskRef, seed, checkpoint and immutable request ID.
    """
    def __init__(self, manager, *, prompt_token_id, agent_name="rift_usami_service"):
        if manager is None or type(prompt_token_id) is not int or prompt_token_id < 0:
            raise ValueError("VERL AgentLoopManager and tokenizer pad ID are required")
        if not agent_name:
            raise ValueError("registered RIFT Agent Loop name required")
        self.manager, self.prompt_token_id, self.agent_name = manager, prompt_token_id, agent_name

    @staticmethod
    def _required(request, key):
        value = request.get(key)
        if value is None or value == "":
            raise ValueError(f"Agent Loop request requires {key}")
        return value

    def _batch(self, requests):
        try:
            import numpy as np
            import torch
            from verl import DataProto
        except ImportError as exc:
            raise RuntimeError("production Agent Loop batching requires pinned VERL, NumPy and PyTorch") from exc
        size = len(requests)
        if size < 1:
            raise ValueError("non-empty Agent Loop request batch required")
        rows = []
        request_ids = set()
        for index, source in enumerate(requests):
            row = dict(source)
            request_id = self._required(row, "rift_request_id")
            if request_id in request_ids:
                raise ValueError("duplicate Agent Loop request ID")
            request_ids.add(request_id)
            self._required(row, "rift_task_ref")
            mode = row.get("rift_mode")
            if mode not in ("train", "probe"):
                raise ValueError("Agent Loop mode must be train or probe")
            if mode == "train":
                for key in ("rift_group_size", "rift_rollout_id", "rift_sampling_seed",
                            "rift_customer_seed", "rift_expected_checkpoint"):
                    self._required(row, key)
                if type(row["rift_group_size"]) is not int or row["rift_group_size"] < 2:
                    raise ValueError("Service request must belong to a complete GRPO group")
            else:
                for key in ("rift_checkpoint", "rift_sampling_seed", "rift_protocol_ref",
                            "rift_protocol_digest"):
                    self._required(row, key)
                if type(row["rift_sampling_seed"]) is not int:
                    raise ValueError("Probe seed must be an integer")
            session_id = row.get("session_id", index)
            if type(session_id) is not int or session_id < 0:
                raise ValueError("Agent Loop session_id must be a non-negative integer")
            row["session_id"] = session_id
            row["uid"] = self._required(row, "uid")
            row["agent_name"] = row.get("agent_name", self.agent_name)
            row["raw_prompt"] = row.get("raw_prompt", [])
            rows.append(row)

        tensor = torch.full((size, 1), self.prompt_token_id, dtype=torch.long)
        non_tensors = {
            "index": np.arange(size, dtype=np.int64),
            "uid": _object_array([row["uid"] for row in rows]),
            "session_id": np.asarray([row["session_id"] for row in rows], dtype=np.int64),
            "agent_name": _object_array([row["agent_name"] for row in rows]),
            "raw_prompt": _object_array([row["raw_prompt"] for row in rows]),
        }
        keys = set().union(*(row.keys() for row in rows)) - set(non_tensors) - {
            "session_id", "uid", "agent_name", "raw_prompt"}
        for key in keys:
            values = [row.get(key) for row in rows]
            non_tensors[key] = _object_array(values)
        return DataProto.from_dict(
            tensors={"prompts": tensor,
                     "attention_mask": torch.ones_like(tensor),
                     "position_ids": torch.zeros_like(tensor)},
            non_tensors=non_tensors,
            meta_info={"validate": False},
        )

    def generate(self, requests):
        requests = tuple(dict(row) for row in requests)
        batch = self._batch(requests)
        result = _sync_await(self.manager.generate_sequences(batch))
        values = getattr(result, "non_tensor_batch", None)
        if not isinstance(values, dict):
            raise ValueError("VERL AgentLoopManager output is missing non-tensor evidence")
        if len(result) != len(requests):
            raise ValueError("VERL AgentLoopManager returned the wrong batch size")
        outputs = {}
        for index in range(len(requests)):
            request_id = self._value(values, "rift_request_id", index)
            if not isinstance(request_id, str) or request_id in outputs:
                raise ValueError("VERL output request IDs are absent or duplicated")
            outputs[request_id] = {key: self._value(values, key, index) for key in values}
        if set(outputs) != {row["rift_request_id"] for row in requests}:
            raise ValueError("VERL AgentLoopManager changed or omitted request identities")
        return tuple(outputs[row["rift_request_id"]] for row in requests)

    @staticmethod
    def _value(values, key, index):
        if key not in values:
            return None
        value = values[key][index]
        if hasattr(value, "item"):
            try:
                return value.item()
            except (ValueError, TypeError):
                pass
        return value


class ProductionProbeEvaluator:
    """Execute frozen Probe jobs on the selected checkpoint through VERL."""
    def __init__(self, batch, *, seed_base=0, metric_sink=None,
                 round_index=None, metric_name='probe_evaluation_seconds'):
        if not callable(getattr(batch, "generate", None)) or type(seed_base) is not int or seed_base < 0:
            raise ValueError("VERL batch adapter and non-negative seed base required")
        self.batch, self.seed_base = batch, seed_base
        self.metric_sink, self.round_index, self.metric_name = metric_sink, round_index, metric_name

    def evaluate(self, task_ref, checkpoint, seed, protocol_ref, protocol_digest):
        return self.evaluate_many(((task_ref, checkpoint, seed, protocol_ref, protocol_digest),))[0]

    def evaluate_many(self, requests):
        from rift.verl.sampling import rollout_seed
        normalized = tuple(requests)
        if not normalized:
            return ()
        rows, request_ids = [], []
        for index, (task_ref, checkpoint, seed, protocol_ref, protocol_digest) in enumerate(normalized):
            if (not task_ref or not isinstance(checkpoint, Checkpoint) or type(seed) is not int
                    or seed < 0 or not protocol_ref or not protocol_digest):
                raise ValueError("complete frozen Probe request required")
            request_id = sha256(
                f"rift-probe-request-v1:{index}:{task_ref}:{checkpoint.weights_digest}:{seed}:{protocol_digest}".encode()
            ).hexdigest()
            request_ids.append(request_id)
            rows.append({"rift_request_id": request_id, "rift_task_ref": task_ref,
                "rift_mode": "probe", "rift_checkpoint": checkpoint_json(checkpoint),
                "rift_sampling_seed": seed, "rift_protocol_ref": protocol_ref,
                "rift_protocol_digest": protocol_digest, "uid": request_id,
                "session_id": index, "raw_prompt": []})
        started = time.perf_counter()
        output_rows = self.batch.generate(rows)
        if len(output_rows) != len(rows):
            raise ValueError("VERL Probe batch omitted one or more frozen requests")
        if self.metric_sink is not None:
            self.metric_sink(self.metric_name, time.perf_counter() - started,
                round_index=self.round_index, request_count=len(rows))
        trials = []
        for expected, result in zip(normalized, output_rows):
            task_ref, checkpoint, seed, protocol_ref, protocol_digest = expected
            request_id = request_ids[len(trials)]
            expected_meta = (request_id, task_ref, "probe", checkpoint_json(checkpoint), seed,
                             protocol_ref, protocol_digest)
            actual_meta = tuple(result.get(key) for key in (
                "rift_request_id", "rift_task_ref", "rift_mode", "rift_checkpoint", "rift_seed",
                "rift_protocol_ref", "rift_protocol_digest"))
            if actual_meta != expected_meta:
                raise ValueError("VERL Probe output does not bind to its original task/checkpoint/seed/protocol")
            outcome = Outcome(result.get("rift_outcome"))
            replay_verified = result.get("rift_replay_verified") is True
            if outcome is not Outcome.UNKNOWN and not replay_verified:
                outcome = Outcome.UNKNOWN
            evidence_ref = result.get("rift_evidence_ref") if replay_verified else None
            trials.append(ObservedTrial(task_ref, result["rift_episode_id"], checkpoint, seed,
                outcome, protocol_digest, evidence_ref, replay_verified,
                EvidenceLevel.REAL_SERVICE, None))
        return tuple(trials)


class ProductionServiceRolloutCollector:
    """Collect complete groups then reuse RIFT's verified package converter."""
    def __init__(self, batch, *, task_registry, capture_store, episode_resolver=None,
                 metric_sink=None, round_index=None):
        if not callable(getattr(batch, "generate", None)):
            raise TypeError("VERL Agent Loop batch adapter required")
        self.batch, self.task_registry, self.capture_store = batch, task_registry, capture_store
        self.episode_resolver = episode_resolver or capture_store.resolve_episode
        self.metric_sink, self.round_index = metric_sink, round_index

    def collect(self, package, checkpoint):
        from rift.verl.sampling import rollout_seed
        requests, request_to_group = [], {}
        episodes_by_request = {}
        session_id = 0
        for group in package.task_groups:
            for member in range(group.group_size):
                rollout_id = sha256(
                    f"rift-service-rollout-v1:{package.package_id}:{group.task_ref}:{member}".encode()
                ).hexdigest()
                request_id = rollout_id
                sampling_seed = rollout_seed(package.sampling_seed, rollout_id)
                customer_seed = rollout_seed(package.sampling_seed, group.task_ref)
                requests.append({"rift_request_id": request_id,
                    "rift_task_ref": group.task_ref, "rift_mode": "train",
                    "rift_group_size": group.group_size, "rift_rollout_id": rollout_id,
                    "rift_sampling_seed": sampling_seed, "rift_customer_seed": customer_seed,
                    "rift_expected_checkpoint": checkpoint_json(checkpoint), "uid": group.group_id,
                    "session_id": session_id, "raw_prompt": []})
                request_to_group[request_id] = group
                session_id += 1

        def verify_episode(request, group, episode):
            if (episode.task.record_hash != group.task_ref
                    or self.capture_store.verify_trajectory(episode, checkpoint) is not True):
                raise ValueError("recovered Service Episode or exact token evidence is invalid")
            calls = [call for call in episode.model_calls if call.role == "service"]
            if not calls:
                raise ValueError("Service Episode has no model decision evidence")
            for call in calls:
                meta = call.request
                if (meta.get("rift_mode") != "train" or meta.get("task_ref") != group.task_ref
                        or meta.get("rollout_id") != request["rift_rollout_id"]
                        or meta.get("group_id") != group.group_id
                        or meta.get("group_size") != group.group_size
                        or meta.get("sampling_seed") != request["rift_sampling_seed"]
                        or meta.get("customer_seed") != request["rift_customer_seed"]):
                    raise ValueError("native Service call no longer matches the frozen request")

        pending = []
        for request in requests:
            request_id = request["rift_request_id"]
            group = request_to_group[request_id]
            recovered = self.capture_store.resolve_rollout(request["rift_rollout_id"], checkpoint)
            if recovered is None:
                pending.append(request)
            else:
                verify_episode(request, group, recovered)
                episodes_by_request[request_id] = recovered
        started = time.perf_counter()
        output_rows = self.batch.generate(pending) if pending else ()
        rollout_seconds = time.perf_counter() - started
        generated_tokens = 0
        if len(output_rows) != len(pending):
            raise ValueError("VERL omitted frozen Service rollout requests")
        for request, result in zip(pending, output_rows):
            request_id = request["rift_request_id"]
            group = request_to_group[request_id]
            if (result.get("rift_request_id") != request_id
                    or result.get("rift_task_ref") != group.task_ref
                    or result.get("rift_mode") != "train"
                    or result.get("rift_group_id") != group.group_id
                    or result.get("rift_checkpoint") != checkpoint_json(checkpoint)
                    or result.get("rift_seed") != request["rift_sampling_seed"]
                    or not result.get("rift_episode_id")):
                raise ValueError("Service Agent Loop output lost task/group/checkpoint/seed provenance")
            episode = self.episode_resolver(result["rift_episode_id"])
            if episode.record_hash != result["rift_episode_id"]:
                raise ValueError("native Episode resolver returned a mismatched record")
            verify_episode(request, group, episode)
            episodes_by_request[request_id] = episode
            generated_tokens += len(self.capture_store.trajectory_for(episode, checkpoint).response)
        if set(episodes_by_request) != set(request_to_group):
            raise ValueError("one or more frozen Service rollout identities could not be recovered")
        episodes = [episodes_by_request[request["rift_request_id"]] for request in requests]
        request = prepare_package_service_update(package, self.task_registry, episodes,
                                                 checkpoint, self.capture_store)
        if self.metric_sink is not None:
            self.metric_sink('service_rollout_seconds', rollout_seconds,
                round_index=self.round_index, request_count=len(pending))
            self.metric_sink('service_generated_tokens', generated_tokens,
                round_index=self.round_index, source='newly_generated_rollouts')
            self.metric_sink('service_tokens_per_second',
                generated_tokens / rollout_seconds if rollout_seconds > 0 else 0.0,
                round_index=self.round_index, source='newly_generated_rollouts')
            self.metric_sink('service_valid_grpo_groups',
                len({row.group_id for row in request.rollouts}), round_index=self.round_index)
        return request
