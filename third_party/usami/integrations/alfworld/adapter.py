"""ALFWorld TextWorld implementation of the RSI Core EnvironmentAdapter."""
from dataclasses import asdict
import copy
import hashlib
import json
import os
from pathlib import Path
import uuid

from rsi_core.protocol import (CandidateError, InfrastructureError, TaskResult,
                               canonical, digest)
from rsi_core.programs import ServiceSnapshot

from .tasks import AlfWorldTask, DOMAIN_ID, TASK_TYPE_IDS


ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_alfworld_action",
        "description": "Choose the next ALFWorld TextWorld command.",
        "parameters": {
            "type": "object",
            "properties": {"action": {"type": "string"}},
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}


def _at_zero(value):
    if isinstance(value, (str, bytes, dict)):
        return value
    try:
        return value[0] if len(value) == 1 else value
    except (TypeError, IndexError):
        return value


def _native_won(info, score):
    if "won" not in info:
        raise InfrastructureError("ALFWorld did not return its requested native 'won' result")
    return bool(_at_zero(info["won"])) or float(_at_zero(score)) >= 1.0


def _episode_hash(record):
    return digest({key: value for key, value in record.items()
                   if key != "trajectory_hash"})


class AlfWorldEnvironmentFactory:
    """Create a one-game AlfredTWEnv without exposing its expert wrapper data."""

    def __init__(self, data_root, config_path=None, *, max_steps=50):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("ALFWorld max_steps must be a positive integer")
        self.data_root = Path(data_root).expanduser().resolve()
        config_path = config_path or os.environ.get("ALFWORLD_CONFIG")
        self.config_path = str(Path(config_path).expanduser().resolve()) if config_path else ""
        self.config_sha256 = (hashlib.sha256(Path(self.config_path).read_bytes()).hexdigest()
                              if self.config_path else "")
        self.max_steps = int(max_steps)
        self.config_ref = digest({"data_root": str(self.data_root),
                                  "config_sha256": self.config_sha256,
                                  "max_steps": self.max_steps,
                                  "domain_randomization": False,
                                  "batch_size": 1})

    def __call__(self, task, seed):
        try:
            import alfworld.agents.environment as environment
        except ImportError as exc:
            raise InfrastructureError(
                "ALFWorld is not installed; install alfworld[full] and download its TextWorld data"
            ) from exc
        os.environ["ALFWORLD_DATA"] = str(self.data_root)
        if not self.config_path:
            raise InfrastructureError("pass --config or set ALFWORLD_CONFIG to ALFWorld configs/base_config.yaml")
        if hashlib.sha256(Path(self.config_path).read_bytes()).hexdigest() != self.config_sha256:
            raise InfrastructureError("ALFWorld config changed after the task registry was frozen")
        try:
            import yaml
        except ImportError as exc:
            raise InfrastructureError("ALFWorld YAML dependency is unavailable") from exc
        with Path(self.config_path).open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        config = copy.deepcopy(config)
        # AlfredTWEnv scans each dataset root at construction. Point it at this
        # task's directory; that directory contains exactly one traj_data.json.
        config["dataset"]["data_path"] = str(Path(task.game_file).parent)
        config["dataset"]["num_train_games"] = 1
        config["env"]["type"] = "AlfredTWEnv"
        config["env"]["task_types"] = [TASK_TYPE_IDS[task.task_type]]
        config["env"]["domain_randomization"] = False
        config["general"]["random_seed"] = int(seed)
        # DQN mode registers the TextWorld limit without installing ALFWorld's
        # expert wrapper, so expert plans are never generated for the episode.
        config["general"]["training_method"] = "dqn"
        config["rl"]["training"]["max_nb_steps_per_episode"] = self.max_steps
        env_type = config["env"]["type"]
        env = environment.get_environment(env_type)(config, train_eval="train")
        if len(env.game_files) != 1 or Path(env.game_files[0]).resolve() != Path(task.game_file).resolve():
            raise InfrastructureError("ALFWorld did not isolate the registered game file")
        gym_env = env.init_env(batch_size=1)
        if callable(getattr(gym_env, "seed", None)):
            gym_env.seed(int(seed))
        return _OfficialTextWorldSession(env, gym_env, task, self.config_ref)


class _OfficialTextWorldSession:
    def __init__(self, owner, env, task, config_ref):
        self.owner, self.env, self.task = owner, env, task
        self.config_ref = config_ref

    def reset(self):
        observations, info = self.env.reset()
        observed_game = info.get("extra.gamefile")
        if observed_game and Path(_at_zero(observed_game)).resolve() != Path(self.task.game_file).resolve():
            raise InfrastructureError("ALFWorld reset returned a different registered game")
        commands = tuple(str(command) for command in _at_zero(info["admissible_commands"]))
        return {"observation": str(_at_zero(observations)),
                "admissible_actions": commands}

    def step(self, action):
        observations, scores, dones, infos = self.env.step([action])
        info = {key: _at_zero(value) for key, value in infos.items()
                if key in {"won"}}
        if "won" not in info:
            raise InfrastructureError("ALFWorld step omitted native 'won' result")
        return {"observation": str(_at_zero(observations)),
                "admissible_actions": tuple(str(command) for command in
                                             _at_zero(infos["admissible_commands"])),
                "score": float(_at_zero(scores)),
                "done": bool(_at_zero(dones)),
                "won": _native_won(info, scores)}

    def close(self):
        self.env.close()


class AlfWorldAdapter:
    domain_id = DOMAIN_ID

    def __init__(self, tasks, root, env_factory, *, max_steps=50, model=None):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("ALFWorld max_steps must be a positive integer")
        self.tasks = dict(tasks)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.env_factory = env_factory
        self.max_steps = int(max_steps)
        self.model = model

    def capabilities(self):
        return {
            "domain_id": self.domain_id,
            "feedback_protocol": "rsi-alfworld-public-trajectory-v1",
            "task_interface": "ALFWorld TextWorld task with native goal observation",
            "actions": "native admissible_commands; selected string is sent unchanged to ALFWorld",
            "score": "ALFWorld native won result; completed unsolved episodes score 0",
            "hidden_observation_fields": ["expert_plan", "game_file", "task_source_certificate"],
            "max_steps": self.max_steps,
        }

    def research_feedback(self, result):
        trajectory = result.feedback["trajectory"]
        return {"protocol": "rsi-alfworld-public-trajectory-v1",
                "run_ref": result.run_ref, "status": result.status,
                "score": result.score,
                "terminal_reason": result.feedback["terminal_reason"],
                "trajectory": trajectory}

    def run_task(self, program, task_ref, models, ledger, worker, *, seed):
        task = self.tasks[task_ref]
        if isinstance(program, ServiceSnapshot):
            if program.business_model_ref != models.business.ref:
                raise CandidateError("Service snapshot does not use fixed business model B")
            policy, experience = program.policy, program.experience.value
            service_ref, experience_ref = program.ref, program.experience_ref
        else:
            policy, experience = program, {}
            service_ref, experience_ref = program.ref, ""
        env = None
        record = {"kind": "alfworld_episode_v1", "task_ref": task_ref,
                  "task_source_split": task.split, "task_type": task.task_type,
                  "origin_ref": task.origin_ref,
                  "game_sha256": task.game_sha256,
                  "trajectory_sha256": task.trajectory_sha256,
                  "program_ref": program.ref, "policy_ref": policy.ref,
                  "service_ref": service_ref, "experience_ref": experience_ref,
                  "models": asdict(models), "seed": int(seed),
                  "environment_config_ref": getattr(self.env_factory, "config_ref", "injected-test-env"),
                  "max_steps": self.max_steps, "initial_state": None,
                  "steps": [], "model_events": [], "terminal": {},
                  "status": "UNKNOWN", "score": 0.0}
        state = {}
        won = False
        try:
            env = self.env_factory(task, int(seed))
            raw_initial = env.reset()
            initial = {"observation": raw_initial["observation"],
                       "admissible_actions": list(raw_initial["admissible_actions"])}
            record["initial_state"] = initial
            observation = initial
        except Exception as exc:
            record["terminal"] = {"kind": "environment_error", "phase": "reset",
                                  "error_type": type(exc).__name__, "error": str(exc)}
            status = "UNKNOWN"
        else:
            status = "VERIFIED"
            for step_index in range(self.max_steps):
                view = {"observation": observation["observation"],
                        "admissible_actions": list(observation["admissible_actions"]),
                        "history": [{"observation": row["observation"],
                                     "action": row["action"],
                                     "next_observation": row["next_observation"]}
                                    for row in record["steps"]],
                        "service_experience": experience}
                try:
                    ledger.charge("solver", research_steps=1)
                    decision = worker.invoke(policy, view, state, ledger, "solver")
                    if not isinstance(decision, dict) or not isinstance(decision.get("state"), dict):
                        raise CandidateError("ALFWorld P must return a decision and explicit state object")
                    state = decision["state"]
                    kind = decision.get("kind")
                    if kind == "ACTION":
                        if set(decision) != {"kind", "state", "action"} or not isinstance(decision["action"], str):
                            raise CandidateError("ALFWorld ACTION must contain one command string")
                        action = decision["action"]
                    elif kind == "MODEL":
                        if set(decision) != {"kind", "state", "messages"}:
                            raise CandidateError("ALFWorld MODEL must return messages only")
                        if self.model is None:
                            raise InfrastructureError("ALFWorld behavior model is not configured")
                        messages = decision["messages"]
                        if (not isinstance(messages, list) or not messages
                                or any(not isinstance(row, dict)
                                       or set(row) != {"role", "content"}
                                       or row["role"] not in {"system", "user", "assistant"}
                                       or not isinstance(row["content"], str) for row in messages)):
                            raise CandidateError("ALFWorld P messages must be role/content text rows")
                        request = {**models.business.sampling,
                                   "model": models.business.model_id,
                                   "messages": messages, "tools": [ACTION_TOOL],
                                   "tool_choice": {"type": "function", "function": {
                                       "name": "choose_alfworld_action"}},
                                   "parallel_tool_calls": False, "stream": False, "n": 1}
                        if ledger.remaining("model_calls") < 1:
                            raise CandidateError("ALFWorld behavior model-call budget exhausted")
                        allowance = min(2048, int(ledger.remaining("model_tokens")))
                        if allowance < 1:
                            raise CandidateError("ALFWorld behavior model-token budget exhausted")
                        ledger.charge("solver", model_calls=1)
                        try:
                            completion = self.model.complete(
                                models.business, request, max_tokens=allowance,
                                seed=int(seed) + step_index)
                        except Exception as exc:
                            record["model_events"].append({
                                "request": request,
                                "status": "ERROR",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            })
                            record["terminal"] = {
                                "kind": "model_error", "phase": "decision",
                                "step": step_index, "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                            status = "UNKNOWN"
                            break
                        ledger.charge("solver", model_tokens=(completion.prompt_tokens
                                                               + completion.completion_tokens))
                        model_event = {"request": request, "completion": asdict(completion)}
                        record["model_events"].append(model_event)
                        action = self._action_from_response(completion.output)
                        model_event["selected_action"] = action
                    else:
                        raise CandidateError("ALFWorld P returned an unregistered decision kind")
                except CandidateError as exc:
                    record["terminal"] = {"kind": "candidate_error", "phase": "decision",
                                          "step": step_index,
                                          "error_type": type(exc).__name__, "error": str(exc)}
                    status = "CANDIDATE_FAILED"
                    break
                try:
                    result = env.step(action)
                except Exception as exc:
                    record["terminal"] = {"kind": "environment_error", "phase": "step",
                                          "step": step_index, "action": action,
                                          "error_type": type(exc).__name__, "error": str(exc)}
                    status = "UNKNOWN"
                    break
                ledger.charge("environment", tool_steps=1)
                row = {"observation": observation["observation"], "action": action,
                       "admissible_actions": list(observation["admissible_actions"]),
                       "next_observation": result["observation"],
                       "next_admissible_actions": list(result["admissible_actions"]),
                       "reward": result["score"], "done": result["done"], "won": result["won"]}
                record["steps"].append(row)
                observation = {"observation": result["observation"],
                               "admissible_actions": result["admissible_actions"]}
                won = result["won"]
                if result["done"]:
                    record["terminal"] = {"kind": "environment_done", "won": won}
                    break
            else:
                record["terminal"] = {"kind": "step_limit", "won": won,
                                      "max_steps": self.max_steps}
            if status == "VERIFIED":
                record["status"] = status
                record["score"] = float(won)
        if status != "VERIFIED":
            record["status"] = status
        if env is not None:
            env.close()
        run_ref = uuid.uuid4().hex
        record["run_ref"] = run_ref
        record["trajectory_hash"] = _episode_hash(record)
        path = self.root / (run_ref + ".json")
        path.write_text(canonical(record) + "\n", encoding="utf-8")
        replay = self.replay(run_ref)
        if replay.get("verified") is not True:
            raise InfrastructureError("ALFWorld native Replay failed; trace retained at " + str(path))
        public_trajectory = [{key: row[key] for key in (
            "observation", "action", "next_observation", "reward", "done", "won")}
            for row in record["steps"]]
        return TaskResult(run_ref, record["score"], status, True,
                          tool_steps=len(record["steps"]),
                          model_calls=len(record["model_events"]),
                          model_tokens=sum(row.get("completion", {}).get("prompt_tokens", 0)
                                           + row.get("completion", {}).get("completion_tokens", 0)
                                           for row in record["model_events"]),
                          feedback={"terminal_reason": record["terminal"].get("kind", ""),
                                    "trajectory": public_trajectory},
                          forbidden_behavior=False)

    @staticmethod
    def _action_from_response(raw):
        try:
            choices = raw["choices"]
            message = choices[0]["message"] if len(choices) == 1 else {}
            calls = message.get("tool_calls", [])
            if len(calls) != 1:
                raise CandidateError("ALFWorld model must return exactly one action function call")
            function = calls[0]["function"]
            if function.get("name") != "choose_alfworld_action":
                raise CandidateError("ALFWorld model returned an unexpected function")
            arguments = json.loads(function["arguments"])
            if (not isinstance(arguments, dict) or set(arguments) != {"action"}
                    or not isinstance(arguments["action"], str)):
                raise CandidateError("ALFWorld action function requires one string action")
            return arguments["action"]
        except CandidateError:
            raise
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise CandidateError("ALFWorld model response has no complete action call") from exc

    def replay(self, run_ref):
        path = self.root / (run_ref + ".json")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("kind") != "alfworld_episode_v1" or record.get("run_ref") != run_ref:
            raise InfrastructureError("ALFWorld episode artifact identity differs")
        if record.get("trajectory_hash") != _episode_hash(record):
            return {"verified": False, "reason": "episode record hash mismatch"}
        task = self.tasks[record["task_ref"]]
        if (record["origin_ref"], record["game_sha256"], record["trajectory_sha256"]) != (
                task.origin_ref, task.game_sha256, task.trajectory_sha256):
            return {"verified": False, "reason": "registered task source changed"}
        env = None
        try:
            terminal = record["terminal"]
            try:
                env = self.env_factory(task, record["seed"])
            except Exception as exc:
                if terminal.get("kind") == "environment_error" and terminal.get("phase") == "reset":
                    verified = (type(exc).__name__ == terminal["error_type"]
                                and str(exc) == terminal["error"])
                    return {"verified": verified, "status": record["status"],
                            "reason": "environment setup error replayed" if verified
                            else "environment setup error differs"}
                return {"verified": False, "reason": "environment setup failed during Replay",
                        "error": type(exc).__name__ + ": " + str(exc)}
            if terminal.get("kind") == "environment_error" and terminal.get("phase") == "reset":
                try:
                    env.reset()
                except Exception as exc:
                    verified = (type(exc).__name__ == terminal["error_type"]
                                and str(exc) == terminal["error"])
                    return {"verified": verified, "status": record["status"],
                            "reason": "reset error replayed" if verified else "reset error differs"}
                return {"verified": False, "reason": "reset failure did not recur"}
            raw_observation = env.reset()
            observation = {"observation": raw_observation["observation"],
                           "admissible_actions": list(raw_observation["admissible_actions"])}
            if observation != record["initial_state"]:
                return {"verified": False, "reason": "initial observation differs"}
            for index, expected in enumerate(record["steps"]):
                if (observation["observation"] != expected["observation"]
                        or list(observation["admissible_actions"]) != expected["admissible_actions"]):
                    return {"verified": False, "reason": "pre-action state differs", "step": index}
                try:
                    actual = env.step(expected["action"])
                except Exception as exc:
                    return {"verified": False, "reason": "recorded action now raises",
                            "step": index, "error": type(exc).__name__ + ": " + str(exc)}
                comparable = {"observation": actual["observation"],
                              "admissible_actions": list(actual["admissible_actions"]),
                              "reward": actual["score"], "done": actual["done"], "won": actual["won"]}
                if comparable != {"observation": expected["next_observation"],
                                  "admissible_actions": expected["next_admissible_actions"],
                                  "reward": expected["reward"], "done": expected["done"],
                                  "won": expected["won"]}:
                    return {"verified": False, "reason": "native transition differs", "step": index}
                observation = {"observation": actual["observation"],
                              "admissible_actions": actual["admissible_actions"]}
            if terminal.get("kind") == "environment_error" and terminal.get("phase") == "step":
                try:
                    env.step(terminal["action"])
                except Exception as exc:
                    if (type(exc).__name__ != terminal["error_type"]
                            or str(exc) != terminal["error"]):
                        return {"verified": False, "reason": "environment error differs"}
                else:
                    return {"verified": False, "reason": "environment error did not recur"}
            current_hash = _episode_hash(record)
            return {"verified": current_hash == record["trajectory_hash"],
                    "status": record["status"], "score": record["score"],
                    "steps": len(record["steps"]), "trajectory_hash": current_hash,
                    "replay_scope": "native environment actions; model calls are not reissued"}
        finally:
            if env is not None:
                env.close()
