"""Disjoint ALFWorld task pools for RSI probe, selection, reward, and audit."""
from dataclasses import dataclass
from collections import Counter, defaultdict

from rsi_core.packages import (PackageProvider, PackageRegistration,
                               ServiceTaskProvider, ServiceTaskRegistration)
from rsi_core.protocol import AdaptationPackage, digest

from .tasks import DATA_VERSION, DOMAIN_ID


DIRECT_PURPOSES = ("service_probe", "service_select", "service_reward")
ADAPT_PURPOSES = ("C_probe", "C_select", "C_reward")


def _round_robin_by_type(tasks):
    grouped = defaultdict(list)
    for task in sorted(tasks, key=lambda row: row.task_ref):
        grouped[task.task_type].append(task)
    result = []
    while any(grouped.values()):
        for task_type in sorted(grouped):
            if grouped[task_type]:
                result.append(grouped[task_type].pop(0))
    return tuple(result)


@dataclass(frozen=True)
class AlfWorldBanks:
    task_provider: ServiceTaskProvider
    task_sets: dict
    package_provider: PackageProvider
    challenge_tasks: tuple
    audit_task_set: object
    manifest_ref: str

    def challenge_catalogue(self):
        counts = Counter(task.task_type for task in self.challenge_tasks)
        return {"benchmark": DATA_VERSION,
                "task_types": [{"task_type": name, "available_tasks": counts[name]}
                               for name in sorted(counts)]}


def build_task_banks(registry, archive, service_snapshot, *, tasks_per_set=1,
                     audit_task_count=1, package_seed=0):
    """Register disjoint train sources and an unseen audit set.

    The nine training roles are split by original task file. Any remaining
    ALFWorld train tasks form the M challenge pool. ``family_id`` and
    ``source_lineage`` identify these disjoint operational subsets; task type
    remains the native ALFRED category and is not treated as a new source.
    """
    if type(tasks_per_set) is not int or tasks_per_set < 1:
        raise ValueError("tasks_per_set must be a positive integer")
    if type(audit_task_count) is not int or audit_task_count < 1:
        raise ValueError("audit_task_count must be a positive integer")
    train = _round_robin_by_type(registry.by_split("train"))
    unseen = tuple(sorted(registry.by_split("valid_unseen"), key=lambda row: row.task_ref))
    roles = ("service_probe", "service_select", "service_reward",
             "C_probe_probe", "C_probe_query", "C_select_probe", "C_select_query",
             "C_reward_probe", "C_reward_query")
    required = len(roles) * tasks_per_set + 1
    if len(train) < required:
        raise ValueError("ALFWorld train split needs at least " + str(required)
                         + " distinct solvable tasks for this RSI period; found " + str(len(train)))
    if len(unseen) < audit_task_count:
        raise ValueError("ALFWorld valid_unseen split needs at least " + str(audit_task_count)
                         + " task(s) for an isolated audit set")

    role_tasks, offset = {}, 0
    for role in roles:
        role_tasks[role] = train[offset:offset + tasks_per_set]
        offset += tasks_per_set
    challenge_tasks = train[offset:]
    if not challenge_tasks:
        raise ValueError("ALFWorld train split has no tasks left for Designer challenge grounding")
    audit_tasks = unseen[:audit_task_count]
    train_origins = {task.origin_ref for task in train}
    if any(task.origin_ref in train_origins for task in audit_tasks):
        raise ValueError("ALFWorld valid_unseen task source overlaps the train split")

    proof = registry.proof
    direct_registrations = []
    for index, purpose in enumerate(DIRECT_PURPOSES):
        source = role_tasks[purpose]
        direct_registrations.append(ServiceTaskRegistration(
            set_id="alfworld-" + purpose,
            domain_id=DOMAIN_ID,
            family_id=DATA_VERSION + "/" + purpose,
            source_lineage=DATA_VERSION + "/train/" + purpose,
            partition="meta-train", purpose=purpose,
            task_proofs=tuple(proof(task.task_ref) for task in source),
            seed=package_seed + index))
    direct_registrations.append(ServiceTaskRegistration(
        set_id="alfworld-audit-valid-unseen", domain_id=DOMAIN_ID,
        family_id=DATA_VERSION + "/valid-unseen-audit",
        source_lineage=DATA_VERSION + "/valid_unseen/audit",
        partition="meta-test", purpose="audit",
        task_proofs=tuple(proof(task.task_ref) for task in audit_tasks),
        seed=package_seed + 1000))
    task_provider = ServiceTaskProvider(direct_registrations, archive, registry.verify_task)
    task_sets = {purpose: task_provider.sets(purpose)[0] for purpose in DIRECT_PURPOSES}
    audit_task_set = task_provider.sets("audit")[0]

    role_refs = {role: tuple(task.task_ref for task in rows)
                 for role, rows in role_tasks.items()}
    package_provider = build_adaptation_provider(
        registry, archive, service_snapshot, role_refs, package_seed=package_seed)
    manifest = archive.save({
        "kind": "alfworld_rsi_task_partition_v1",
        "data_version": DATA_VERSION,
        "train_refs_by_role": {role: [task.task_ref for task in rows]
                               for role, rows in role_tasks.items()},
        "challenge_task_refs": [task.task_ref for task in challenge_tasks],
        "challenge_task_types": dict(Counter(task.task_type for task in challenge_tasks)),
        "audit_task_refs": [task.task_ref for task in audit_tasks],
        "direct_bank_ref": task_provider.manifest_ref,
        "adaptation_bank_ref": package_provider.manifest_ref,
    })
    return AlfWorldBanks(task_provider, task_sets, package_provider,
                         challenge_tasks, audit_task_set, manifest)


def build_adaptation_provider(registry, archive, service_snapshot, role_refs, *, package_seed=0):
    """Bind disjoint registered ALFWorld adaptation files to the current S_ref."""
    registrations = []
    for index, purpose in enumerate(ADAPT_PURPOSES):
        probe_refs = tuple(role_refs[purpose + "_probe"])
        query_refs = tuple(role_refs[purpose + "_query"])
        package = AdaptationPackage(
            package_id="alfworld-" + purpose,
            domain_id=DOMAIN_ID,
            family_id=DATA_VERSION + "/" + purpose,
            source_lineage=DATA_VERSION + "/train/" + purpose,
            partition="meta-train", purpose=purpose,
            p0_ref=service_snapshot.ref, support=(),
            probe_refs=probe_refs, query_refs=query_refs,
            capabilities={"task_interface": "ALFWorld TextWorld; observation plus native admissible commands"},
            seed=package_seed + 10 + index)
        registrations.append(PackageRegistration(
            package, (), tuple(registry.proof(task_ref) for task_ref in probe_refs + query_refs)))
    return PackageProvider(registrations, archive, registry.verify_task)


class AlfWorldChallengeBridge:
    """Ground a typed Designer challenge into one registered train task."""

    def __init__(self, registry, challenge_tasks):
        self.registry = registry
        self.challenge_tasks = tuple(challenge_tasks)
        self.by_type = defaultdict(list)
        for task in self.challenge_tasks:
            self.by_type[task.task_type].append(task)
        self.catalogue = {task_type: len(rows)
                          for task_type, rows in sorted(self.by_type.items())}

    def public_catalogue(self):
        return {"benchmark": DATA_VERSION,
                "task_types": [{"task_type": name, "available_tasks": count}
                               for name, count in self.catalogue.items()]}

    def certify(self, proposal):
        if (not isinstance(proposal, dict)
                or set(proposal) != {"task_type", "hypothesis"}
                or not isinstance(proposal.get("task_type"), str)
                or not isinstance(proposal.get("hypothesis"), str)
                or not proposal["hypothesis"].strip()):
            return {"status": "INVALID", "reason_code": "invalid_alfworld_challenge_shape"}
        candidates = self.by_type.get(proposal["task_type"], ())
        if not candidates:
            return {"status": "INVALID", "reason_code": "task_type_not_in_train_challenge_pool"}
        # Deterministic grounding is bound to the full registered candidate set.
        ordered = sorted(candidates, key=lambda task: task.task_ref)
        index = int(digest({"proposal": proposal,
                            "candidates": [row.task_ref for row in ordered]})[:16], 16) % len(ordered)
        task = ordered[index]
        task_proof = self.registry.proof(task.task_ref)
        if self.registry.verify_task(DOMAIN_ID, task_proof) is not True:
            return {"status": "UNVERIFIED", "reason_code": "alfworld_task_source_certificate_invalid"}
        return {"status": "CERTIFIED", "task_ref": task.task_ref,
                "certificate_ref": task.certificate_ref,
                "certification_scope": "alfworld_train_task_source_v1",
                "grounding": {"task_type": task.task_type,
                              "selection": "stable-hash-over-registered-train-candidates"},
                "frozen": True}
