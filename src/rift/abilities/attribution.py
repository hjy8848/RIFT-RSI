"""Constrained hypothesis-to-ability matching from public diagnosis evidence."""
from dataclasses import dataclass
from math import isfinite
import unicodedata
from typing import Protocol, Sequence

from rift.abilities.records import AbilityAttributionRecord, AbilityCandidate
from rift.abilities.registry import AbilityRegistry
from rift.banks.core import public_projection
from rift.banks.records import TaskEvidenceRecord, CapabilityState, EvidenceLevel
from rift.schemas import DataPurpose


@dataclass(frozen=True)
class AbilityMatchCandidate:
    """A judge may select only a registered ID and cite visible diagnosis rows."""
    ability_id: str
    confidence: float
    evidence_indices: tuple[int, ...]
    rationale: str


@dataclass(frozen=True)
class AbilityFailureMatch:
    candidate_index: int
    confidence: float
    rationale: str


@dataclass(frozen=True)
class AbilityResolution:
    ability_id: str
    confidence: float
    decision: str
    method: str
    rationale: str
    evidence_refs: tuple[str, ...]


class ConstrainedAbilityMatcher(Protocol):
    def match(self, hypothesis: str, public_summaries: Sequence,
              candidates: Sequence[AbilityCandidate],
              public_failure_summaries: Sequence) -> Sequence[AbilityMatchCandidate]: ...
    def associate_failures(self, ability: AbilityCandidate,
                           public_failure_summaries: Sequence) -> Sequence[AbilityFailureMatch]: ...


def _normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in text if character.isalnum())


class ExactAbilityMatcher:
    """No embedding or hidden labels: match normalized descriptions/aliases only."""
    def match(self, hypothesis, public_summaries, candidates, public_failure_summaries=()):
        target = _normalize(hypothesis)
        matches = []
        for candidate in candidates:
            phrases = (candidate.description,) + candidate.aliases
            if target and any(target == _normalize(phrase) for phrase in phrases):
                evidence_indices = self._matching_indices(candidate, public_summaries)
                if evidence_indices:
                    matches.append(AbilityMatchCandidate(candidate.ability_id, 1.0,
                        evidence_indices,
                        "normalized exact ability match grounded in public failure observations"))
        return tuple(matches)

    def associate_failures(self, ability, public_failure_summaries):
        return tuple(AbilityFailureMatch(index, 1.0,
            "normalized exact ability phrase appears in the public failure observation")
            for index in self._matching_indices(ability, public_failure_summaries))

    @staticmethod
    def _matching_indices(ability, public_failure_summaries):
        phrases = tuple(_normalize(text) for text in (ability.description,) + ability.aliases)
        selected = []
        for index, summary in enumerate(public_failure_summaries):
            visible = _normalize(" ".join(event.content for event in summary.events))
            if any(phrase and phrase in visible for phrase in phrases):
                selected.append(index)
        return tuple(selected)


class AbilityAttributor:
    def __init__(self, matcher=None, *, confidence_threshold=0.8,
                 min_margin=0.15, failure_link_threshold=0.8,
                 record_verifier=None):
        if (not 0 <= failure_link_threshold <= confidence_threshold <= 1
                or not 0 <= min_margin <= 1):
            raise ValueError("ability confidence thresholds must be ordered in [0, 1]")
        self.matcher = matcher or ExactAbilityMatcher()
        if not callable(getattr(self.matcher, "match", None)):
            raise TypeError("ConstrainedAbilityMatcher required")
        self.confidence_threshold, self.min_margin = confidence_threshold, min_margin
        self.failure_link_threshold = failure_link_threshold
        self.record_verifier = record_verifier
        if not callable(getattr(record_verifier, "verify_record", None)):
            raise TypeError("trusted TaskEvidenceRecord verifier required")

    def resolve(self, hypothesis, diagnosis_records, registry: AbilityRegistry, *, round_index,
                failure_candidates=()):
        """Match a Designer hypothesis; ambiguous cases get a new provisional ID.

        The matcher receives only public summaries and ability descriptions.
        Host TaskRefs and evidence refs are joined after the decision; no
        private identity, Oracle annotation, or evidence reference is exposed.
        """
        if not isinstance(registry, AbilityRegistry) or not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError("hypothesis and persistent AbilityRegistry required")
        if type(round_index) is not int or round_index < 0:
            raise ValueError("non-negative attribution round required")
        rows = tuple(diagnosis_records)
        if not rows or any(not isinstance(row, TaskEvidenceRecord) for row in rows):
            raise ValueError("verified public diagnosis records required")
        summaries = []
        host_evidence_refs = []
        for row in rows:
            if (row.purpose is not DataPurpose.DIAGNOSIS
                    or row.identity.purpose is not DataPurpose.DIAGNOSIS
                    or row.state() is not CapabilityState.ACTIVE_WEAKNESS
                    or any(trial.valid and trial.evidence_level is not EvidenceLevel.REAL_SERVICE
                           for trial in row.trials)):
                raise ValueError("ability attribution accepts active Diagnosis failures only")
            if self.record_verifier.verify_record(row) is not True:
                raise ValueError("Diagnosis evidence failed trusted verification")
            summary = public_projection(row).summary
            refs = tuple(trial.evidence_ref for trial in row.trials
                         if trial.valid and trial.public_summary is not None
                         and trial.outcome.value == "FAILURE")
            if not refs:
                raise ValueError("Diagnosis record has no verified public failure evidence")
            summaries.append(summary)
            host_evidence_refs.append(refs)
            registry.mark_diagnosis_seen(row.task_ref, row.identity)

        hidden_rows, hidden_summaries, hidden_refs = [], [], []
        for row in tuple(failure_candidates):
            if (not isinstance(row, TaskEvidenceRecord)
                    or row.purpose is not DataPurpose.RECOVERY_FEEDBACK
                    or row.identity.purpose is not DataPurpose.RECOVERY_FEEDBACK
                    or row.state() is not CapabilityState.ACTIVE_WEAKNESS
                    or any(trial.valid and trial.evidence_level is not EvidenceLevel.REAL_SERVICE
                           for trial in row.trials)):
                raise ValueError("ability failure candidates must be held-out Recovery failures")
            if self.record_verifier.verify_record(row) is not True:
                raise ValueError("held-out failure candidate evidence failed verification")
            failures = tuple(trial for trial in row.trials if trial.valid
                             and trial.outcome.value == "FAILURE"
                             and trial.public_summary is not None)
            if not failures:
                continue
            hidden_rows.append(row)
            hidden_summaries.append(failures[-1].public_summary)
            hidden_refs.append(tuple(trial.evidence_ref for trial in failures))

        all_evidence = tuple(dict.fromkeys(
            tuple(ref for refs in host_evidence_refs for ref in refs)
            + tuple(ref for refs in hidden_refs for ref in refs)))
        previous = registry.prior_attribution(hypothesis, round_index, all_evidence)
        if previous is not None:
            return AbilityResolution(previous.ability_id, previous.confidence,
                previous.decision, previous.method, previous.rationale, previous.evidence_refs)

        candidates = registry.candidates()
        matches = tuple(self.matcher.match(hypothesis.strip(), tuple(summaries),
                                           candidates, tuple(hidden_summaries)))
        for match in matches:
            if (not isinstance(match, AbilityMatchCandidate)
                    or match.ability_id not in {candidate.ability_id for candidate in candidates}
                    or not isinstance(match.confidence, (int, float))
                    or not isfinite(match.confidence) or not 0 <= match.confidence <= 1
                    or not match.rationale
                    or not match.evidence_indices
                    or len(set(match.evidence_indices)) != len(match.evidence_indices)
                    or any(type(index) is not int or index < 0 or index >= len(rows)
                           for index in match.evidence_indices)):
                raise ValueError("ability matcher returned an ungrounded or unregistered match")
        ranked = sorted(matches, key=lambda item: (-item.confidence, item.ability_id))
        best = ranked[0] if ranked else None
        margin = best.confidence - ranked[1].confidence if len(ranked) > 1 else 1.0
        accepted = (best is not None and best.confidence >= self.confidence_threshold
                    and margin >= self.min_margin)

        if accepted:
            ability = registry.get(best.ability_id)
            evidence_indices = best.evidence_indices
            decision, method = "matched", type(self.matcher).__name__
            confidence, rationale = best.confidence, best.rationale
        else:
            ability = registry.create(hypothesis.strip(), round_index=round_index)
            evidence_indices = ()
            decision = "provisional"
            method = type(self.matcher).__name__ if best is not None else "no_confident_match"
            confidence = best.confidence if best is not None else 0.0
            rationale = ("ambiguous or low-confidence match; created a provisional ability"
                         if best is not None else "no existing ability matched; created a provisional ability")

        linked_refs = []
        if accepted:
            for index in evidence_indices:
                row = rows[index]
                refs = host_evidence_refs[index]
                linked_refs.extend(refs)
                registry.link_failure_task(ability.ability_id, row, refs,
                    evidence_verifier=self.record_verifier, confidence=confidence,
                    round_index=round_index, rationale=rationale)
        associate = getattr(self.matcher, "associate_failures", None)
        ability_candidate = next(candidate for candidate in registry.candidates()
                                 if candidate.ability_id == ability.ability_id)

        def apply_associations(associations, source_rows, source_refs):
            seen_indices = set()
            for association in associations:
                if (not isinstance(association, AbilityFailureMatch)
                        or type(association.candidate_index) is not int
                        or association.candidate_index < 0
                        or association.candidate_index >= len(source_rows)
                        or association.candidate_index in seen_indices
                        or not isinstance(association.confidence, (int, float))
                        or not isfinite(association.confidence)
                        or not 0 <= association.confidence <= 1
                        or not association.rationale):
                    raise ValueError("failure matcher returned an invalid candidate association")
                seen_indices.add(association.candidate_index)
                if association.confidence < self.failure_link_threshold:
                    continue
                index = association.candidate_index
                row, refs = source_rows[index], source_refs[index]
                registry.link_failure_task(ability.ability_id, row, refs,
                    evidence_verifier=self.record_verifier,
                    confidence=association.confidence, round_index=round_index,
                    rationale=association.rationale)
                linked_refs.extend(refs)

        if callable(associate):
            apply_associations(tuple(associate(ability_candidate, tuple(summaries))),
                               rows, host_evidence_refs)
            if hidden_rows:
                apply_associations(tuple(associate(ability_candidate, tuple(hidden_summaries))),
                                   hidden_rows, hidden_refs)
        evidence_refs = tuple(dict.fromkeys(linked_refs))
        attribution = AbilityAttributionRecord(hypothesis.strip(), ability.ability_id,
            round_index, confidence, decision, method, rationale, evidence_refs,
            all_evidence)
        registry.record_attribution(attribution)
        return AbilityResolution(ability.ability_id, confidence, decision,
                                 method, rationale, evidence_refs)
