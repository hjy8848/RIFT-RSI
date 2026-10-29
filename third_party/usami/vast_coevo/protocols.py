"""Only standard protocols connect the Core to environments and providers."""
from __future__ import annotations
from typing import Dict, Protocol, Tuple
from pydantic import JsonValue

from .models import EpisodeConfig, EpisodeEvidence, PublicContext, ReplayResult
from .skills.library import LibrarySnapshot


class CompletionClient(Protocol):
    def complete(self, request: Dict[str, JsonValue]) -> Dict[str, JsonValue]: ...


class EnvironmentAdapter(Protocol):
    def run_episode(self, config: EpisodeConfig, customer: LibrarySnapshot,
                    service: LibrarySnapshot, episode_id: str) -> EpisodeEvidence: ...
    def run_service_task(self, config: EpisodeConfig, task_ref: str,
                         customer: LibrarySnapshot, service: LibrarySnapshot,
                         episode_id: str) -> EpisodeEvidence: ...
    def replay(self, evidence: EpisodeEvidence, customer: LibrarySnapshot,
               service: LibrarySnapshot) -> ReplayResult: ...
    def run_failure_case(self, source: EpisodeEvidence, customer: LibrarySnapshot,
                         service: LibrarySnapshot, episode_id: str) -> EpisodeEvidence: ...
    def public_capabilities(self, config: EpisodeConfig) -> Tuple[str, ...]: ...
    def initial_observation(self, config: EpisodeConfig) -> PublicContext: ...
