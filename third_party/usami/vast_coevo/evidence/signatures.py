"""Conservative signatures of verified facts supplied by the environment."""
from __future__ import annotations
from ..schema import digest


def failure_fingerprint(domain, operation, reference_codes, relevant_prestate,
                        permission_scope, causal_prerequisites, expected, actual,
                        normalization_scope="pending_causal_normalization"):
    # Unknown paths MUST retain an exact causal discriminator from the adapter.
    if normalization_scope == "pending_causal_normalization" and not relevant_prestate.get("exact_prestate_hash"):
        raise ValueError("provisional normalization requires an exact prestate discriminator")
    return digest(dict(schema_version=1, domain=domain, operation=operation,
                       reference_codes=sorted(set(reference_codes)), relevant_prestate=relevant_prestate,
                       permission_scope=permission_scope, causal_prerequisites=list(causal_prerequisites),
                       expected=expected, actual=actual, normalization_scope=normalization_scope))


def failure_counts(evidence):
    failures = [failure for item in evidence for failure in item.failures]
    return {"raw_events": len(failures), "unique_fingerprints": len({f.fingerprint for f in failures}),
            "confirmed_feature_types": len({f.fingerprint for f in failures if f.normalization_scope == "policy_relative_features"}),
            "provisional_types": len({f.fingerprint for f in failures if f.normalization_scope == "pending_causal_normalization"})}
