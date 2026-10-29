"""Host-owned update receipts for verified VERL checkpoint artifacts.

This ledger is an audit record created by the trusted backend after its worker
and synchronization calls. It is not a defense against a malicious host and
cannot by itself prove that the worker really ran.
"""
from dataclasses import asdict
import json
from pathlib import Path
from rift.training.trace import digest
from rift.training.contracts import CompletedUpdate

class UpdateLedger:
    def __init__(self, root, checkpoint_store):
        self.root, self.checkpoint_store = Path(root), checkpoint_store
    def _path(self, run_id):
        if not run_id or len(run_id) > 128 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
            raise ValueError('invalid update run ID')
        return self.root / (run_id + '.json')
    def record_update(self, run_id, request, after, metrics):
        before = request.policy_before
        if not self.checkpoint_store.verify(before) or not self.checkpoint_store.verify(after):
            raise ValueError('checkpoint evidence was not verified')
        if before.policy != after.policy or before.tokenizer != after.tokenizer or before.weights_digest == after.weights_digest:
            raise ValueError('update does not bind distinct same-policy weights')
        if metrics is None: raise ValueError('worker update result is required')
        path = self._path(run_id)
        row = {'run_id':run_id, 'request_digest':digest(asdict(request)), 'before':asdict(before),
            'after':asdict(after), 'before_artifact':self.checkpoint_store.artifact_digests[before],
            'after_artifact':self.checkpoint_store.artifact_digests[after]}
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(row, stream, sort_keys=True)
        return str(path.resolve())
    def verify_update(self, update):
        if not isinstance(update, CompletedUpdate): return False
        try:
            path = self._path(update.run_id)
            if Path(update.evidence_ref).resolve() != path.resolve(): return False
            row = json.loads(path.read_text(encoding='utf-8'))
            before, after = update.request.policy_before, update.policy_after
            return (row['run_id'] == update.run_id and row['request_digest'] == digest(asdict(update.request))
                and row['before'] == asdict(before) and row['after'] == asdict(after)
                and row['before_artifact'] == self.checkpoint_store.artifact_digests[before]
                and row['after_artifact'] == self.checkpoint_store.artifact_digests[after]
                and self.checkpoint_store.verify(before) and self.checkpoint_store.verify(after))
        except (OSError, ValueError, TypeError, KeyError):
            return False
