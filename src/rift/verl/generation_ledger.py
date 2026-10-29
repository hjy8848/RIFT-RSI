"""Host-owned generation receipts, captured at the inference call boundary.

Files are trusted host evidence, not cryptographic proof against a malicious
host. The serving verifier must independently bind vLLM's loaded weights to the
checkpoint; no model or remote service is started here.
"""
from dataclasses import asdict
import json
from pathlib import Path
from rift.training.trace import digest

class GenerationLedger:
    def __init__(self, root, serving_verifier):
        self.root = Path(root)
        self.serving_verifier = serving_verifier
    def verify_serving(self, checkpoint):
        return self.serving_verifier.verify_serving(checkpoint) is True
    def verify_generation(self, checkpoint, serving_fields):
        verify = getattr(self.serving_verifier, 'verify_generation', None)
        return callable(verify) and verify(checkpoint, serving_fields) is True
    def _path(self, generation_id):
        if not generation_id or any(c not in '0123456789abcdef' for c in generation_id):
            raise ValueError('invalid generation identity')
        return self.root / (generation_id + '.json')
    def record_generation(self, generation_id, checkpoint, rollout_id, group_id, prompt_ids,
                          sampling_params, output_ids, logprobs, serving_fields):
        if (not rollout_id or not self.verify_serving(checkpoint)
                or not self.verify_generation(checkpoint, serving_fields)):
            raise ValueError('serving checkpoint identity was not verified')
        record = {'generation_id': generation_id, 'checkpoint': asdict(checkpoint),
            'rollout_id': rollout_id, 'group_id': group_id, 'prompt_ids': list(prompt_ids),
            'sampling_params': dict(sampling_params), 'generated_token_ids': list(output_ids),
            'log_probs': None if logprobs is None else list(logprobs),
            'serving_fields': dict(serving_fields)}
        # Validate JSON (including nonfinite values) before creating any file.
        payload = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._path(generation_id).open('x', encoding='utf-8') as stream:
            stream.write(payload)
    def verify_capture(self, trace, request, response):
        try:
            generation_id = request['generation_id']
            record = json.loads(self._path(generation_id).read_text(encoding='utf-8'))
            response_logs = response.get('generated_log_probs', response.get('log_probs'))
            return (record['generation_id'] == generation_id and record['checkpoint'] == asdict(trace.policy)
                and record['rollout_id'] == request['rollout_id'] and record['group_id'] == request.get('group_id')
                and record['prompt_ids'] == request['prompt_ids'] == list(trace.tokens.prompt)
                and record['sampling_params'] == request['sampling_params']
                and record['generated_token_ids'] == response['generated_token_ids'] == list(trace.tokens.response)
                and record['log_probs'] == response_logs
                and record['serving_fields'] == response['serving_fields']
                and self.verify_generation(trace.policy, record['serving_fields'])
                and trace.request_digest == digest(request) and trace.response_digest == digest(response))
        except (OSError, ValueError, KeyError, TypeError):
            return False
