"""Replay scoring against an already initialized VERL 0.9.1 actor worker."""
from rift.schemas import Tokens
from rift.verl.version import require_verl_version


def padded_batch(batch, pad_token_id):
    if not batch or any(not isinstance(row, Tokens) for row in batch):
        raise ValueError('nonempty token batch required')
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError('valid padding token required')
    p = max(len(row.prompt) for row in batch)
    r = max(len(row.response) for row in batch)
    result = {k: [] for k in ('prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids', 'response_mask')}
    for row in batch:
        prompt = [pad_token_id] * (p - len(row.prompt)) + list(row.prompt)
        response = list(row.response) + [pad_token_id] * (r - len(row.response))
        attention = [0] * (p - len(row.prompt)) + [1] * (len(row.prompt) + len(row.response)) + [0] * (r - len(row.response))
        position, n = [], 0
        for enabled in attention:
            position.append(n if enabled else 0)
            n += enabled
        result['prompts'].append(prompt); result['responses'].append(response)
        result['input_ids'].append(prompt + response); result['attention_mask'].append(attention)
        result['position_ids'].append(position)
        result['response_mask'].append([int(x) for x in row.mask] + [0] * (r - len(row.response)))
    return result


class VerlPolicyScorer:
    """Uses FSDP actor's load_checkpoint and compute_log_prob, never creates workers."""
    def __init__(self, worker_group, checkpoint_store, *, pad_token_id):
        self.worker_group = worker_group
        self.checkpoint_store = checkpoint_store
        self.pad_token_id = pad_token_id

    def score(self, checkpoint, batch):
        if not self.checkpoint_store.verify(checkpoint):
            raise ValueError('checkpoint integrity verification failed')
        require_verl_version()
        data = padded_batch(batch, self.pad_token_id)
        # Heavy libraries are loaded only during an explicit scoring call.
        import torch
        from verl import DataProto
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding
        tensors = {key: torch.tensor(value, dtype=torch.long) for key, value in data.items()}
        proto = DataProto.from_dict(tensors=tensors)
        self.worker_group.load_checkpoint(self.checkpoint_store.resolve(checkpoint))
        packed = left_right_2_no_padding(proto.to_tensordict())
        tu.assign_non_tensor(packed, calculate_entropy=False, compute_loss=False, temperature=1.0)
        result = self.worker_group.compute_log_prob(packed)
        log_probs = no_padding_2_padding(tu.get(result, 'log_probs'), packed).float().cpu().tolist()
        if len(log_probs) != len(batch): raise ValueError('scorer returned an incorrect batch size')
        return [row[:len(item.response)] for row, item in zip(log_probs, batch)]
