import json
import sys

import pytest

sys.path[:0] = ['src', 'third_party/usami']

from rift.verl.sampling import LOGPROB_FLOOR, finite_logprobs


def test_negative_infinity_becomes_a_finite_floor_that_json_accepts():
    values = finite_logprobs([-0.5, float('-inf'), -3.0])
    assert values == [-0.5, LOGPROB_FLOOR, -3.0]
    json.dumps(values, allow_nan=False)      # the generation ledger's exact serialisation


def test_none_and_ordinary_values_pass_through_unchanged():
    assert finite_logprobs(None) is None
    assert finite_logprobs([-1.25, 0.0]) == [-1.25, 0.0]


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nan_and_positive_infinity_are_still_errors(bad):
    with pytest.raises(ValueError):
        finite_logprobs([-1.0, bad])
