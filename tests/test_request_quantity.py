import sys
import tempfile
import unittest

sys.path[:0] = ['src', 'third_party/usami']

from rift.runtime import curriculum_run as run


def _request(kit, family, knobs, seed):
    refs = kit.certify_specs([{'family': family, 'knobs': knobs, 'instances': 1, 'seed': seed}])
    ref = next(iter(refs.values()))[0]
    return kit.task_registry.resolve_record(ref).task.customer_request


class RequestQuantityTests(unittest.TestCase):
    def test_a_partial_refund_of_a_multi_unit_line_states_the_quantity(self):
        """F2 default buys 3 and wants 1 back; without the count 16/16 refunded all 3."""
        with tempfile.TemporaryDirectory() as root:
            text = _request(run.WorldKit(root), 'F2', {}, 4100)
        self.assertIn('for 1 item', text)

    def test_larger_quantities_still_read_as_before(self):
        with tempfile.TemporaryDirectory() as root:
            text = _request(run.WorldKit(root), 'F2', {'purchased_quantity': 5, 'refund_quantity': 2}, 4200)
        self.assertIn('for 2 items', text)

    def test_a_single_unit_line_keeps_its_original_wording(self):
        with tempfile.TemporaryDirectory() as root:
            text = _request(run.WorldKit(root), 'F1', {}, 4300)
        self.assertNotIn(' for 1 item', text)
        self.assertNotIn(' items for', text)


if __name__ == '__main__':
    unittest.main()
