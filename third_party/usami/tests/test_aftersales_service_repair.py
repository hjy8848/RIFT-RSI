from experiments.vast_coevo.examples.aftersales_service_repair import _json


class _NestedModel:
    def model_dump(self, mode="python"):
        return {"mode": mode, "value": 7}


def test_json_conversion_recurses_into_nested_models_and_tuples():
    value = {"config": _NestedModel(), "contexts": (_NestedModel(),)}

    assert _json(value) == {
        "config": {"mode": "json", "value": 7},
        "contexts": [{"mode": "json", "value": 7}],
    }
