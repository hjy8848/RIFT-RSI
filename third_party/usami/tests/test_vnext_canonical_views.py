"""Canonical JSON archives must preserve both query data and rendered choices."""
import importlib,json
import pytest
from usami_benchmark.registry import default_spec,generate


@pytest.mark.parametrize('domain',['aftersales','airline','property'])
def test_native_views_and_candidate_order_survive_sorted_json(domain):
    adapter=importlib.import_module('usami_'+domain+'.adapter');w=generate(default_spec(domain,42))
    restored=adapter.World.model_validate(json.loads(json.dumps(w.model_dump(mode='json'),sort_keys=True)))
    profile_id=next(iter(w.profiles));p=adapter.private_state(w,profile_id);q=adapter.private_state(restored,profile_id)
    assert adapter.observation(w,p.identity)==adapter.observation(restored,p.identity)
    assert adapter.candidate_intents(w,p)==adapter.candidate_intents(restored,q)
    if domain=='aftersales':
        order=w.orders[p.facts['order_id']];scope=order.merchant_id;resource=order.id;probes=[('get_order',{})]
    elif domain=='airline':
        b=w.bookings[p.facts['booking_id']];scope=b.carrier_id;resource=b.id
        probes=[('get_booking',{}),('search_flights',{'origin':'A','destination':'B'})]
    else:
        unit=w.units[p.facts['unit_id']];scope=unit.property_id;resource=unit.id;probes=[('get_unit',{}),('get_technicians',{})]
    for name,params in probes:
        op=adapter.Operation(name=name,scope_id=scope,resource_id=resource,params=params)
        left=adapter.decide(w,p.identity,op);right=adapter.decide(restored,p.identity,op)
        assert left.allowed and right.allowed and left.data==right.data
