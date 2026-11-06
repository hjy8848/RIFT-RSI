"""Retail extension: scene builder support and the Service tool profile."""
import json

import pytest

from usami.service.models import ServiceError
from usami_v2.commerce.invariants import invariants
from usami_v2.runtime.agents import OpenAIServiceAgent, ServiceAction, service_tools
from usami_v2.scenarios.builder import build_world
from usami_v2.scenarios.schema import (
    AddressSpec, Attribute, BuildBudget, CustomerAddress, IntegerRange, LineVariable, OrderVariable,
    PaymentVariable, ScenarioSpec, SKUVariable, WarehouseVariable,
)

HOME = AddressSpec(street="1 Home Rd", city="Denver", state="CO", postal_code="80201")


def fixed(value):
    return IntegerRange(low=value, high=value)


def spec(**updates):
    return ScenarioSpec(
        scenario_id="retail-surface", provenance_kind="offline_fixture", customer_aliases=("me",),
        skus=(SKUVariable(alias="a", family="shoe", price_cents=fixed(5000), attributes=(Attribute(name="c", value="red"),)),
              SKUVariable(alias="b", family="shoe", price_cents=fixed(6000), attributes=(Attribute(name="c", value="blue"),))),
        warehouses=(WarehouseVariable(alias="w", delivery_days=fixed(2), delivery_fee_cents=fixed(0)),),
        orders=(OrderVariable(alias="o", customer_alias="me", delivered_days_ago=fixed(3),
                              fulfillment_status="unshipped",
                              lines=(LineVariable(alias="l", original_sku="a", quantity=fixed(1), paid_cents=fixed(5000)),),
                              **updates.pop("order", {})),),
        default_stock=fixed(3), **updates)


def test_a_scene_without_retail_fields_serializes_exactly_as_before():
    dump = spec().model_dump(mode="json")
    assert not {"payment_methods", "default_addresses"} & set(dump)
    assert not {"payment_alias", "shipping_address"} & set(dump["orders"][0])


def test_scene_builds_payment_methods_addresses_and_a_paying_instrument():
    scene = spec(
        payment_methods=(PaymentVariable(alias="gc", customer_alias="me", kind="gift_card", balance_cents=3000),
                         PaymentVariable(alias="cc", customer_alias="me", kind="credit_card")),
        default_addresses=(CustomerAddress(customer_alias="me", address=HOME),),
        order={"payment_alias": "cc", "shipping_address": HOME})
    result = build_world(scene, 5, BuildBudget(max_nodes=16))
    assert result.status == "SAT" and invariants(result.world) == []
    ids = dict(result.private_aliases)
    world = result.world
    assert world.payment_methods[ids["gc"]].balance_cents == 3000
    assert world.orders[ids["o"]].payment_channel == ids["cc"] == world.orders[ids["o"]].payment_method_id
    assert world.customers[ids["me"]].default_address.city == "Denver"
    assert world.orders[ids["o"]].shipping_address.postal_code == "80201"


def test_scene_rejects_a_gift_card_without_balance_and_a_foreign_paying_instrument():
    with pytest.raises(ValueError):
        PaymentVariable(alias="gc", customer_alias="me", kind="gift_card")
    with pytest.raises(ValueError):
        PaymentVariable(alias="cc", customer_alias="me", kind="credit_card", balance_cents=5)
    with pytest.raises(ValueError):
        spec(order={"payment_alias": "ghost"})


def message(name, arguments):
    return {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call-1", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}}]}


NEW_TOOLS = {"get_customer_profile", "modify_order_address", "update_default_address", "modify_order_items"}


def test_retail_profile_adds_exactly_the_new_tools_and_full9_is_unchanged():
    nine = {row["function"]["name"] for row in service_tools("full9")}
    retail = {row["function"]["name"] for row in service_tools("retail")}
    assert not NEW_TOOLS & nine and retail == nine | NEW_TOOLS
    assert next(row for row in service_tools("full9")
                if row["function"]["name"] == "return_delivered_order_items"
                )["function"]["parameters"]["properties"]["payment_method"]["enum"] == ["original"]
    assert "enum" not in next(row for row in service_tools("retail")
                              if row["function"]["name"] == "return_delivered_order_items"
                              )["function"]["parameters"]["properties"]["payment_method"]
    assert OpenAIServiceAgent.system_for_profile("retail").startswith(OpenAIServiceAgent.SYSTEM)
    assert OpenAIServiceAgent.system_for_profile("retail") != OpenAIServiceAgent.system_for_profile("full9")


def parse(name, arguments, retail=True):
    return ServiceAction.from_tool_message(message(name, arguments), "c", (), ("shop",), retail=retail)


def test_new_tools_parse_into_native_operations_only_under_the_retail_profile():
    address = {"street": "9 Away St", "city": "Austin", "state": "TX", "postal_code": "73301"}
    op = parse("modify_order_address", {"order_id": "o1", **address}).operation
    assert (op.name, op.resource_id, op.params) == ("modify_order_address", "o1", address)
    op = parse("update_default_address", address).operation
    assert (op.name, op.resource_id) == ("update_default_address", "shop")
    assert parse("get_customer_profile", {}).operation.resource_id == "shop"
    op = parse("modify_order_items", {"order_id": "o1", "changes": [{"line_id": "l1", "target_sku_id": "s2"}],
                                       "payment_method": "gc"}).operation
    assert op.params == {"changes": [{"line_id": "l1", "target_sku_id": "s2"}], "payment_method": "gc"}
    for name, args in (("get_customer_profile", {}), ("modify_order_address", {"order_id": "o1", **address})):
        with pytest.raises(ServiceError):
            parse(name, args, retail=False)


def test_payment_method_other_than_original_is_a_retail_feature():
    args = {"line_id": "l1", "quantity": 1, "payment_method": "gc"}
    assert parse("return_delivered_order_items", args).operation.params["payment_method"] == "gc"
    with pytest.raises(ServiceError):
        parse("return_delivered_order_items", args, retail=False)
    exchange = parse("exchange_delivered_order_items", {"line_id": "l1", "quantity": 1, "target_sku_id": "s2"})
    assert exchange.operation.params == {"quantity": 1, "target_sku_id": "s2"}      # default stays out of the record
    with pytest.raises(ServiceError):
        parse("modify_order_items", {"order_id": "o1", "changes": "nope"})
