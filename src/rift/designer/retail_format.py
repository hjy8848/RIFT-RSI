"""Constrained Designer format for compact retail scenes."""
from typing import Annotated, List, Literal, Union

from pydantic import Field, ValidationError

from .format import PlanError, _Strict, extract_json


class PlanItem(_Strict):
    ref: str = Field(min_length=1)
    product: str = Field(pattern=r"^[a-z]+(?: [a-z]+){0,2}$")
    price_cents: int = Field(ge=100, le=1_000_000, strict=True)


class PlanOrder(_Strict):
    ref: str = Field(min_length=1)
    status: Literal["unshipped", "shipped", "delivered"]
    pays_with: Literal["gift_card", "credit_card", "paypal"]
    items: List[PlanItem] = Field(min_length=1, max_length=3)


class PlanVariant(_Strict):
    ref: str = Field(min_length=1)
    of: str = Field(min_length=1)
    price_cents: int = Field(ge=100, le=1_000_000, strict=True)
    in_stock: bool = Field(strict=True)


class PlanChange(_Strict):
    item: str
    want: List[str] = Field(min_length=1)


class PlanOrderAddress(_Strict):
    type: Literal["set_order_address"]
    order: str
    to: str


class PlanDefaultAddress(_Strict):
    type: Literal["set_default_address"]
    to: str


class PlanChangeItems(_Strict):
    type: Literal["change_items"]
    order: str
    changes: List[PlanChange] = Field(min_length=1)
    pay_with: List[Literal["original", "gift_card", "credit_card", "paypal"]] = Field(default_factory=lambda: ["original"], min_length=1)
    otherwise: Literal["nothing", "cancel_order"] = "nothing"


PlanRequest = Annotated[Union[PlanOrderAddress, PlanDefaultAddress, PlanChangeItems], Field(discriminator="type")]


class PlanTask(_Strict):
    requests: List[PlanRequest] = Field(min_length=1, max_length=3)
    ask_total: bool = False


class RetailPlan(_Strict):
    hypothesis: str = Field(min_length=1)
    gift_card_balance_cents: int = Field(ge=0, le=1_000_000, strict=True)
    orders: List[PlanOrder] = Field(min_length=1, max_length=4)
    variants: List[PlanVariant] = Field(default_factory=list, max_length=8)
    tasks: List[PlanTask] = Field(min_length=1, max_length=3)


def retail_plan_schema():
    return RetailPlan.model_json_schema()


def parse_retail_plan(value):
    try:
        return RetailPlan.model_validate(extract_json(value) if isinstance(value, str) else value)
    except ValidationError as exc:
        raise PlanError("SCHEMA", str(exc)) from exc


def retail_plan_scene(plan):
    from usami_v2.scenarios.retail_scene import RetailScene, SceneError, SceneTask, validate_scene
    if not isinstance(plan, RetailPlan):
        plan = parse_retail_plan(plan)
    value = plan.model_dump()
    scene = RetailScene.model_validate({key: value[key] for key in ("gift_card_balance_cents", "orders", "variants")})
    tasks = [SceneTask.model_validate(task) for task in value["tasks"]]
    try:
        validate_scene(scene, tasks)
    except SceneError as exc:
        raise PlanError(exc.code, str(exc)) from exc
    return scene, tasks


def plan_from_scene(scene, scene_tasks, hypothesis):
    from usami_v2.scenarios.retail_scene import validate_scene
    validate_scene(scene, scene_tasks)
    def without_hashes(value):
        if isinstance(value, dict):
            return {k: without_hashes(v) for k, v in value.items() if k != "record_hash"}
        if isinstance(value, list):
            return [without_hashes(v) for v in value]
        return value
    return RetailPlan.model_validate(without_hashes({**scene.model_dump(mode="json"), "hypothesis": hypothesis,
                                      "tasks": [task.model_dump(mode="json", exclude_defaults=True)
                                                for task in scene_tasks]}))
