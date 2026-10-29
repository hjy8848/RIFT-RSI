import pytest

from usami import GeneratorConfig, generate_world
from usami.world.models import Membership


@pytest.fixture
def world():
    world = generate_world(42, GeneratorConfig(user_count=4, organization_count=2, profile="test_composed"))
    # Small lawful fixture, separate from the procedural generator's coverage tests.
    mid = "membership_000_user_003"
    world.memberships[mid] = Membership(id=mid, org_id="org_000", user_id="user_003", role="member")
    sub = world.subscriptions["sub_000"]
    sub.plan_id = "starter"
    sub.seats = sum(m.org_id == "org_000" and m.status == "active" for m in world.memberships.values())
    sub.auto_renew = True
    return world
