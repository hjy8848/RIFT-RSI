"""Constrained sampling of relational worlds; no hidden wall clock or task labels."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from pydantic import Field, model_validator

from usami.randomness import StableRandom

from usami.world.models import (GENERATOR_VERSION, Invoice, Membership, Model, Organization,
                               Refund, RulesConfig, Subscription, User, WorldState, digest)
from usami.world.rules import check_invariants


class GeneratorConfig(Model):
    version: str = GENERATOR_VERSION
    user_count: int = Field(default=8, ge=3, le=1000)
    organization_count: int = Field(default=3, ge=2, le=100)
    clock: datetime = datetime(2026, 1, 15, 12, tzinfo=timezone.utc)
    profile: Literal["standard", "train_single", "test_composed"] = "standard"
    rules: RulesConfig = Field(default_factory=RulesConfig)
    workflow: Optional[str] = Field(default=None, exclude_if=lambda v: v is None)

    @model_validator(mode="after")
    def usable(self) -> GeneratorConfig:
        if self.clock.tzinfo is None:
            raise ValueError("generator clock must include timezone")
        if not self.rules.plans or any(k != p.id for k, p in self.rules.plans.items()):
            raise ValueError("at least one consistent plan is required")
        if self.profile == "test_composed" and self.user_count < 3:
            raise ValueError("composition requires three users")
        if self.version not in {GENERATOR_VERSION, "saas-generator-0.2.0"}:
            raise ValueError("unsupported generator version")
        if (self.version.endswith("0.2.0")) != (self.rules.version == "saas-rules-0.2.0"):
            raise ValueError("generator and rules versions must agree")
        if self.workflow is not None and self.version == GENERATOR_VERSION:
            raise ValueError("workflow conditioning requires v0.2")
        return self


def composition_signature(world: WorldState) -> bool:
    """Held-out combination: role boundary + cycle state + invoice eligibility."""
    for a in world.memberships.values():
        if a.status != "active" or a.role != "billing_admin":
            continue
        sa = world.subscriptions[world.organizations[a.org_id].subscription_id]
        recent = any(i.org_id == a.org_id and i.status == "paid" and
                     world.clock - i.paid_at <= timedelta(days=world.rules.refund_window_days)
                     for i in world.invoices.values())
        for b in world.memberships.values():
            if b.user_id != a.user_id or b.org_id == a.org_id or b.role != "member" or b.status != "active":
                continue
            sb = world.subscriptions[world.organizations[b.org_id].subscription_id]
            stale = any(i.org_id == b.org_id and i.status == "paid" and
                        world.clock - i.paid_at > timedelta(days=world.rules.refund_window_days)
                        for i in world.invoices.values())
            if sa.status == "active" and sb.status == "expired" and recent and stale:
                return True
    return False


class WorldGenerator:
    def __init__(self, seed: int, config: Optional[GeneratorConfig] = None):
        self.seed = seed
        self.config = (config or GeneratorConfig()).model_copy(deep=True)

    def generate(self) -> WorldState:
        if self.config.version == "saas-generator-0.2.0":
            from usami.world.generator_v2 import generate_v2
            return generate_v2(self.seed, self.config)
        cfg, rng = self.config, StableRandom(self.seed)
        now = cfg.clock.astimezone(timezone.utc)
        users = {f"user_{n:03d}": User(id=f"user_{n:03d}", name=f"User {n}",
                                      email=f"user{n}@usami.example") for n in range(cfg.user_count)}
        user_ids = list(users)
        orgs, members, subs, invoices = {}, {}, {}, {}
        homogeneous_status = rng.choice(["active", "expired"])
        for n in range(cfg.organization_count):
            oid, sid = f"org_{n:03d}", f"sub_{n:03d}"
            plan = rng.choice([cfg.rules.plans[k] for k in sorted(cfg.rules.plans)])
            owner = user_ids[n % cfg.user_count]
            if cfg.profile == "test_composed" and n < 2:
                owner = user_ids[n]
            eligible = [uid for uid in user_ids if uid != owner]
            count = rng.randint(1, min(len(eligible) + 1, plan.max_seats, 8))
            selected = [owner] + rng.sample(eligible, count - 1)
            if cfg.profile == "test_composed" and n < 2:
                # A shared principal has two different active roles; reserve its seat.
                if plan.max_seats < 2:
                    plan = next((cfg.rules.plans[k] for k in sorted(cfg.rules.plans) if cfg.rules.plans[k].max_seats >= 2), None)
                    if plan is None:
                        raise ValueError("test_composed needs a plan with at least two seats")
                selected = [owner, user_ids[2]] + [u for u in selected if u not in {owner, user_ids[2]}]
                selected = selected[:plan.max_seats]
            orgs[oid] = Organization(id=oid, name=f"Workspace {n}", owner_id=owner, subscription_id=sid)
            for j, uid in enumerate(selected):
                mid = f"membership_{n:03d}_{uid}"
                role = "owner" if j == 0 else rng.choice(["billing_admin", "member"])
                status = "active" if j == 0 or rng.random() < 0.85 else "inactive"
                if cfg.profile == "test_composed" and n < 2 and uid == user_ids[2]:
                    role, status = ("billing_admin" if n == 0 else "member"), "active"
                members[mid] = Membership(id=mid, org_id=oid, user_id=uid, role=role, status=status)
            active = sum(m.org_id == oid and m.status == "active" for m in members.values())
            status = rng.choice(["active", "active", "expired"])
            if cfg.profile == "train_single":
                status = homogeneous_status
            if cfg.profile == "test_composed" and n < 2:
                status = "active" if n == 0 else "expired"
            end = now + timedelta(days=rng.randint(1, 25)) if status == "active" else now - timedelta(days=rng.randint(0, 30))
            subs[sid] = Subscription(id=sid, org_id=oid, plan_id=plan.id,
                                     seats=rng.randint(active, plan.max_seats),
                                     auto_renew=rng.choice([True, False]) if status == "active" else False,
                                     period_start=end - timedelta(days=30), period_end=end, status=status)
            # Sample coherent payment histories, covering the four eligibility classes.
            window_seconds = cfg.rules.refund_window_days * 86400
            for kind in ("recent", "stale", "unpaid", "refunded"):
                iid = f"invoice_{n:03d}_{kind}"
                amount = rng.randint(10, 500) * 100
                method = rng.choice(["card_visa", "card_mastercard", "bank_transfer"])
                age = rng.randint(0, window_seconds) if kind in {"recent", "refunded"} else window_seconds + rng.randint(1, 50 * 86400)
                paid = now - timedelta(seconds=age)
                inv = Invoice(id=iid, org_id=oid, amount_cents=amount, refundable_amount_cents=amount,
                              status="unpaid" if kind == "unpaid" else "refunded" if kind == "refunded" else "paid",
                              paid_at=None if kind == "unpaid" else paid,
                              payment_method=None if kind == "unpaid" else method)
                if kind == "refunded":
                    inv.refunds.append(Refund(id=f"historical_{iid}", amount_cents=amount,
                                              payment_method=method, refunded_at=paid + timedelta(seconds=age // 2)))
                invoices[iid] = inv
        config_hash = digest(cfg)
        world = WorldState(world_id="world_" + digest({"seed": self.seed, "config": config_hash})[:20],
                           seed=self.seed, config_hash=config_hash, clock=now, rules=cfg.rules.model_copy(deep=True),
                           users=users, organizations=orgs, memberships=members, subscriptions=subs, invoices=invoices)
        problems = check_invariants(world)
        if problems:
            raise ValueError(f"generator produced invalid state: {problems}")
        if cfg.profile == "train_single" and composition_signature(world):
            raise ValueError("training world violates held-out composition")
        if cfg.profile == "test_composed" and not composition_signature(world):
            raise ValueError("test world lacks held-out composition")
        return world


def generate_world(seed: int = 42, config: Optional[GeneratorConfig] = None) -> WorldState:
    return WorldGenerator(seed, config).generate()
