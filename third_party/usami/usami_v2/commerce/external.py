"""Frozen external events: independent reference projection and actual application."""


def apply_event(world, event_id):
    new = world.clone()
    event = new.external_events[event_id]
    if event_id in new.applied_events:
        raise ValueError("external event already consumed")
    if event.at.tzinfo is None:
        raise ValueError("external event lacks aware business time")
    if event.kind == "review_result":
        case = new.cases[event.target_id]
        if not case.review_requested:
            raise ValueError("review not requested")
        case.review_approved = event.approved
    elif event.kind == "return_received":
        ret = next((row for row in new.returns.values() if row.case_id == event.target_id), None)
        if not ret or ret.status != "scheduled":
            raise ValueError("return not scheduled")
        ret.status, ret.received_event_id = "received", event_id
        new.cases[ret.case_id].status = "ready"
    elif event.kind == "stock_adjustment":
        inv = new.inventory[event.target_id]
        if inv.available + event.integer_value < 0:
            raise ValueError("cannot consume committed inventory")
        inv.available += event.integer_value
        inv.adjustments += event.integer_value
        inv.revision += 1
    elif event.kind == "route_change":
        if event.integer_value < 0:
            raise ValueError("negative delivery delay")
        route = new.routes[event.target_id]
        route.days = event.integer_value
        route.revision += 1
    new.clock = max(new.clock, event.at)
    new.applied_events = new.applied_events + (event_id,)
    return new


def reference_event(world, event_id):
    """Does not call apply_event. Reject unregistered, unready and reused events."""
    if event_id not in world.external_events or event_id in world.applied_events:
        raise ValueError("unregistered or reused external event")
    event = world.external_events[event_id]
    if event.at.tzinfo is None:
        raise ValueError("timezone-aware external schedule required")
    expected = world.clone()
    if event.kind == "review_result":
        if event.target_id not in world.cases or not world.cases[event.target_id].review_requested:
            raise ValueError("no pending review")
        expected.cases[event.target_id].review_approved = event.approved
    elif event.kind == "return_received":
        matching = [row for row in world.returns.values() if row.case_id == event.target_id and row.status == "scheduled"]
        if len(matching) != 1:
            raise ValueError("no unique scheduled return")
        expected.returns[matching[0].id].status = "received"
        expected.returns[matching[0].id].received_event_id = event.id
        expected.cases[event.target_id].status = "ready"
    elif event.kind == "stock_adjustment":
        balance = world.inventory[event.target_id]
        if balance.available + event.integer_value < 0:
            raise ValueError("stock event would consume reserved goods")
        expected.inventory[event.target_id].available = balance.available + event.integer_value
        expected.inventory[event.target_id].adjustments = balance.adjustments + event.integer_value
        expected.inventory[event.target_id].revision = balance.revision + 1
    elif event.kind == "route_change":
        if event.target_id not in world.routes or event.integer_value < 0:
            raise ValueError("invalid route event")
        expected.routes[event.target_id].days = event.integer_value
        expected.routes[event.target_id].revision = world.routes[event.target_id].revision + 1
    expected.applied_events = tuple(list(world.applied_events) + [event.id])
    if event.at > world.clock:
        expected.clock = event.at
    return expected
