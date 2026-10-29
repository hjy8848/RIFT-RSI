"""Pre-registered operational disruption, not a model-invented cancellation."""
from datetime import timedelta


def expected(world,event_id):
    e=world.external_events.get(event_id)
    if not e or event_id in world.applied_events:raise ValueError('unknown or consumed airline event')
    new=world.clone();f=new.flights[e.flight_id];new.clock=max(world.clock,e.at);f.status=e.status
    if e.status=='delayed':f.departure+=timedelta(minutes=e.delay_minutes);f.arrival+=timedelta(minutes=e.delay_minutes)
    new.applied_events.append(e.id);return new


def apply(world,event_id):
    if event_id not in world.external_events or event_id in world.applied_events:raise ValueError('unknown or consumed airline event')
    new=world.clone();update=new.external_events[event_id];target=new.flights[update.flight_id]
    target.status=update.status
    if update.status=='delayed':
        delta=timedelta(minutes=update.delay_minutes);target.arrival=target.arrival+delta;target.departure=target.departure+delta
    new.clock=max(new.clock,update.at);new.applied_events.append(event_id);return new
