"""Frozen multi-profile scope helpers for versioned composed dialogue tasks."""

from .models import CustomerIntent


def load_profiles(adapter, world, primary_id, profile_ids=()):
    ids=list(profile_ids) if profile_ids else [primary_id]
    if not ids or ids[0]!=primary_id or len(ids)!=len(set(ids)):
        raise ValueError('composed Customer scope must be unique and begin with the primary profile')
    rows=[]
    for profile_id in ids:
        if profile_id not in world.profiles:
            raise ValueError('composed Customer profile is absent from the frozen world')
        private=adapter.private_state(world,profile_id)
        adapter.validate_private(world,private)
        rows.append((profile_id,private))
    if any(row.identity!=rows[0][1].identity for _,row in rows):
        raise ValueError('composed Customer profiles must have the same authenticated identity')
    return rows


def profile_for_intent(adapter, world, profiles, intent):
    """Resolve one exact frozen profile for a Customer turn; never merge facts."""
    intent=CustomerIntent.model_validate(intent.model_dump(mode='json') if isinstance(intent,CustomerIntent) else intent)
    declarations=list(intent.requested_operations)+([intent.confirm_operation] if intent.confirm_operation is not None else [])
    if declarations:
        matches=[]
        for profile_id,private in profiles:
            try:
                if all(adapter.intent_allowed(world,private,raw) for raw in declarations):matches.append((profile_id,private))
            except (KeyError,TypeError,ValueError):
                continue
        if not matches:
            raise ValueError('Customer writes do not belong to one selected frozen profile')
        return matches[0]
    if not intent.disclosures and not intent.preferences:
        return profiles[0]
    matches=[]
    for profile_id,private in profiles:
        facts_ok=all(k in private.facts and private.facts[k]==v for k,v in intent.disclosures.items())
        preferences_ok=all(k in private.preference_space and v in private.preference_space[k] for k,v in intent.preferences.items())
        if facts_ok and preferences_ok:matches.append((profile_id,private))
    if not matches:
        raise ValueError('Customer facts or preferences do not belong to one selected frozen profile')
    return matches[0]


def customer_candidates(adapter, world, profiles):
    """Union independently authorized native Customer choices across scope."""
    from usami_harness.models import digest
    rows={}
    for profile_id,private in profiles:
        for raw in adapter.candidate_intents(world,private):
            if adapter.intent_allowed(world,private,raw):rows[digest(raw)]=raw
    return [rows[key] for key in sorted(rows)]
