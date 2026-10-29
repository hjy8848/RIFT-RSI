"""Actual airline simulator mutations, independent of the Reference implementation."""
from datetime import timedelta
from usami_harness.models import Result,digest,signed_notice
from .models import Quote,Verification,LedgerEntry


def _mutate(world,actor,op,event_index=0):
    if op.is_query:
        from .observations import query_data
        return world.clone(),query_data(world,actor,op)
    new=world.clone();b=new.bookings[op.resource_id];p=op.params;ctx=new.interaction;rule=new.fare_rules[b.fare_rule_id]
    if op.name=='verify_passenger':
        user=new.users[actor]
        if user.document_token!=p['document_token'] or not user.simulated_identity_valid:raise ValueError('passenger identity verification failed in the simulator')
        key='verification_'+b.id;payload={'booking_id':b.id,'passenger_id':actor,'verified':True}
        new.verifications[key]=Verification(id=key,booking_id=b.id,verified=True,notice=signed_notice(new.domain,'passenger-identity',payload,new.clock))
        return new,{'booking_id':b.id,'verified':True}
    if op.name=='commit_quote':
        q=new.quotes[p['quote_id']];b.revision+=1;q.status='committed'
        if q.action=='rebook':
            b.flights=list(q.flights);b.cabin=q.cabin;b.fare_value_cents=q.fare_value_cents;b.retained_fees_cents+=q.fee_cents
        elif q.action=='refund':
            b.status='cancelled';b.fare_value_cents=0;b.retained_fees_cents+=b.baggage_paid_cents+q.fee_cents;b.baggage_paid_cents=0;b.extra_baggage_kg=0
        else:b.extra_baggage_kg=q.baggage_kg;b.baggage_paid_cents+=q.charge_cents
        if q.charge_cents:
            key='charge_'+q.id;binding={'booking_id':b.id,'quote_id':q.id,'kind':'charge','amount_cents':q.charge_cents}
            new.ledger[key]=LedgerEntry(id=key,booking_id=b.id,kind='charge',amount_cents=q.charge_cents,quote_id=q.id,notice=signed_notice(new.domain,'airline-payment',binding,new.clock))
        if q.refund_cents:
            key='refund_'+q.id;binding={'booking_id':b.id,'quote_id':q.id,'kind':'refund','amount_cents':q.refund_cents}
            new.ledger[key]=LedgerEntry(id=key,booking_id=b.id,kind='refund',amount_cents=q.refund_cents,quote_id=q.id,notice=signed_notice(new.domain,'airline-payment',binding,new.clock))
        return new,{'booking_id':b.id,'status':b.status,'flights':b.flights,'charge_cents':q.charge_cents,'refund_cents':q.refund_cents}
    cancelled=any(new.flights[f].status=='cancelled' for f in b.flights)
    charge=refund=fee=0;flights=list(b.flights);cabin=b.cabin;baggage=b.extra_baggage_kg;value=b.fare_value_cents
    if op.name=='quote_change':
        action='rebook';flights=list(p['flights']);cabin=p['cabin'];value=sum(new.flights[f].fares[cabin] for f in flights)
        fee=0 if cancelled else rule.change_fee_cents;charge=max(value-b.fare_value_cents,0)+fee;refund=max(b.fare_value_cents-value,0)
    elif op.name=='quote_refund':
        action='refund';fee=0 if cancelled else min(rule.refund_fee_cents,b.fare_value_cents);refund=b.fare_value_cents-fee;value=0;baggage=0
    else:
        action='baggage';baggage=p['extra_kg'];charge=max(0,baggage-b.extra_baggage_kg)*rule.baggage_price_per_kg_cents
    key='quote_'+digest([b.id,b.revision,action,flights,cabin,baggage,ctx.revision,new.clock.isoformat()])[:20]
    if key not in new.quotes:
        q=Quote(id=key,booking_id=b.id,action=action,flights=flights,cabin=cabin,charge_cents=charge,refund_cents=refund,fee_cents=fee,fare_value_cents=value,baggage_kg=baggage,
            revision=ctx.revision,booking_revision=b.revision,expires_at=new.clock+timedelta(minutes=30),quote_hash='')
        q.quote_hash=digest(q.model_dump(mode='json',exclude={'quote_hash','status'}));new.quotes[key]=q
    return new,{'quote':new.quotes[key].model_dump(mode='json')}


def apply(world,actor,op,event_index=0):
    try:
        new,data=_mutate(world,actor,op,event_index)
        return new,Result(success=True,data=data)
    except (KeyError,ValueError,TypeError,AttributeError):
        return world.clone(),Result(success=False,error_code='MUTATION_NOT_REPRESENTABLE')
