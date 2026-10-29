"""Actual enrollment/credit mutations, independent of Reference policy."""
from datetime import timedelta
from usami_harness.models import Result,digest,signed_notice
from .models import Case,Quote,Appointment,Ledger


def _mutate(w,actor,op):
    new=w.clone();p=op.params
    if op.is_query:
        from .observations import query_data
        return new,query_data(w,actor,op)
    e=new.enrollments[op.resource_id];ctx=new.interaction;case=new.cases.get(p.get('case_id',''))
    if op.name=='open_request':
        key='case_'+digest([e.id,p['request_id']])[:20]
        if key not in new.cases:new.cases[key]=Case(id=key,enrollment_id=e.id,request_id=p['request_id'],requester_id=actor,reason=p['reason'],created_at=new.clock)
        return new,{'case_id':key,'reason':new.cases[key].reason}
    if op.name=='request_extension_review':
        if case.pending_extension_days!=p['days'] or case.review_version!=e.version:case.review_notice={}
        case.pending_extension_days=p['days'];case.review_version=e.version
        return new,{'case_id':case.id,'status':'pending'}
    if op.name=='cancel_lesson':
        a=new.appointments[p['appointment_id']];a.status='cancelled';e.version+=1
        case=new.cases[a.case_id]
        if case.last_appointment_id==a.id:case.last_lesson_status='cancelled'
        return new,{'appointment_id':a.id,'status':'cancelled'}
    if op.name=='commit_quote':
        q=new.quotes[p['quote_id']];case=new.cases[q.case_id];q.status='committed';e.version+=1
        e.remaining_credit_cents=q.resulting_credit_cents;e.retained_fees_cents+=q.fee_cents;e.expires_at=q.resulting_expiry
        if q.action=='refund':e.status='refunded'
        elif q.action=='transfer':e.class_id=q.target_class_id;e.course_id=new.classes[q.target_class_id].course_id
        elif q.action=='extend':e.extended_days+=q.extension_days
        else:
            key='lesson_'+q.id;price=new.courses[e.course_id].unit_price_cents
            new.appointments[key]=Appointment(id=key,enrollment_id=e.id,case_id=case.id,quote_id=q.id,course_id=e.course_id,slot_id=q.slot_id,unit_price_cents=price);case.last_slot_id=q.slot_id;case.last_appointment_id=key;case.last_lesson_status='scheduled'
        for kind,amount in [('charge',q.charge_cents),('refund',q.refund_cents)]:
            if amount>0:
                key=kind+'_'+q.id;payload={'enrollment_id':e.id,'case_id':case.id,'quote_id':q.id,'kind':kind,'amount_cents':amount,'channel':p['payment_channel']}
                new.ledger[key]=Ledger(id=key,enrollment_id=e.id,case_id=case.id,quote_id=q.id,kind=kind,amount_cents=amount,channel=p['payment_channel'],notice=signed_notice(new.domain,'education-payment',payload,new.clock))
        return new,{'enrollment_id':e.id,'action':q.action,'status':e.status,'remaining_credit_cents':e.remaining_credit_cents,'charge_cents':q.charge_cents,'refund_cents':q.refund_cents}
    contract=new.contracts[e.contract_id];action={'quote_refund':'refund','quote_transfer':'transfer','quote_extension':'extend','quote_lesson':'tutoring'}[op.name]
    fee=charge=refund=days=topup=0;target=slot=None;credit=e.remaining_credit_cents;expiry=e.expires_at
    if action=='refund':fee=min(credit,contract.refund_fee_cents);refund=credit-fee;credit=0
    elif action=='transfer':
        target=p['target_class_id'];topup=p['topup_units'];course=new.courses[new.classes[target].course_id]
        fee=contract.transfer_fee_cents;charge=fee+topup*course.unit_price_cents;credit+=topup*course.unit_price_cents
    elif action=='extend':days=p['days'];fee=days*contract.extension_fee_per_day_cents;charge=fee;expiry=e.expires_at+timedelta(days=days)
    else:slot=p['slot_id']
    params={'target_class_id':target,'slot_id':slot,'extension_days':days,'topup_units':topup}
    key='quote_'+digest([e.id,case.id,action,params,e.version,ctx.revision,new.clock.isoformat()])[:20]
    if key not in new.quotes:
        q=Quote(id=key,enrollment_id=e.id,case_id=case.id,action=action,**params,charge_cents=charge,refund_cents=refund,fee_cents=fee,resulting_credit_cents=credit,resulting_expiry=expiry,
            revision=ctx.revision,enrollment_version=e.version,expires_at=new.clock+timedelta(hours=1),quote_hash='')
        q.quote_hash=digest(q.model_dump(mode='json',exclude={'quote_hash','status'}));new.quotes[key]=q
    return new,{'quote':new.quotes[key].model_dump(mode='json')}


def apply(world,actor,op,event_index=0):
    try:
        new,data=_mutate(world,actor,op);return new,Result(success=True,data=data)
    except (KeyError,ValueError,TypeError,AttributeError):return world.clone(),Result(success=False,error_code='MUTATION_NOT_REPRESENTABLE')
