"""Privileged offline software fixtures, never described as actual LLM results."""
import json
from usami.service.models import ModelSettings
from .models import PARAMETERS
from .rules import decide


def completion(text=None,operations=(),index=0):
    calls=[{'id':f'offline_{index}_{n}','type':'function','function':{'name':o.name,'arguments':json.dumps(o.model_dump(mode='json',exclude={'name'}))}} for n,o in enumerate(operations)]
    message={'role':'assistant','content':text}
    if calls:message['tool_calls']=calls
    return {'choices':[{'finish_reason':'tool_calls' if calls else 'stop','message':message}]}


class ProofClient:
    def __init__(self,settings,task):self.settings=settings;self.task=task;self.index=0
    def complete(self,request):
        self.index+=1
        if self.task.kind=='policy_refusal':return completion(self.task.refusal_code+': request declined by offline policy certificate')
        path=self.task.witnesses[0].operations
        return completion(operations=path,index=self.index) if self.index==1 else completion('Offline proof completed')


class OneStepClient:
    def __init__(self,settings,world,task):self.settings=settings;self.world=world;self.task=task;self.index=0
    def complete(self,request):
        self.index+=1
        if self.task.kind=='policy_refusal':return completion(self.task.refusal_code+': offline refusal')
        if self.index==1:
            before=sum(p.satisfied(self.world) for p in self.task.goal.predicates)
            for op in self.task.capabilities:
                d=decide(self.world,self.task.identity,op)
                if d.allowed and sum(p.satisfied(d.expected_state) for p in self.task.goal.predicates)>before:return completion(operations=[op],index=self.index)
        return completion('Offline one-step baseline finished')


class StopCustomer:
    def __init__(self,settings):self.settings=settings
    def complete(self,request):return completion(json.dumps({'type':'stop','reason':'Offline protocol fixture; no discovery capability claim'}))
