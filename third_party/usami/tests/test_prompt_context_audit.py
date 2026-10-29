"""Audit must account for original evidence without modifying production requests."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'experiments/vast_coevo/examples/audit_prompt_context.py'
spec = importlib.util.spec_from_file_location('prompt_context_audit', PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_partition_exact_bytes_and_no_mutation():
    request = {'model':'qwen', 'messages':[
        {'role':'system','content':'identity + exact grants'},
        {'role':'system','content':'<VAST_SKILLS>\n<VAST_SKILL id="service.a">x'},
        {'role':'user','content':'old'}, {'role':'assistant','content':'reply'},
        {'role':'user','content':'current'},
        {'role':'tool','tool_call_id':'a','content':'{"状态": "active"}'}],
        'tools':[{'type':'function'}]}
    original=copy.deepcopy(request)
    row=audit.measure(request)
    assert request == original
    assert sum(row['component_bytes'].values()) == len(audit.encoded(request))
    assert row['skill_ids'] == ['service.a']
    assert row['component_bytes']['conversation_history'] > 0


def test_repeated_result_is_measured_not_removed():
    request={'messages':[{'role':'tool','tool_call_id':str(i),'content':'{"x": 1}'} for i in range(3)]}
    row=audit.measure(request)
    assert row['repeated_tool_payload_bytes'] == 2*len('{"x": 1}'.encode())
    assert len(request['messages']) == 3


@pytest.mark.parametrize('text', ['{"x":1,"x":2}', '{"x":NaN}', 'plain text'])
def test_ambiguous_or_invalid_json_not_compacted(text):
    assert audit.compact_json_text(text) == text


def test_whitespace_candidate_preserves_values():
    value={'identity':'u0','grants':[{'project':'p1','params':{}}], 'success':False}
    text=json.dumps(value,indent=2)
    compact=audit.compact_json_text(text)
    assert json.loads(compact)==value
    assert len(compact)<len(text)


def test_ledger_usage_reasoning_not_double_counted(tmp_path):
    calls=tmp_path/'wire-ledger/calls'; calls.mkdir(parents=True)
    (calls/'a.started.json').write_text(json.dumps({'id':'a','role':'service','request':{'messages':[]}}))
    (calls/'a.completed.json').write_text(json.dumps({'response':{'usage':{'prompt_tokens':10,'completion_tokens':8,'total_tokens':18,'completion_tokens_details':{'reasoning_tokens':5}}}}))
    result=audit.audit(tmp_path,'ledger')
    group=result['groups']['service']
    assert group['total_tokens']==18 and group['output_tokens']==8
    assert result['online_calls']==0
    assert result['runtime_modified'] is False


def test_missing_usage_explicit_not_fabricated(tmp_path):
    calls=tmp_path/'wire-ledger/calls'; calls.mkdir(parents=True)
    (calls/'a.started.json').write_text(json.dumps({'id':'a','role':'customer','request':{'messages':[]}}))
    result=audit.audit(tmp_path,'ledger')
    assert result['groups']['customer']['missing_usage_calls']==1
    assert result['calls'][0]['input_tokens'] is None
