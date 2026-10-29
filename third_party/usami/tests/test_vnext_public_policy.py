"""Static public code vocabulary must not depend on a task, world or hidden answer."""
import pytest
from usami_benchmark.contracts import DOMAINS
from usami_benchmark.registry import RULES
from usami_benchmark.public_policy import catalogue
from usami_harness.models import digest


@pytest.mark.parametrize('domain',DOMAINS)
def test_all_domains_have_versioned_public_rules_and_code_catalogue(domain):
    c=catalogue(domain,RULES[domain]);assert c['business_rules'] and c['errors']
    assert c['catalogue_hash']==digest({k:v for k,v in c.items() if k!='catalogue_hash'})
    assert 'world' not in c and 'witness' not in c and 'expected_outcome' not in c
    c['errors'].clear();c['business_rules'].append('cache corruption')
    other=catalogue(domain,RULES[domain]);assert other['errors'] and 'cache corruption' not in other['business_rules']


def test_readable_schema_codes_include_education_decisions_beyond_old_short_policy():
    c=catalogue('education',RULES['education'])
    assert {'TEACHER_QUALIFICATION_REQUIRED','EXTENSION_TOO_SHORT','STALE_ENROLLMENT_QUOTE'}<=set(c['errors'])
    with pytest.raises(ValueError):catalogue('education','unregistered')


def test_aftersales_public_rules_explain_case_creation_window_boundary():
    c=catalogue('aftersales',RULES['aftersales'])
    assert any('A case can be opened only' in rule and 'actual delivery date' in rule
               and 'manager approval does not extend it' in rule for rule in c['business_rules'])
    assert 'AFTERSALES_WINDOW_EXPIRED' in c['errors']
