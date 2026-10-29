"""Post-training, read-only Snapshot comparisons. Never a training dependency."""
from pathlib import Path
import re

from ..evidence.archive import EvidenceArchive
from ..models import Consumption, EpisodeConfig, EpisodeEvidence
from ..schema import atomic_json, digest
from ..skills.library import LibrarySnapshot
from ..storage.runs import add_consumption


def evaluate_cross(adapter, pairs, configs, output, *, evaluation_id, purpose,
                   excluded_world_hashes, max_llm_calls=500, max_tool_calls=1000):
    """Run new independent Episodes; no reflector, gate or mutable library API.

    Adapter must own an independent native output directory. Failed evaluations
    retain their manifest/progress and cannot silently restart external calls.
    ``purpose`` seals development diagnostics separately from frozen held-out.
    """
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', evaluation_id):
        raise ValueError('invalid evaluation ID')
    if purpose not in ('development-diagnostic', 'frozen-heldout'):
        raise ValueError('explicit evaluation purpose required')
    if any(type(v) is not int or v < 0 for v in (max_llm_calls, max_tool_calls)):
        raise ValueError('evaluation budgets must be nonnegative integers')
    configs = tuple(EpisodeConfig.model_validate(c.model_dump(mode='json')) for c in configs)
    excluded = set(excluded_world_hashes)
    if not configs or not pairs or not excluded:
        raise ValueError('pairs, independent worlds and training-world exclusions are required')
    partition = 'heldout' if purpose == 'frozen-heldout' else 'development'
    if any(c.partition != partition or c.initial_world_hash in excluded for c in configs):
        raise ValueError('evaluation world overlaps training or purpose/partition differs')
    if len({c.config_hash for c in configs}) != len(configs):
        raise ValueError('duplicate evaluation cases')
    if len({(digest(c.customer_model), digest(c.service_model)) for c in configs}) != 1:
        raise ValueError('evaluation models must be fixed')
    copied = {}
    for label, pair in pairs.items():
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', label):
            raise ValueError('invalid pairing label')
        c, s = [LibrarySnapshot.model_validate(v.model_dump(mode='json')) for v in pair]
        if (c.role, s.role) != ('customer', 'service'):
            raise ValueError('pairing Snapshot roles differ')
        copied[label] = (c, s)
    output = Path(output)
    if output.exists(): raise ValueError('evaluation already exists; no repeat or overwrite')
    manifest = dict(protocol='post-training-cross-v1', purpose=purpose, selection_authority=False,
                    reflection_calls=0, evaluation_id=evaluation_id,
                    excluded_training_worlds=sorted(excluded), configs=[c.model_dump(mode='json') for c in configs],
                    pairs={k:[v.model_dump(mode='json') for v in pair] for k,pair in copied.items()},
                    max_llm_calls=max_llm_calls, max_tool_calls=max_tool_calls,
                    failure_policy='retain original error; no retries; no training feedback')
    atomic_json(output/'manifest.json', manifest, immutable=True)
    archive = EvidenceArchive(output/'evidence')
    rows=[];used=Consumption();pending_reservation=None
    try:
        for label, (customer, service) in copied.items():
            for n, config in enumerate(configs):
                b=config.budget
                if (used.llm_calls+b.max_customer_llm_calls+b.max_service_llm_calls > max_llm_calls
                        or used.tool_calls+b.max_service_tool_calls > max_tool_calls):
                    raise ValueError('independent evaluation budget exhausted before new Episode')
                ident=evaluation_id+'.'+label+'.'+str(n)
                pending_reservation=Consumption(customer_llm_calls=b.max_customer_llm_calls,service_llm_calls=b.max_service_llm_calls,tool_calls=b.max_service_tool_calls)
                atomic_json(output/'attempts'/(digest(ident)+'.started.json'),dict(episode_id=ident,config_hash=config.config_hash,customer=customer.snapshot_hash,service=service.snapshot_hash,reservation=pending_reservation.model_dump(mode='json')),immutable=True)
                e=EpisodeEvidence.model_validate(adapter.run_episode(config,customer,service,ident).model_dump(mode='json'))
                used=add_consumption((used,e.consumption))
                atomic_json(output/'attempts'/(digest(ident)+'.completed.json'),e,immutable=True)
                reserve=pending_reservation;pending_reservation=None
                if any(getattr(e.consumption,k)>getattr(reserve,k) for k in Consumption.model_fields):
                    raise ValueError('evaluation exceeded its reserved Episode budget')
                if (e.episode_id!=ident or e.config!=config or e.track!='discovery'
                        or (e.customer_snapshot_hash,e.service_snapshot_hash)!=(customer.snapshot_hash,service.snapshot_hash)):
                    raise ValueError('evaluation returned another pairing or configuration')
                checked=adapter.replay(e,customer,service)
                if not e.replay.verified or not checked.verified:raise ValueError('evaluation Replay failed')
                archive.save(e)
                rows.append(dict(pair=label,case=n,evidence_hash=e.evidence_hash,trace=e.trace.model_dump(mode='json'),
                    termination=e.termination,successful_business_violations=sum(v.classification=='successful_business_violation' for v in e.events),
                    independent_failures=sorted({v.fingerprint for v in e.failures}),consumption=e.consumption.model_dump(mode='json')))
                atomic_json(output/'progress.json',dict(rows=rows,consumption=used.model_dump(mode='json')))
        result=dict(status='completed',protocol=manifest['protocol'],purpose=purpose,selection_authority=False,
                    reflection_calls=0,manifest_hash=digest(manifest),rows=rows,consumption=used.model_dump(mode='json'))
        atomic_json(output/'result.json',result,immutable=True)
        return result
    except BaseException as exc:
        atomic_json(output/'result.json',dict(status='failed',error=type(exc).__name__+': '+str(exc),rows=rows,
            consumption=used.model_dump(mode='json'),unresolved_reservation=pending_reservation.model_dump(mode='json') if pending_reservation else None,selection_authority=False,reflection_calls=0),immutable=True)
        raise
