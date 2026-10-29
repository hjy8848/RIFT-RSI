"""Recomputable world-split audit and finite, independent private pool generation.

World disjointness is not task-semantic disjointness. This module reports that
unimplemented axis explicitly instead of promoting new seeds to a leakage proof.
"""
from itertools import combinations
from pathlib import Path
from usami_harness.models import digest
from vast_coevo.schema import atomic_json
from .contracts import WorldSpec
from .worlds import FrozenWorld,WorldPool,freeze


def verified_world(entry):
    entry=FrozenWorld.model_validate(entry.model_dump(mode='json') if isinstance(entry,FrozenWorld) else entry).checked()
    if freeze(entry.spec)!=entry:raise ValueError('world registration/signatures/source do not match independent generation')
    return entry


def audit_world_splits(entries):
    rows=[verified_world(e) for e in entries]
    keys=[r.registration_hash for r in rows]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate world registration in audit')
    partitions=sorted({r.spec.partition for r in rows});overlaps={}
    for a,b in combinations(partitions,2):
        left=[r for r in rows if r.spec.partition==a];right=[r for r in rows if r.spec.partition==b]
        def common(attribute):
            return len({(r.spec.domain,getattr(r,attribute)) for r in left}&{(r.spec.domain,getattr(r,attribute)) for r in right})
        overlaps[a+' / '+b]={'exact_worlds':common('world_hash'),'anonymous_structures':common('structure_signature')}
    return {'protocol':'eight-domain-world-split-audit-v1','registered_worlds':len(rows),
        'registration_commitment':digest(sorted(keys)),
        'partitions':{p:{'worlds':sum(r.spec.partition==p for r in rows),'domains':sorted({r.spec.domain for r in rows if r.spec.partition==p})} for p in partitions},
        'overlaps':overlaps,'world_isolation_pass':not any(v['exact_worlds'] or v['anonymous_structures'] for v in overlaps.values()),
        'task_semantic_isolation':'NOT RUN','task_combination_isolation':'NOT RUN',
        'scope':'Exact independently regenerated worlds and anonymous business structures only; not a full task leakage audit.'}


def generate_private_heldout(specs,known_worlds,root,*,requested_count):
    """Freeze candidates before execution; exclude previously seen world/structures.

    The finite input schedule is preregistered verbatim. No task, agent, result,
    skill or feedback participates in generation or exclusion. No auto-resampling.
    """
    if type(requested_count) is not int or requested_count<=0:raise ValueError('positive requested count required')
    specs=[WorldSpec.model_validate(s.model_dump(mode='json') if isinstance(s,WorldSpec) else s) for s in specs]
    if len(specs)<requested_count or any(s.partition!='heldout' for s in specs):raise ValueError('finite heldout candidate schedule required')
    if len({s.spec_hash for s in specs})!=len(specs):raise ValueError('duplicate candidate specification')
    known=[verified_world(e) for e in known_worlds]
    root=Path(root).resolve();repo=Path(__file__).resolve().parents[1]
    try:root.relative_to(repo)
    except ValueError:pass
    else:raise ValueError('private Held-out must be generated outside the public repository')
    root.mkdir(mode=0o700,parents=True,exist_ok=False)
    plan={'protocol':'private-heldout-generation-v1','requested_count':requested_count,'candidate_schedule':[s.model_dump(mode='json') for s in specs],
        'known_registration_commitment':digest(sorted(r.registration_hash for r in known)),
        'exclusion_policy':'Exact world or anonymous structure previously registered; no task conditioning or feedback.',
        'evaluation_status':'NOT RUN'}
    atomic_json(root/'registration.json',plan,immutable=True)
    blocked_worlds={(r.spec.domain,r.world_hash) for r in known};blocked_structures={(r.spec.domain,r.structure_signature) for r in known}
    selected=[];excluded=[];failed=[];processed=0
    for spec in specs:
        if len(selected)==requested_count:break
        processed+=1
        try:row=freeze(spec)
        except (ValueError,TypeError) as exc:failed.append({'spec_hash':spec.spec_hash,'reason':str(exc)});continue
        if (spec.domain,row.world_hash) in blocked_worlds:excluded.append({'spec_hash':spec.spec_hash,'reason':'previously_seen_world'});continue
        if (spec.domain,row.structure_signature) in blocked_structures:excluded.append({'spec_hash':spec.spec_hash,'reason':'previously_seen_anonymous_structure'});continue
        selected.append(row);blocked_worlds.add((spec.domain,row.world_hash));blocked_structures.add((spec.domain,row.structure_signature))
    if selected:WorldPool(selected).save(root/'world-pool.json')
    report={'protocol':'private-heldout-generation-report-v1','registration_hash':digest(plan),'requested_count':requested_count,'actual_count':len(selected),
        'processed_candidates':processed,'complete':len(selected)==requested_count,'excluded':excluded,'generation_failures':failed,
        'world_split_audit':audit_world_splits(known+selected),'evaluation_status':'NOT RUN',
        'task_semantic_isolation':'NOT RUN','privacy':'Full worlds/seeds remain only in the private output directory. Do not commit it.'}
    atomic_json(root/'report.json',report,immutable=True)
    return report
