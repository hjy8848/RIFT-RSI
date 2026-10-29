"""Independent native generation and content-addressed registration, never Designer edits."""
import hashlib
import json
import random
from pathlib import Path
from typing import Literal, Tuple
from pydantic import Field
from vast_coevo.schema import FrozenModel
from vast_coevo.designer.storage import digest, atomic_json
from usami_cloudops.generator import generate_world
from usami_cloudops.models import World
from usami_cloudops.rules import invariants
from integrations.usami.customer_probes import business_shape


class WorldSpec(FrozenModel):
    id: str=Field(pattern=r'^[a-zA-Z][a-zA-Z0-9_-]{0,63}$')
    seed: int=Field(strict=True)
    partition: Literal['development','heldout']='development'
    generator: Literal['cloudops-native-v1']='cloudops-native-v1'


class FrozenWorld(FrozenModel):
    spec: WorldSpec
    world: dict
    world_hash: str
    structure_hash: str
    summary: dict
    generator_sha256: str

    def checked(self):
        generated=make_world(self.spec)
        if generated!=self:raise ValueError('frozen world differs from registered independent generator')
        return self


def make_world(spec):
    from usami_cloudops import generator
    raw=json.loads(json.dumps(generate_world(spec.seed).model_dump(mode='json'),sort_keys=True))
    world=World.model_validate(raw)
    if invariants(world):raise ValueError('generator produced invalid world')
    shapes=sorted(business_shape(world,uid) for uid in world.users)
    return FrozenWorld(spec=spec,world=raw,world_hash=world.state_hash,structure_hash=digest(shapes),
        summary={'projects':len(world.projects),'identities':len(world.users),'instances':len(world.instances),
                 'project_states':sorted(p.status for p in world.projects.values()),
                 'roles':sorted(a.role for a in world.access.values()),'dependency_edges':sum(len(i.dependencies) for i in world.instances.values())},
        generator_sha256=hashlib.sha256(Path(generator.__file__).read_bytes()).hexdigest())


class WorldPool:
    def __init__(self,specs):
        specs=tuple(WorldSpec.model_validate(s.model_dump(mode='json') if isinstance(s,WorldSpec) else s) for s in specs)
        if not specs or len(specs)>128 or len({s.id for s in specs})!=len(specs):raise ValueError('world pool requires 1..128 unique registrations')
        if any(s.partition!='development' for s in specs):raise ValueError('heldout worlds cannot enter Designer pool')
        self.entries={s.id:make_world(s) for s in specs}
        if len({e.world_hash for e in self.entries.values()})!=len(self.entries):raise ValueError('duplicate world initial states')
        self._entry_seals={key:digest(value) for key,value in self.entries.items()}
        self.pool_hash=digest(self.manifest())
    def manifest(self):return {'protocol':'frozen-world-pool-v1','entries':[self.entries[k].model_dump(mode='json') for k in sorted(self.entries)]}
    def load(self,world_id,world_hash):
        entry=self.entries.get(world_id)
        if entry is None or entry.world_hash!=world_hash:raise ValueError('unknown world or frozen world hash mismatch')
        if digest(entry)!=self._entry_seals[world_id] or digest(entry.world)!=entry.world_hash:raise ValueError('registered world was mutated')
        return FrozenWorld.model_validate(entry.model_dump(mode='json'))
    def window(self,seed,limit=4):
        keys=sorted(self.entries);random.Random(seed).shuffle(keys)
        return [self.load(k,self.entries[k].world_hash) for k in keys[:limit]]
    def save(self,root):
        atomic_json(Path(root)/'world-pool.json',self.manifest(),immutable=True)
    def verify_saved(self,root):
        if json.loads((Path(root)/'world-pool.json').read_text())!=self.manifest():raise ValueError('frozen world pool changed')
