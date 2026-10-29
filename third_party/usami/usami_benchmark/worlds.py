"""Content-addressed immutable worlds and partition-aware independent pools."""
from pathlib import Path
from pydantic import Field
from usami_harness.models import Model,digest
from vast_coevo.schema import atomic_json
from .contracts import WorldSpec
from .registry import generate,invariants,source_modules
from .signatures import signatures


class FrozenWorld(Model):
    spec: WorldSpec
    world: dict
    world_hash: str
    structure_signature: str
    permission_signature: str
    generator_source_hash: str
    generation_diagnostics: list=Field(default_factory=list)
    rules_source_hash: str
    def checked(self):
        if digest(self.world)!=self.world_hash:raise ValueError('frozen world content hash differs')
        return self
    @property
    def registration_hash(self):return digest(self)


def freeze(spec):
    import hashlib,importlib
    diagnostics=[];world=generate(spec,diagnostics=diagnostics);sig=signatures(spec.domain,world)
    generator,rules=source_modules(spec)
    return FrozenWorld(spec=spec,world=world.model_dump(mode='json'),world_hash=world.state_hash,
        structure_signature=sig['structure'],permission_signature=sig['permissions'],generation_diagnostics=diagnostics,
        generator_source_hash=hashlib.sha256(Path(generator.__file__).read_bytes()).hexdigest(),
        rules_source_hash=hashlib.sha256(Path(rules.__file__).read_bytes()).hexdigest()).checked()


class WorldPool:
    def __init__(self,entries,*,discovery=False):
        entries=[FrozenWorld.model_validate(e.model_dump(mode='json') if isinstance(e,FrozenWorld) else e).checked() for e in entries]
        if not entries or len({e.registration_hash for e in entries})!=len(entries):raise ValueError('pool requires distinct registered worlds')
        if discovery and any(e.spec.partition not in {'train','validation'} for e in entries):raise ValueError('test/heldout cannot enter training Discovery pool')
        self.entries={e.registration_hash:e for e in entries};self.seals={k:digest(e) for k,e in self.entries.items()}
        self.pool_hash=digest(sorted(self.seals))
    def load(self,key):
        row=self.entries.get(key)
        if row is None or digest(row)!=self.seals[key]:raise ValueError('unknown or mutated world registration')
        row.checked()
        return FrozenWorld.model_validate(row.model_dump(mode='json'))
    def save(self,path):atomic_json(Path(path),{'protocol':'eight-domain-world-pool-v1','pool_hash':self.pool_hash,'worlds':[self.load(k).model_dump(mode='json') for k in sorted(self.entries)]},immutable=True)


def partition_for(structure):
    bucket=int(structure[:16],16)%20
    return 'train' if bucket<14 else 'validation' if bucket<17 else 'test_demo'
