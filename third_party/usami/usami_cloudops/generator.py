"""Small deterministic worlds; discovery never receives workflow conditioning."""
from usami.randomness import StableRandom
from .models import World, Organization, Identity, Project, Quota, Access, Instance, Invoice, Payment


def generate_world(seed=42):
    rng = StableRandom(seed)
    users = {f'u{n}': Identity(id=f'u{n}', name=f'Employee {n}') for n in range(5)}
    orgs = {f'o{n}': Organization(id=f'o{n}', name=f'Organization {n}', owner_id=f'u{n * 4}') for n in range(2)}
    projects, access, instances, invoices = {}, {}, {}, {}
    for n, name in enumerate(('production', 'development', 'staging', 'production')):
        pid = f'p{n}'; org = 'o0' if n < 3 else 'o1'; iid = f'bill{n}'
        projects[pid] = Project(id=pid, name=name, org_id=org, status='suspended' if n == 2 else 'active',
            invoice_id=iid, quota=Quota(cpu=4, memory=8, instances=2))
        payments = [] if n == 2 else [Payment(id=f'original{n}', key='original', method='card_approved',
            amount_cents=1000 + 100 * n, outcome='succeeded', at=World.model_fields['clock'].default)]
        invoices[iid] = Invoice(id=iid, project_id=pid, amount_cents=1000 + 100 * n,
            status='unpaid' if n == 2 else 'paid', payments=payments)
        for index, suffix in enumerate(('db', 'app', 'web')):
            rid = pid + '_' + suffix
            deps = [] if index == 0 else [pid + '_' + ('db' if index == 1 else 'app')]
            instances[rid] = Instance(id=rid, name=suffix.title() + ' ' + str(rng.randint(100, 999)), project_id=pid,
                cpu=2, memory=4, dependencies=deps)
    for pid, uid, role in [('p0','u0','admin'), ('p1','u0','admin'), ('p0','u1','viewer'), ('p0','u2','viewer'), ('p1','u1','admin'),
        ('p1','u2','operator'), ('p1','u3','viewer'), ('p2','u2','operator')]:
        key = pid + '_' + uid
        access[key] = Access(id=key, project_id=pid, user_id=uid, role=role)
    world = World(seed=seed, users=users, organizations=orgs, projects=projects, access=access, instances=instances, invoices=invoices)
    from .rules import invariants
    if invariants(world): raise ValueError('invalid generated CloudOps world')
    return world
