"""Role-filtered initial views; replies never include evaluator data automatically."""
from .rules import owner, role


def observation(world, identity):
    if identity not in world.users: raise ValueError('unknown customer identity')
    projects = {p.id: p for p in world.projects.values() if owner(world, identity, p.id) or role(world, identity, p.id)}
    admin = {p for p in projects if owner(world, identity, p) or role(world, identity, p) == 'admin'}
    orgs = {p.org_id for p in projects.values()}
    return {'domain': 'cloudops', 'version': world.version, 'identity': world.users[identity].model_dump(mode='json'),
        'clock': world.clock.isoformat(), 'organizations': [o.model_dump(mode='json') for o in world.organizations.values() if o.id in orgs],
        'projects': [p.model_dump(mode='json') for p in projects.values()],
        'instances': [i.model_dump(mode='json') for i in world.instances.values() if i.project_id in projects],
        'project_access': [a.model_dump(mode='json') for a in world.access.values() if a.project_id in projects],
        'invoices': [i.model_dump(mode='json') for i in world.invoices.values() if i.project_id in admin],
        'public_directory': [{'id': u.id, 'name': u.name} for u in world.users.values()]}


def visible_ids(view):
    return {obj['id'] for key in ('organizations','projects','instances','project_access','invoices','public_directory') for obj in view[key]} | {view['identity']['id']}
