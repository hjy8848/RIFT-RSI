import { useEffect, useState } from 'react';
import { api } from './api';

export type Domain = 'saas' | 'cloudops' | 'fulfillment' | 'stayops';
export const names: Record<Domain, string> = { saas: 'SaaS', cloudops: 'CloudOps', fulfillment: 'FulfillmentOps', stayops: 'StayOps' };
export interface DomainInfo { id: Domain; name: string; version: string; scope: string; description: string; metrics: Record<string, number> }
export default function DomainSwitch({ domain, change }: { domain: Domain; change: (domain: Domain) => void }) {
  const [catalog, setCatalog] = useState<DomainInfo[]>([]);
  useEffect(() => { let active = true; void api<{ domains: DomainInfo[] }>('/domains').then(d => { if (active) setCatalog(d.domains); }).catch(() => { /* Domain names remain available when the API is offline. */ }); return () => { active = false; }; }, []);
  return <label className="domain-switch-label">BUSINESS DOMAIN<select aria-label="Business domain" value={domain} onChange={e => change(e.target.value as Domain)}>{Object.entries(names).map(([id, name]) => { const info = catalog.find(d => d.id === id); return <option key={id} value={id}>{name}{info ? ` · ${info.metrics.frozen_service_tasks} tasks` : ''}</option>; })}</select></label>;
}
