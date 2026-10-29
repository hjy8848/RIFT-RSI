import { Activity, ArrowDownToLine, ArrowRight, Box, CircleHelp, Clock3, Code2, ExternalLink, FlaskConical, LayoutDashboard, Loader2, Network, Plus, RefreshCw, RotateCcw, ShieldCheck, X } from 'lucide-react';
import { useEffect, useState } from 'react';
import { api, ApiError } from './api';
import { Badge, CopyValue, Inspector, JsonView, ModeSwitch, Status, short } from './ui';
import DomainSwitch, { names, type Domain, type DomainInfo } from './DomainSwitch';
import DomainBenchmark from './DomainBenchmark';
import DomainGraph from './DomainGraph';
import type { DomainSnapshot, Entity, ManualResult, Schema, Tool } from './domainTypes';

type DomainPage = 'overview' | 'entities' | 'graph' | 'playground' | 'benchmark' | 'timeline';
const pages = [
  { id: 'overview', label: 'World Overview', icon: LayoutDashboard }, { id: 'entities', label: 'Business Entities', icon: Box },
  { id: 'graph', label: 'World Graph', icon: Network }, { id: 'playground', label: 'Operation Playground', icon: FlaskConical },
  { id: 'benchmark', label: 'Benchmark & Pilots', icon: ShieldCheck }, { id: 'timeline', label: 'Event Timeline', icon: Activity },
] as const;
export type Inspect = (title: string, value: unknown) => void;
export type Mutate = <T>(work: () => Promise<T>) => Promise<T | undefined>;
const scopeCollections = { cloudops: 'projects', fulfillment: 'warehouses', stayops: 'properties' };

export default function DomainExplorer({ domain, changeDomain }: { domain: Exclude<Domain, 'saas'>; changeDomain: (domain: Domain) => void }) {
  const base = `/domains/${domain}`;
  const [snapshot, setSnapshot] = useState<DomainSnapshot | null>(null);
  const [page, setPage] = useState<DomainPage>('overview');
  const [seed, setSeed] = useState('42');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [catalog, setCatalog] = useState<DomainInfo[]>([]);
  const [inspector, setInspector] = useState<{ title: string; value: unknown } | null>(null);
  const inspect: Inspect = (title, value) => setInspector({ title, value });
  const replace = (s: DomainSnapshot) => { setSnapshot(s); setSeed(String(s.world.seed)); };
  const restart = (s: DomainSnapshot) => { replace(s); setEpoch(value => value + 1); setInspector(null); };
  const mutate: Mutate = async work => {
    setBusy(true); setError('');
    try { return await work(); } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      if (e instanceof ApiError && e.status === 409) { try { replace(await api<DomainSnapshot>(`${base}/world`)); } catch { /* Original error stays visible. */ } }
    } finally { setBusy(false); }
  };
  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        let data: DomainSnapshot;
        try { data = await api<DomainSnapshot>(`${base}/world`); }
        catch (e) { if (!(e instanceof ApiError) || e.status !== 401) throw e; data = await api<DomainSnapshot>(`${base}/sessions`, { seed: 42 }); }
        if (active) replace(data);
      } catch (e) { if (active) setError(String(e)); }
    })();
    void api<{ domains: DomainInfo[] }>('/domains').then(data => { if (active) setCatalog(data.domains); }).catch(e => { if (active) setError(String(e)); });
    return () => { active = false; };
  }, [base]);
  async function download(kind: 'world' | 'trace') {
    const response = await fetch(`/api${base}/export/${kind}`);
    if (!response.ok) throw new Error(`Export failed: ${response.status}`);
    const url = URL.createObjectURL(await response.blob()); const a = document.createElement('a');
    a.href = url; a.download = `${domain}-${kind}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const activePage = pages.find(p => p.id === page)!;
  const info = catalog.find(d => d.id === domain);
  const scope = scopeCollections[domain];
  return <div className="app-shell domain-explorer">
    <aside className="sidebar"><a className="brand" href="#" onClick={e => { e.preventDefault(); setPage('overview'); }}><span className="brand-mark"><Box size={23} /></span><div><strong>usami<span>.</span></strong><small>WORLD EXPLORER</small></div></a>
      <DomainSwitch domain={domain} change={changeDomain} /><div className="workspace-switch"><span className="workspace-icon"><Box size={17} /></span><div><strong>{names[domain]}</strong><span>Independent business world</span></div><Badge>v0.1</Badge></div>
      <div className="nav-label">EXPLORE</div><nav aria-label="Main navigation">{pages.map(p => <button key={p.id} className={`nav-item ${page === p.id ? 'active' : ''}`} aria-current={page === p.id ? 'page' : undefined} onClick={() => setPage(p.id)}><p.icon size={18} /><span>{p.label}</span>{page === p.id && <i className="nav-dot" />}</button>)}</nav>
      <div className="sidebar-bottom"><div className="local-status"><span className={`live-dot ${snapshot ? '' : 'inactive'}`} />{snapshot ? 'Python engine connected' : 'Connecting to engine'}</div><div className="sidebar-info"><ShieldCheck size={17} /><p>真实业务状态<br /><strong>独立 Oracle 验证</strong></p></div><a href="https://github.com/hjy8848/Usami" target="_blank" rel="noreferrer">View repository<ExternalLink size={13} /></a></div>
    </aside>
    <div className="main-shell"><header className="topbar"><div className="breadcrumb"><span>Workspace</span><ArrowRight size={13} /><strong>{names[domain]}</strong></div><div className="topbar-right"><span>IN-MEMORY · LOCAL</span><Badge tone="blue">{snapshot?.world.version ?? `${domain}-0.1.0`}</Badge></div></header>
      <div className="world-toolbar"><div className="world-ident"><span className="live-dot" /><span>WORLD</span>{snapshot && <CopyValue value={snapshot.state_hash} label={short(snapshot.state_hash)} />}</div><form className="generator-controls" onSubmit={e => { e.preventDefault(); void mutate(async () => restart(await api(`${base}/world/generate`, { seed: Number(seed) }))); }}><label className="seed-control"><span>Seed</span><input aria-label="Seed" type="number" required step="1" value={seed} disabled={busy} onChange={e => setSeed(e.target.value)} /></label><button className="button button-primary compact" disabled={busy || !snapshot}><Plus size={15} />Generate World</button></form><div className="toolbar-actions"><button className="button compact" disabled={busy || !snapshot} onClick={() => void mutate(async () => restart(await api(`${base}/world/reset`, {})))}><RotateCcw size={14} />Reset World</button><button className="button compact" disabled={!snapshot || busy} onClick={() => void mutate(() => download('world'))}><ArrowDownToLine size={14} />Export JSON</button></div></div>
      {error && <div className="error-banner" role="alert"><CircleHelp size={18} /><span>{error}</span><button className="icon-button" aria-label="Dismiss error" onClick={() => setError('')}><X size={16} /></button></div>}
      {!snapshot ? <div className="loading-screen"><Loader2 size={26} className="spin" /><h2>Connecting to {names[domain]}</h2><p>加载独立 Python 业务引擎。</p></div> : <main className={`page page-${page}`}>
        <div className="page-heading"><div><div className="eyebrow">{names[domain].toUpperCase()} · EXECUTABLE BUSINESS WORLD</div><h1>{activePage.label}</h1><p>{info?.description ?? '独立领域模型、实际状态转换与可重放证据。'}</p></div><div className="sim-clock"><Clock3 size={15} /><div><span>FROZEN CLOCK · UTC</span><strong>{snapshot.world.clock.replace('T', ' ').slice(0, 16)}</strong></div></div></div>
        <div className="scenario-notice"><ShieldCheck size={18} /><div><strong>{snapshot.exploration_source ? 'Frozen instance · manual exploration' : 'Manual research workspace'}</strong><span>{snapshot.exploration_source ? `${snapshot.exploration_source.instance_id} · 固定实例身份 ${snapshot.exploration_source.identity}。` : ''}手动请求由所选主体明确声明，不产生正式 Benchmark 分数，也不构成 Service LLM 的 Customer 授权。</span></div></div>
        {!!snapshot.invariants.length && <div className="warning-banner"><CircleHelp size={17} />当前世界存在 {snapshot.invariants.length} 项不变量违规；后续判定可能为 Unknown。请查看证据或重置。</div>}
        {page === 'overview' && <Overview snapshot={snapshot} scope={scope} catalog={catalog} change={changeDomain} inspect={inspect} navigate={setPage} />}
        {page === 'entities' && <Entities snapshot={snapshot} inspect={inspect} />}
        {page === 'graph' && <DomainGraph base={base} snapshot={snapshot} scopes={snapshot.collections[scope]} inspect={inspect} onError={setError} />}
        {page === 'playground' && <Playground key={`${snapshot.initial_state_hash}:${epoch}:${snapshot.exploration_source?.instance_id ?? ''}`} base={base} snapshot={snapshot} scope={scope} busy={busy} mutate={mutate} replace={replace} inspect={inspect} />}
        {page === 'benchmark' && <DomainBenchmark base={base} snapshot={snapshot} busy={busy} mutate={mutate} replace={replace} inspect={inspect} navigate={setPage} />}
        {page === 'timeline' && <Timeline key={epoch} base={base} snapshot={snapshot} busy={busy} mutate={mutate} inspect={inspect} download={download} />}
        <footer className="page-footer"><span><Box size={12} />{names[domain]} · actual Engine / independent Oracle</span><span>{snapshot.operations_used} / {snapshot.operation_budget} manual operations</span></footer>
      </main>}
    </div>{busy && <div className="busy-indicator" role="status"><Loader2 size={14} className="spin" />Running Python engine…</div>}{inspector && <Inspector title={inspector.title} close={() => setInspector(null)}><JsonView value={inspector.value} /></Inspector>}
  </div>;
}

function Overview({ snapshot: s, scope, catalog, change, inspect, navigate }: { snapshot: DomainSnapshot; scope: string; catalog: DomainInfo[]; change: (d: Domain) => void; inspect: Inspect; navigate: (p: DomainPage) => void }) {
  const primary = s.domain === 'cloudops' ? ['projects', 'instances', 'access', 'invoices'] : s.domain === 'fulfillment' ? ['warehouses', 'lots', 'orders', 'shipments'] : ['properties', 'rooms', 'reservations', 'stays'];
  return <><section className="intro-card domain-intro"><div className="intro-copy"><Badge tone="blue">INDEPENDENT DOMAIN</Badge><h2>{names[s.domain as Domain]}.<br />Real state dependencies.</h2><p>{s.domain === 'cloudops' ? '资源拓扑、CPU / Memory 配额、账单和项目权限共同影响后续操作。' : s.domain === 'fulfillment' ? 'On-hand、Reserved、Available、Picked 与 Shipped 分开建模，操作改变后续履约与退货的可行性。' : '房型按每晚计库存，具体房间独立分配；清洁、维护、押金和身份验证共同约束入住。'}</p><button className="intro-link" onClick={() => navigate('graph')}>Explore the world graph<ArrowRight size={15} /></button></div><div className="domain-intro-evidence"><ShieldCheck size={32} /><strong>Reference → Engine → Oracle → Replay</strong><span>真实转换 · 独立规则 · 完整事件</span><Status status={s.evaluation.status} /></div></section>
    <section className="stats-grid">{primary.map(k => <div className="stat-card" key={k}><div className="stat-top"><span>{label(k)}</span><Box size={17} /></div><strong>{s.summary[k] ?? 0}</strong><small>Live world entities</small></div>)}</section>
    <div className="section-heading"><div><h2>{label(scope)}</h2><p>Scope boundaries are enforced by the actual domain rules.</p></div><button className="text-button" onClick={() => navigate('entities')}>Browse all entities<ArrowRight size={14} /></button></div>
    <section className="org-grid">{s.collections[scope]?.map(row => <button key={row.id} className="org-card domain-scope-card" onClick={() => inspect(`${label(scope)} · ${row.id}`, row)}><div className="org-card-heading"><span className="org-symbol"><Box size={21} /></span><div><h3>{String(row.name ?? row.id)}</h3><code>{row.id}</code></div><ArrowRight size={16} /></div><div className="domain-scope-fields">{Object.entries(row).filter(([k]) => !['id', 'name'].includes(k)).map(([k, v]) => <div key={k}><span>{label(k)}</span><strong>{text(v)}</strong></div>)}</div></button>)}</section>
    <section className="panel domain-register"><div className="panel-heading"><div><h2>Four-domain benchmark registry</h2><p>实际发布文件中的指标；每个领域单独统计。</p></div><ShieldCheck size={18} /></div><div className="domain-registry-grid">{catalog.map(d => <button key={d.id} onClick={() => change(d.id)}><strong>{d.name}<Badge>v{d.version}</Badge></strong><b>{d.metrics.frozen_service_tasks}</b><span>Frozen Service tasks</span><small>{d.description}</small></button>)}</div><button className="text-button" onClick={() => navigate('benchmark')}>Browse this domain's tasks and real pilots<ArrowRight size={14} /></button></section>
  </>;
}

export const label = (name: string) => name.replaceAll('_', ' ').replace(/^./, c => c.toUpperCase());
const text = (value: unknown) => value == null ? '—' : typeof value === 'object' ? JSON.stringify(value) : String(value);
function Entities({ snapshot: s, inspect }: { snapshot: DomainSnapshot; inspect: Inspect }) {
  const [collection, setCollection] = useState(Object.keys(s.collections).find(k => !['users', 'access'].includes(k)) ?? 'users');
  const [search, setSearch] = useState(''); const [scope, setScope] = useState('');
  const scopes = s.collections[scopeCollections[s.domain as keyof typeof scopeCollections]] ?? [];
  const rows = (s.collections[collection] ?? []).filter(r => JSON.stringify(r).toLowerCase().includes(search.toLowerCase()) && (!scope || r.property_id === scope || r.warehouse_id === scope || r.project_id === scope || r.source_warehouse_id === scope || r.target_warehouse_id === scope || (r.lot_id && s.collections.lots?.some(l => l.id === r.lot_id && l.warehouse_id === scope)) || r.id === scope));
  const fields = [...new Set(rows.flatMap(r => Object.keys(r)))].filter(k => k !== 'id').slice(0, 7);
  return <div className="domain-entity-layout"><section className="panel domain-collections"><h2>Collections</h2>{Object.keys(s.collections).map(k => <button key={k} className={collection === k ? 'selected' : ''} onClick={() => setCollection(k)}><span>{label(k)}</span><Badge>{s.summary[k]}</Badge></button>)}</section><section className="panel"><div className="panel-heading"><h2>{label(collection)} <Badge>{rows.length}</Badge></h2></div><div className="domain-filters"><label>Search entities<input aria-label="Search entities" value={search} onChange={e => setSearch(e.target.value)} placeholder="ID, state, role, date…" /></label><label>Resource scope<select aria-label="Entity scope" value={scope} onChange={e => setScope(e.target.value)}><option value="">All scopes</option>{scopes.map(r => <option key={r.id} value={r.id}>{String(r.name ?? r.id)}</option>)}</select></label></div><div className="table-scroll"><table><thead><tr><th>Entity</th>{fields.map(k => <th key={k}>{label(k)}</th>)}</tr></thead><tbody>{rows.map(r => <tr key={r.id}><td><button className="table-link" onClick={() => inspect(`${label(collection)} · ${r.id}`, r)}>{r.id}</button></td>{fields.map(k => <td key={k} className="domain-cell" title={text(r[k])}>{['status', 'housekeeping', 'outcome'].includes(k) ? <Status status={String(r[k])} /> : text(r[k])}</td>)}</tr>)}</tbody></table></div>{!rows.length && <p className="empty-small">No entities match this view.</p>}</section></div>;
}

function defaults(schema: Schema, key: string, s: DomainSnapshot): unknown {
  if (schema.default !== undefined) return schema.default;
  if (schema.const !== undefined) return schema.const;
  if (schema.enum) return schema.enum[0];
  if (schema.type === 'boolean') return false;
  if (schema.type === 'integer') return Math.max(schema.minimum ?? 1, (schema.exclusiveMinimum ?? 0) + 1);
  if (schema.type === 'array') return [defaults(schema.items ?? {}, key, s)];
  if (schema.type === 'object') return Object.fromEntries(Object.entries(schema.properties ?? {}).map(([k, v]) => [k, defaults(v, k, s)]));
  if (schema.format === 'date') { const d = new Date(s.world.clock); if (key === 'departure') d.setUTCDate(d.getUTCDate() + 2); return d.toISOString().slice(0, 10); }
  const collections: Record<string, string> = { sku_id: 'skus', lot_id: 'lots', room_type_id: 'room_types', rate_plan_id: 'rate_plans', guest_id: 'guests', room_id: 'rooms', user_id: 'users', invoice_id: 'invoices', dependency_id: 'instances' };
  return collections[key] ? s.collections[collections[key]]?.[0]?.id ?? '' : '';
}

function Playground({ base, snapshot: s, scope, busy, mutate, replace, inspect }: { base: string; snapshot: DomainSnapshot; scope: string; busy: boolean; mutate: Mutate; replace: (s: DomainSnapshot) => void; inspect: Inspect }) {
  const [tools, setTools] = useState<Tool[]>([]); const [scopeField, setScopeField] = useState('scope_id');
  const [name, setName] = useState(''); const [actor, setActor] = useState(s.exploration_source?.identity ?? s.collections.users[0].id);
  const [scopeId, setScopeId] = useState(s.collections[scope][0].id); const [resource, setResource] = useState(s.collections[scope][0].id);
  const [params, setParams] = useState<Record<string, unknown>>({}); const [mode, setMode] = useState<'reference' | 'sandbox'>('reference'); const [result, setResult] = useState<ManualResult | null>(null);
  useEffect(() => { let active = true; void api<{ scope_field: string; tools: Tool[] }>(`${base}/tools`).then(d => { if (active) { setTools(d.tools); setScopeField(d.scope_field); choose(d.tools[0]); } }).catch(e => { void mutate(async () => { throw e; }); }); return () => { active = false; }; }, [base]);
  function choose(tool: Tool) { setName(tool.name); setParams(Object.fromEntries(Object.entries(tool.params.properties ?? {}).map(([k, v]) => [k, ['object', 'array'].includes(v.type ?? '') ? JSON.stringify(defaults(v, k, s), null, 2) : defaults(v, k, s)]))); setResult(null); }
  const tool = tools.find(t => t.name === name);
  const ids = Object.values(s.collections).flatMap(rows => rows.map(r => r.id));
  return <div className="playground-grid"><section className="panel operation-form"><div className="panel-heading"><div><h2>Explicit manual request</h2><p>每次提交使用真实领域 Engine，并返回独立 Oracle。</p></div><FlaskConical size={18} /></div><button className="text-button" disabled={busy} onClick={() => void mutate(async () => inspect('Scoped Customer observation · ' + actor, await api(`${base}/observation?identity=${encodeURIComponent(actor)}`)))}>Inspect this identity's Customer view<ArrowRight size={14} /></button><form onSubmit={e => { e.preventDefault(); void mutate(async () => {
    const typed = Object.fromEntries(Object.entries(params).map(([key, value]) => { const schema = tool?.params.properties?.[key]; return [key, ['object', 'array'].includes(schema?.type ?? '') ? JSON.parse(String(value)) : schema?.type === 'integer' ? Number(value) : value]; }));
    const output = await api<{ snapshot: DomainSnapshot; operation: ManualResult }>(`${base}/operations`, { actor_id: actor, operation: { name, [scopeField]: scopeId, resource_id: resource, params: typed }, mode, expected_state_hash: s.state_hash }); replace(output.snapshot); setResult(output.operation);
  }); }}><div className="form-step"><label>Acting identity<select aria-label="Acting identity" value={actor} onChange={e => setActor(e.target.value)}>{s.collections.users.map(r => <option key={r.id} value={r.id}>{String(r.name)} · {r.id}</option>)}</select></label><label>Business scope<select aria-label="Business scope" value={scopeId} onChange={e => setScopeId(e.target.value)}>{s.collections[scope].map(r => <option key={r.id} value={r.id}>{String(r.name)} · {r.id}</option>)}</select></label><label>Business operation<select aria-label="Business operation" value={name} onChange={e => choose(tools.find(t => t.name === e.target.value)!)}>{tools.map(t => <option key={t.name} value={t.name}>{t.name} · {t.query ? 'query' : 'write'}</option>)}</select></label><p className="field-help">{tool?.description}</p><label>Target resource ID<input aria-label="Target resource ID" required list="domain-resource-ids" value={resource} onChange={e => setResource(e.target.value)} /></label><datalist id="domain-resource-ids">{[...new Set(ids)].map(id => <option key={id} value={id} />)}</datalist></div><div className="parameter-box">{Object.entries(tool?.params.properties ?? {}).map(([key, schema]) => <label key={name + key}>{label(key)}{schema.enum ? <select aria-label={`Parameter ${key}`} value={String(params[key])} onChange={e => setParams(p => ({ ...p, [key]: schema.type === 'integer' ? Number(e.target.value) : e.target.value }))}>{schema.enum.map(v => <option key={String(v)} value={String(v)}>{String(v)}</option>)}</select> : schema.type === 'boolean' ? <select aria-label={`Parameter ${key}`} value={String(params[key])} onChange={e => setParams(p => ({ ...p, [key]: e.target.value === 'true' }))}><option value="false">false</option><option value="true">true</option></select> : ['array', 'object'].includes(schema.type ?? '') ? <textarea aria-label={`Parameter ${key}`} required value={String(params[key])} onChange={e => setParams(p => ({ ...p, [key]: e.target.value }))} rows={5} /> : <input aria-label={`Parameter ${key}`} required={tool?.params.required?.includes(key)} type={schema.type === 'integer' ? 'number' : schema.format === 'date' ? 'date' : 'text'} step={schema.type === 'integer' ? '1' : undefined} value={String(params[key] ?? '')} list={schema.type === 'string' ? 'domain-resource-ids' : undefined} onChange={e => setParams(p => ({ ...p, [key]: e.target.value }))} />}</label>)}</div><div className="domain-mode"><ModeSwitch mode={mode} setMode={setMode} /><p className="field-help">Reference 正确拒绝非法业务操作；Sandbox 可产生实际状态违规，Oracle 独立检查。</p></div><button className="button button-primary" disabled={busy || !tool}>Execute operation<ArrowRight size={15} /></button><button className="text-button" type="button" onClick={() => inspect('Operation parameter schema', tool)}>Inspect typed schema<Code2 size={14} /></button></form></section><section className="panel domain-result">{result ? <><div className="panel-heading"><h2>Operation result</h2><Badge tone={result.event.payload.result?.success ? 'blue' : 'amber'}>{result.event.payload.result?.success ? 'Executed' : 'Rejected'}</Badge></div><div data-testid="domain-operation-result">{(result.event.payload.result?.error_code ?? result.evaluation.reference_error_code) && <code className="error-code">{result.event.payload.result?.error_code ?? result.evaluation.reference_error_code}</code>}<div className="oracle-box"><h3>Independent Oracle</h3><p className="domain-classification">{result.evaluation.classification}</p><div className="verdict-grid"><div><span>Authorized manual request</span><strong>{result.evaluation.authorized ? 'Yes' : 'No'}</strong></div><div><span>Operation correct</span><strong>{result.evaluation.operation_correct === null ? 'Unknown' : result.evaluation.operation_correct ? 'Yes' : 'No'}</strong></div></div>{result.evaluation.violations.map(v => <code className="error-code" key={v}>{v}</code>)}</div><details open><summary>Tool / engine return data</summary><JsonView value={result.event.payload.result?.data} /></details><h3>Actual state changes <Badge>{result.changes.length}</Badge></h3><div className="diff-list">{result.changes.map(c => <div className="diff" key={c.path}><code>{c.path}</code><div><span className="diff-before">{text(c.before)}</span><ArrowRight size={14} /><span className="diff-after">{text(c.after)}</span></div></div>)}</div>{!result.changes.length && <p className="empty-small">No fields changed.</p>}<button className="button" onClick={() => inspect('Complete manual operation evidence', result)}>Inspect before / after / Oracle</button></div></> : <div className="result-empty"><ShieldCheck size={36} /><h3>State change. Independent verdict.</h3><p>执行后展示真实返回、状态差异与 Oracle 分类。</p></div>}</section></div>;
}

function Timeline({ base, snapshot: s, busy, mutate, inspect, download }: { base: string; snapshot: DomainSnapshot; busy: boolean; mutate: Mutate; inspect: Inspect; download: (kind: 'world' | 'trace') => Promise<void> }) {
  const [replay, setReplay] = useState<unknown>(null);
  return <><section className="panel"><div className="panel-heading"><div><h2>Manual execution trace</h2><p>{s.events.length} recorded events · {s.operations_used} actual operations</p></div><div className="toolbar-actions"><button className="button" disabled={busy} onClick={() => void mutate(() => download('trace'))}><ArrowDownToLine size={14} />Export Trace JSON</button><button className="button" disabled={busy} onClick={() => void mutate(async () => setReplay(await api(`${base}/replay`, {})))}><RefreshCw size={14} />Replay Trace</button></div></div>{s.events.map(event => { const verdict = s.evaluation.operations.find(r => r.event_id === event.id); return <button className="domain-event" key={event.id} onClick={() => void mutate(async () => inspect(event.id, await api(`${base}/events/${event.id}`)))}><code>{event.id}</code><Badge>{event.kind}</Badge><strong>{String(event.payload.operation?.name ?? '')}</strong><span>{verdict?.classification ?? 'Recorded provenance / authority'}</span><span>{event.after_hash ? short(event.after_hash) : ''}</span><ArrowRight size={14} /></button>; })}</section>{replay !== null && <div className="replay-banner"><strong>Recorded event replay</strong><JsonView value={replay} /></div>}<section className="panel"><h2>Current independent Oracle</h2><JsonView value={s.evaluation} /></section></>;
}
