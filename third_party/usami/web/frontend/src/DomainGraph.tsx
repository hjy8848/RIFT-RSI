import { Background, Controls, MiniMap, Position, ReactFlow, ReactFlowProvider, useEdgesState, useNodesState, type Node } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useEffect, useState } from 'react';
import { Network } from 'lucide-react';
import { api } from './api';
import { Badge } from './ui';
import type { DomainSnapshot, Entity } from './domainTypes';
import type { Inspect } from './DomainExplorer';

interface Graph { state_hash: string; nodes: { id: string; kind: string; label: string; entity: Entity }[]; edges: { id: string; source: string; target: string; kind: string; label: string; entity: Entity }[] }
const colors = ['#4774e6', '#7b67bd', '#20a49d', '#bf954d', '#c16d91', '#6f8ca5'];
function Canvas({ graph, inspect }: { graph: Graph; inspect: Inspect }) {
  const kinds = [...new Set(graph.nodes.map(n => n.kind))];
  const initial: Node[] = graph.nodes.map(n => ({ id: n.id, type: 'default', sourcePosition: Position.Right, targetPosition: Position.Left, position: { x: kinds.indexOf(n.kind) * 290, y: graph.nodes.filter(v => v.kind === n.kind).findIndex(v => v.id === n.id) * 95 }, style: { width: 230, borderColor: colors[kinds.indexOf(n.kind) % colors.length], borderRadius: 9 }, data: { label: <div className="domain-graph-node"><small>{n.kind}</small><strong>{n.label}</strong><code>{n.entity.id}</code></div>, raw: n } }));
  const [nodes, , onNodesChange] = useNodesState(initial);
  const [edges, , onEdgesChange] = useEdgesState(graph.edges.map(e => ({ ...e, data: { raw: e }, style: { stroke: e.kind === 'dependency' ? '#8c70c4' : '#b3c4dc' }, labelStyle: { fontSize: 10 } })));
  return <ReactFlow nodes={nodes} edges={edges} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} nodesConnectable={false} deleteKeyCode={null} minZoom={0.05} maxZoom={3} fitView onNodeClick={(_, n) => inspect(n.id, (n.data.raw as Graph['nodes'][0]).entity)} onEdgeClick={(_, e) => inspect(e.label as string, e.data?.raw)}><Background color="#d9e0ed" gap={20} /><Controls showInteractive={false} /><MiniMap pannable zoomable /></ReactFlow>;
}
export default function DomainGraph({ base, snapshot, scopes, inspect, onError }: { base: string; snapshot: DomainSnapshot; scopes: Entity[]; inspect: Inspect; onError: (message: string) => void }) {
  const [scope, setScope] = useState(''); const [calendar, setCalendar] = useState(false); const [graph, setGraph] = useState<Graph | null>(null);
  useEffect(() => { let active = true; setGraph(null); void api<Graph>(`${base}/world/graph?scope=${encodeURIComponent(scope)}&calendar=${calendar}`).then(d => { if (active) setGraph(d); }).catch(e => { if (active) onError(String(e)); }); return () => { active = false; }; }, [base, snapshot.state_hash, scope, calendar]);
  return <section className="panel graph-panel"><div className="graph-toolbar"><div><Network size={18} /><strong>Actual entity relationships</strong><Badge>{graph?.nodes.length ?? '…'} nodes</Badge><Badge>{graph?.edges.length ?? '…'} edges</Badge></div><label>Focus scope<select aria-label="Graph scope" value={scope} onChange={e => setScope(e.target.value)}><option value="">All scopes</option>{scopes.map(s => <option key={s.id} value={s.id}>{String(s.name ?? s.id)}</option>)}</select></label>{snapshot.domain === 'stayops' && <label className="domain-checkbox"><input type="checkbox" checked={calendar} onChange={e => setCalendar(e.target.checked)} />Include per-night calendar</label>}</div><div className="graph-legend">{[...new Set(graph?.nodes.map(n => n.kind) ?? [])].map((k, i) => <span key={k}><i style={{ background: colors[i % colors.length] }} />{k}</span>)}</div><div className="graph-canvas" data-testid="world-graph">{graph ? <ReactFlowProvider key={`${graph.state_hash}:${scope}:${calendar}`}><Canvas graph={graph} inspect={inspect} /></ReactFlowProvider> : <div className="graph-loading">Loading live relationships…</div>}</div><div className="graph-bottom"><span>Drag · zoom · inspect actual entities and references</span><span>Relationships from World data</span></div></section>;
}
