import { Background, BackgroundVariant, Controls, MiniMap, ReactFlow, ReactFlowProvider, useEdgesState, useNodesState, Position, type Node, type Edge } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Building2, Layers3, Network, Users, Wallet } from 'lucide-react';
import { useEffect, useState, type ReactNode } from 'react';
import { api } from './api';
import { Badge, JsonView } from './ui';
import type { GraphData, GraphEdge, GraphNode, Json, Snapshot } from './types';

type EntityNode = Node<{ label: ReactNode; raw: GraphNode }>;
type EntityEdge = Edge<{ raw: GraphEdge }>;
const icons: Record<string, typeof Users> = { user: Users, organization: Building2, subscription: Layers3, invoice: Wallet, payment: Wallet, refund: Wallet };
const colors: Record<string, string> = { user: '#20a49d', organization: '#4772df', subscription: '#9470cb', invoice: '#bf954d', payment: '#20a49d', refund: '#d4819e' };

function Canvas({ data, inspect }: { data: GraphData; inspect: (title: string, data: Json, extra?: ReactNode) => void }) {
  const orgs = data.nodes.filter(n => n.kind === 'organization');
  const initial: EntityNode[] = data.nodes.map(n => {
    const Icon = icons[n.kind];
    const orgIndex = n.kind === 'organization' ? orgs.findIndex(o => o.id === n.id) : orgs.findIndex(o => o.id === n.entity.org_id);
    const base = Math.max(0, orgIndex) * 290;
    const index = data.nodes.filter(p => p.kind === n.kind).findIndex(p => p.id === n.id);
    const localIndex = data.nodes.filter(p => p.kind === n.kind && p.entity.org_id === n.entity.org_id).findIndex(p => p.id === n.id);
    return { id: n.id, type: 'default', className: `entity-node entity-${n.kind}`, sourcePosition: Position.Right, targetPosition: Position.Left,
      position: n.kind === 'user' ? { x: 0, y: index * 86 + 20 } : n.kind === 'organization' ? { x: 360, y: base + 95 } : n.kind === 'subscription' ? { x: 710, y: base + 90 } : { x: n.kind === 'invoice' ? 1010 : n.kind === 'payment' ? 1350 : 1670, y: base + localIndex * 62 },
      data: { raw: n, label: <div className="graph-node-label"><span className="graph-node-icon"><Icon size={16} /></span><div><strong>{n.kind === 'invoice' ? String(n.entity.status).charAt(0).toUpperCase() + String(n.entity.status).slice(1) + ' invoice' : n.label}</strong><small>{n.id}</small></div>{n.kind === 'subscription' && <span className="graph-node-status">{String(n.entity.status)}</span>}</div> },
      style: { width: n.kind === 'invoice' ? 210 : 205 },
    };
  });
  const initialEdges: EntityEdge[] = data.edges.map(e => ({ ...e, type: 'default', data: { raw: e }, animated: false,
    style: { stroke: e.kind === 'membership' ? e.entity?.role === 'owner' ? '#9581c9' : e.entity?.role === 'billing_admin' ? '#668bdb' : '#acb5c5' : '#b4c4dc', strokeWidth: 1.3, strokeDasharray: e.entity && e.entity.status !== 'active' ? '4 4' : undefined },
    labelStyle: { fontSize: 10, fill: '#5f6c80', fontWeight: 500 }, labelBgStyle: { fill: '#fbfcff', fillOpacity: 0.96 }, labelBgPadding: [5, 3], labelBgBorderRadius: 4,
  }));
  const [nodes, , onNodesChange] = useNodesState<EntityNode>(initial);
  const [edges, , onEdgesChange] = useEdgesState<EntityEdge>(initialEdges);
  return <ReactFlow nodes={nodes} edges={edges} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} nodesConnectable={false} deleteKeyCode={null}
    onNodeClick={(_, node) => inspect(node.data.raw.label, node.data.raw.entity, node.data.raw.memberships && <><h3>Real memberships</h3><JsonView value={node.data.raw.memberships} /></>)}
    onEdgeClick={(_, edge) => { if (edge.data?.raw.entity) inspect(`Membership · ${edge.data.raw.label}`, edge.data.raw.entity); }}
    fitView fitViewOptions={{ padding: 0.12 }} minZoom={0.12} maxZoom={2.5} aria-label="Business world relationship graph">
    <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#d9e0ed" /><Controls showInteractive={false} /><MiniMap pannable zoomable nodeColor={node => colors[(node.data.raw as GraphNode).kind] ?? '#ddd'} maskColor="rgba(233,239,249,.65)" />
  </ReactFlow>;
}

export default function WorldGraph({ snapshot: s, inspect, onError }: { snapshot: Snapshot; inspect: (title: string, data: Json, extra?: ReactNode) => void; onError: (e: string) => void }) {
  const [filter, setFilter] = useState('');
  const [data, setData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(false);
  useEffect(() => { let alive = true; setLoading(true); void api<GraphData>(`/world/graph${filter ? `?org_id=${encodeURIComponent(filter)}` : ''}`).then(d => { if (alive) setData(d); }).catch(e => { if (alive) { setData(null); onError(String(e)); } }).finally(() => { if (alive) setLoading(false); }); return () => { alive = false; }; }, [s.state_hash, filter]);
  return <section className="panel graph-panel"><div className="graph-toolbar"><div><Network size={18} /><strong>Live entity relationships</strong><Badge>{data?.nodes.length ?? '…'} nodes</Badge><Badge>{data?.edges.length ?? '…'} edges</Badge></div><label>Focus organization<select aria-label="Graph organization" value={filter} onChange={e => setFilter(e.target.value)}><option value="">All organizations</option>{s.organizations.map(o => <option key={o.id} value={o.id}>{o.name}</option>)}</select></label></div><div className="graph-legend">{Object.entries(colors).map(([kind, color]) => <span key={kind}><i style={{ background: color }} />{kind.charAt(0).toUpperCase() + kind.slice(1)}</span>)}<span className="graph-legend-help">Membership edges carry real roles · dashed = pending / inactive</span></div><div className="graph-canvas" data-testid="world-graph">{data && <ReactFlowProvider key={`${data.state_hash}:${data.nodes.map(n => n.id).join(",")}`}><Canvas data={data} inspect={inspect} /></ReactFlowProvider>}{loading && <div className="graph-loading">Loading Python world relationships…</div>}</div><div className="graph-bottom"><span>Drag nodes · scroll to zoom · click any entity or membership edge to inspect</span><span>Read-only relationships</span></div></section>;
}
