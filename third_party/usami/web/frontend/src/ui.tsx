import { Check, ChevronRight, Copy, X } from 'lucide-react';
import { useEffect, useRef, useState, type ReactNode } from 'react';
import type { Json, Mode, OperationView } from './types';

export const roleLabel = (role: string) => ({ owner: 'Owner', billing_admin: 'Billing Admin', member: 'Member' }[role] ?? role);
export const operationLabel = (name: string) => ({ invite_member: 'Invite member', activate_member: 'Activate member', deactivate_member: 'Deactivate member', remove_member: 'Remove member', pay_invoice: 'Pay invoice', issue_renewal: 'Issue renewal invoice', renew_subscription: 'Renew subscription', reactivate_subscription: 'Reactivate subscription', expire_subscription: 'Expire subscription', change_plan: 'Change plan', change_seats: 'Change seats', cancel_renewal: 'Cancel renewal', refund_invoice: 'Refund invoice', change_member_role: 'Change member role', get_subscription: 'View subscription', get_invoice: 'View invoice', get_organization: 'View organization', list_members: 'List members' }[name] ?? name);
export const date = (value?: string | null) => value ? new Intl.DateTimeFormat('en-GB', { day: '2-digit', month: 'short', year: 'numeric', timeZone: 'UTC' }).format(new Date(value)) : '—';
export const amount = (cents: number) => (cents / 100).toFixed(2);
export const short = (value: string) => `${value.slice(0, 8)}…${value.slice(-6)}`;
export const boolLabel = (value: boolean | null) => value === null ? 'Unknown' : value ? 'Yes' : 'No';

export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: string }) { return <span className={`badge badge-${tone}`}>{children}</span>; }
export function Status({ status }: { status: string }) { return <Badge tone={status === 'active' || status === 'paid' || status === 'pass' ? 'green' : status === 'expired' || status === 'unpaid' || status === 'pending' || status === 'partially_refunded' || status === 'unknown' ? 'amber' : status === 'fail' || status === 'failed' ? 'red' : 'blue'}><span className="status-dot" />{status.charAt(0).toUpperCase() + status.slice(1)}</Badge>; }
export function Avatar({ name, index = 0 }: { name: string; index?: number }) { return <span className={`avatar avatar-${index % 5}`}>{name.split(' ').map(p => p[0]).join('').slice(0, 2)}</span>; }
export function JsonView({ value }: { value: Json }) { return <pre className="json-view">{JSON.stringify(value, null, 2)}</pre>; }
export function CopyValue({ value, label }: { value: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  return <button className="copy-value" title={value} onClick={() => { void navigator.clipboard.writeText(value).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); }); }}><code>{label ?? short(value)}</code>{copied ? <Check size={12} /> : <Copy size={12} />}</button>;
}
export function ModeSwitch({ mode, setMode, disabled }: { mode: Mode; setMode: (m: Mode) => void; disabled?: boolean }) { return <div className="segmented" aria-label="Execution mode">{(['reference', 'sandbox'] as const).map(m => <button key={m} type="button" disabled={disabled} aria-pressed={mode === m} className={mode === m ? `selected ${m}` : ''} onClick={() => setMode(m)}>{m === 'reference' ? 'Reference' : 'Sandbox'}</button>)}</div>; }

export function Inspector({ title, children, close }: { title: string; children: ReactNode; close: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const prior = document.activeElement as HTMLElement | null;
    const key = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close();
      if (e.key === 'Tab') {
        const elements = ref.current?.querySelectorAll<HTMLElement>('button, a, input, select, [tabindex="0"]');
        if (!elements?.length) return;
        const first = elements[0], last = elements[elements.length - 1];
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
      }
    };
    document.addEventListener('keydown', key); ref.current?.focus();
    return () => { document.removeEventListener('keydown', key); prior?.focus(); };
  }, [close]);
  return <div className="inspector-backdrop" onClick={close}><div ref={ref} className="inspector" role="dialog" aria-modal="true" aria-label={title} tabIndex={-1} onClick={e => e.stopPropagation()}><div className="inspector-head"><div><span className="eyebrow">ENTITY INSPECTOR</span><h2>{title}</h2></div><button className="icon-button" aria-label="Close inspector" onClick={close}><X size={20} /></button></div><div className="inspector-body">{children}</div></div></div>;
}

function valueText(value: Json) { if (value === null) return 'null'; if (typeof value === 'object') return JSON.stringify(value); return String(value); }
export function ResultView({ result, showEvent }: { result: OperationView; showEvent: (id: string) => void }) {
  const [tab, setTab] = useState('changes');
  const outcome = result.event.payload.result!;
  const evaluation = result.evaluation;
  return <div className="result-content" data-testid="operation-result">
    <div className="result-heading"><h3>Operation result</h3><Badge tone={outcome.success ? 'blue' : 'amber'}>{outcome.success ? 'Executed' : 'Rejected'}</Badge></div>
    {(outcome.error_code || result.reference_error_code) && <code className="error-code">{outcome.error_code ?? result.reference_error_code}</code>}
    <p className="result-reason">{result.reason}</p>
    <div className="oracle-box"><div className="oracle-label"><span className="oracle-mark">✦</span><h3>Independent Oracle</h3><Badge tone={evaluation.operation_correct === null ? 'amber' : evaluation.operation_correct ? 'green' : 'red'}>{evaluation.operation_correct === null ? 'Unknown' : evaluation.operation_correct ? 'Correct' : 'Violation'}</Badge></div><div className="verdict-grid"><div><span>Valid request</span><strong data-testid="valid-request">{boolLabel(evaluation.valid_request)}</strong></div><div><span>Operation correct</span><strong data-testid="operation-correct">{boolLabel(evaluation.operation_correct)}</strong></div></div>
      {evaluation.violations.length > 0 && <div className="violation-list">{evaluation.violations.map((v, i) => <div key={`${v.code}${i}`}><code>{v.code}</code><p>{v.detail}</p><small>{v.state_path ?? v.entity_id}</small></div>)}</div>}
      <div className="evidence"><span>Evidence</span>{evaluation.evidence_event_ids.map(id => <button key={id} onClick={() => showEvent(id)}>{id}<ChevronRight size={12} /></button>)}</div>
    </div>
    <details className="event-json"><summary>Tool / engine return data</summary><JsonView value={outcome.data} /></details>
    <div className="tabs" role="tablist" aria-label="State changes">{['changes', 'before', 'after', 'request'].map(t => <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? 'active' : ''} onClick={() => setTab(t)}>{t === 'changes' ? `Changes (${result.changes.length})` : t.charAt(0).toUpperCase() + t.slice(1)}</button>)}</div>
    {tab === 'changes' ? result.changes.length === 0 ? <div className="empty-small"><Check size={16} />No fields changed.</div> : <div className="diff-list">{result.changes.map(change => <div className="diff" key={change.path}><code>{change.path}</code><div><span className="diff-before">{valueText(change.before)}</span><ChevronRight size={14} /><span className="diff-after">{valueText(change.after)}</span></div></div>)}</div> : <JsonView value={tab === 'before' ? result.event.before : tab === 'after' ? result.event.after : result.event.payload} />}
  </div>;
}
