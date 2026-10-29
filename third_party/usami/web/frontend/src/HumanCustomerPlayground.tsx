import { ArrowLeft, ArrowRight, Check, CheckCircle2, CircleHelp, Download, Loader2, MessageCircle, RotateCcw, ShieldCheck, StopCircle, Wrench } from 'lucide-react';
import { useCallback, useEffect, useMemo, useState, type FormEvent } from 'react';
import { api } from './api';
import './HumanCustomerPlayground.css';
import './human-goal-note.css';

type Scenario = {
  scenario_id: string; title: string; description: string; customer_request: string;
  customer: { id: string; name: string };
  orders: Array<{ order_id: string; delivered_at: string; paid_cents: number; lines: Array<{
    line_id: string; product: string; attributes: Record<string, string>; quantity: number;
    paid_cents: number; issue: string;
  }> }>;
  world_version: string; initial_world_hash: string; world_build_verified: boolean;
  task_certificate_status: string; task_certificate_verified: boolean;
};
type TimelineItem = { kind: string; text?: string; source?: string; intent?: string; success?: boolean;
  error_code?: string | null; operation?: { name?: string }; offer?: Offer | null };
type Offer = { remedy?: string; quantity?: number; refund_cents?: number; extra_charge_cents?: number;
  promised_arrival?: string | null; return_required?: boolean; pickup_slot?: string | null; operation?: string };
type Session = { session_id: string; status: 'starting' | 'service_running' | 'awaiting_customer' | 'cancelling' | 'completed' | 'failed' | 'cancelled';
  session_version: number; scenario: Scenario; timeline: TimelineItem[]; pending_turn: null | { turn_id: string; session_version: number; offer: Offer | null };
  model_name: string | null; model_calls_limit: number; tool_calls: number; error: string | null;
  study_type: string; customer_source: string };
type Outcome = {
  business_goal_success: boolean | null; process_errors: { status: string; evidence_refs: string[] };
  safety_violations: { status: string; evidence_refs: string[] };
  interaction_reliability: { status: string; evidence_refs: string[] };
  strict_task_success: boolean | null; final_goal_reached: boolean | null; evaluation_status: string;
};
type Report = { episode: { status: string; terminal_reason: string; tool_executions: unknown[]; events: unknown[]; model_calls: unknown[]; final_world: { settlements: Record<string, unknown> } };
  outcome: Outcome; replay: { verified: boolean; event_count: number; operation_count: number; final_world_hash: string; errors: string[] };
  costs: { customer_calls: number; service_calls: number; tool_calls: number; known_total_tokens: number | null; unknown_usage_calls: number };
  benchmark_eligible: boolean; training_evidence_eligible: boolean; persisted_at: string | null; model_name: string | null };
type Config = { configured: boolean; model: string | null; service: string; credential_source_configured: boolean; message: string };

const labels: Record<string, string> = {
  starting: '正在启动', service_running: '客服正在处理', awaiting_customer: '等待你回复',
  cancelling: '正在取消并封存', completed: '对话已结束', failed: '运行异常', cancelled: '已取消',
  open_remedy_case: '创建售后案件', verify_submitted_evidence: '核验证据',
  prepare_remedy_quote: '准备处理报价', request_policy_review: '申请政策复核',
  commit_accepted_quote: '提交已同意的方案', release_accepted_refund: '释放退款',
};
const SESSION_STORAGE_KEY = 'usami-human-session-id';
const money = (cents?: number | null) => cents == null ? '—' : `¥${(cents / 100).toFixed(2)}`;
const display = (status: string) => labels[status] ?? status;

export default function HumanCustomerPlayground() {
  const entryParams = useMemo(() => new URLSearchParams(location.search), []);
  const researchRunId = entryParams.get('researchRunId');
  const comparisonArm = entryParams.get('arm') === 'S1' ? 'S1' : entryParams.get('arm') === 'S0' ? 'S0' : null;
  const skillSnapshotHash = entryParams.get('skillSnapshotHash');
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [config, setConfig] = useState<Config | null>(null);
  const [scenarioId, setScenarioId] = useState('');
  const [session, setSession] = useState<Session | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [reportError, setReportError] = useState('');
  const scenario = scenarios.find(row => row.scenario_id === scenarioId) ?? scenarios[0];

  const loadCatalog = useCallback(async () => {
    try {
      const [catalog, model] = await Promise.all([
        api<{ scenarios: Scenario[] }>('/human/scenarios'),
        api<Config>('/human/config'),
      ]);
      setScenarios(catalog.scenarios);
      const requestedScenario = entryParams.get('scenario');
      setScenarioId(current => current || (catalog.scenarios.some(row => row.scenario_id === requestedScenario)
        ? requestedScenario! : catalog.scenarios[0]?.scenario_id || ''));
      setConfig(model);
    } catch (cause) { setError(String(cause)); }
  }, [entryParams]);
  useEffect(() => {
    void loadCatalog();
    const savedSession = sessionStorage.getItem(SESSION_STORAGE_KEY);
    if (savedSession) {
      void (async () => {
        try {
          const restored = await api<Session>(`/human/sessions/${savedSession}`);
          setSession(restored);
          if (['completed', 'failed', 'cancelled'].includes(restored.status)) {
            const finished = await api<Report>(`/human/sessions/${savedSession}/report`);
            setReport(finished);
          }
        } catch {
          // The backend's HttpOnly owner cookie is authoritative. A stale local
          // session reference is discarded; it can never authorize a session.
          sessionStorage.removeItem(SESSION_STORAGE_KEY);
        }
      })();
    }
  }, [loadCatalog]);

  const poll = useCallback(async (id: string) => {
    try {
      const latest = await api<Session>(`/human/sessions/${id}`);
      setSession(latest);
      if (['completed', 'failed', 'cancelled'].includes(latest.status)) {
        try {
          const finished = await api<Report>(`/human/sessions/${id}/report`);
          setReport(finished); setReportError('');
        } catch (cause) {
          if (!String(cause).includes('尚未结束')) setReportError(String(cause));
        }
      }
    } catch (cause) { setError(String(cause)); }
  }, []);
  useEffect(() => {
    if (!session || ['completed', 'failed', 'cancelled'].includes(session.status)) return;
    const timer = window.setInterval(() => { void poll(session.session_id); }, 700);
    return () => window.clearInterval(timer);
  }, [session?.session_id, session?.status, poll]);

  async function begin() {
    if (!scenario) return;
    setBusy(true); setError(''); setReport(null); setReportError('');
    try {
      const created = await api<Session>('/human/sessions', { scenario_id: scenario.scenario_id,
        research_run_id: researchRunId, comparison_arm: comparisonArm,
        service_skill_snapshot_hash: comparisonArm === 'S1' ? skillSnapshotHash : null });
      setSession(created);
      sessionStorage.setItem(SESSION_STORAGE_KEY, created.session_id);
    } catch (cause) { setError(String(cause)); }
    finally { setBusy(false); }
  }
  async function submit(intent: 'message' | 'accept_offer' | 'reject_offer' | 'finish', text = message) {
    if (!session?.pending_turn) return;
    setBusy(true); setError('');
    try {
      const next = await api<Session>(`/human/sessions/${session.session_id}/turn`, {
        ...session.pending_turn, intent, message: text,
      });
      setSession(next); setMessage('');
    } catch (cause) { setError(String(cause)); await poll(session.session_id); }
    finally { setBusy(false); }
  }
  async function cancel() {
    if (!session) return;
    setBusy(true);
    try { setSession(await api<Session>(`/human/sessions/${session.session_id}/cancel`, {})); }
    catch (cause) { setError(String(cause)); }
    finally { setBusy(false); }
  }
  async function reset() {
    const current = session?.scenario.scenario_id ?? scenario?.scenario_id;
    setSession(null); setReport(null); setError(''); setReportError(''); setMessage('');
    sessionStorage.removeItem(SESSION_STORAGE_KEY);
    if (current) {
      setBusy(true);
      try {
        const created = await api<Session>('/human/sessions', { scenario_id: current,
          research_run_id: researchRunId, comparison_arm: comparisonArm,
          service_skill_snapshot_hash: comparisonArm === 'S1' ? skillSnapshotHash : null });
        setSession(created);
        sessionStorage.setItem(SESSION_STORAGE_KEY, created.session_id);
      }
      catch (cause) { setError(String(cause)); }
      finally { setBusy(false); }
    }
  }
  async function exportRecord() {
    if (!session) return;
    try {
      const response = await fetch(`/api/human/sessions/${session.session_id}/export`, { credentials: 'same-origin' });
      if (!response.ok) throw new Error(`导出失败：HTTP ${response.status}`);
      const url = URL.createObjectURL(await response.blob());
      const anchor = document.createElement('a'); anchor.href = url;
      anchor.download = `usami-human-episode-${session.session_id.slice(0, 12)}.json`; anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (cause) { setError(String(cause)); }
  }
  function sendMessage(event: FormEvent) { event.preventDefault(); if (message.trim()) void submit('message'); }

  const activity = useMemo(() => [...(session?.timeline ?? [])].reverse(), [session?.timeline]);
  const waiting = session?.status === 'awaiting_customer' && !!session.pending_turn;
  const finished = session && ['completed', 'failed', 'cancelled'].includes(session.status);
  const activeOffer = session?.pending_turn?.offer ?? null;

  return <div className="human-app" data-testid="human-playground">
    <header className="human-topbar"><a href="/?" className="human-back"><ArrowLeft size={16} />World Explorer</a><a href="/?" className="human-wordmark">USAMI <span>2.0</span></a><div className="human-mode-tag"><span />真人顾客实验 · Reference</div></header>
    <main className="human-main">
      <div className="human-heading"><div><span className="human-kicker">QUALITATIVE MANUAL PILOT</span><h1>真人顾客实验台</h1><p>你亲自扮演顾客，真实客服通过业务工具处理已认证的售后请求。</p></div>{session && <span className={`human-status status-${session.status}`} role="status"><span />{display(session.status)}</span>}</div>
      {error && <div className="human-error" role="alert"><CircleHelp size={17} />{error}<button onClick={() => setError('')}>关闭</button></div>}
      {!session ? <>
        <section className="human-intro"><div><ShieldCheck size={18} /><strong>场景和目标已由 Usami 独立认证</strong><span>你可以自由回复、追问或拒绝报价。服务端保留固定身份、订单和评分目标。</span></div><div className={`human-config ${config?.configured ? 'ready' : 'missing'}`} data-testid="model-config">{config?.configured ? <><CheckCircle2 size={16} /><span>真实 Service 已配置</span><small>{config.model}</small></> : <><CircleHelp size={16} /><span>尚未检测到模型配置</span><small>请在后端设置 USAMI_LLM_MODEL / USAMI_LLM_BASE_URL / 凭据</small></>}</div></section>
        <section className="human-scenario-grid" aria-label="已认证业务场景">
          {scenarios.map(row => <button key={row.scenario_id} className={`human-scenario-card ${row.scenario_id === scenarioId ? 'selected' : ''}`} onClick={() => setScenarioId(row.scenario_id)}>
            <div className="scenario-card-top"><span className="scenario-glyph"><MessageCircle size={18} /></span><span className="certified-pill"><Check size={12} />已认证</span></div>
            <h2>{row.title}</h2><p>{row.description}</p><blockquote>“{row.customer_request}”</blockquote>
            <div className="scenario-card-meta"><span>{row.orders.length} 张订单</span><span>{row.orders.reduce((sum, order) => sum + order.lines.length, 0)} 件商品</span><span>{row.world_version}</span></div>
          </button>)}
        </section>
        {scenario && <section className="human-scene-preview panel">
          <div className="scene-preview-heading"><div><span className="human-kicker">YOUR CUSTOMER VIEW</span><h2>{scenario.customer.name} · 订单信息</h2></div><div className="scene-verified"><CheckCircle2 size={15} />Reference 任务可达性已证明</div></div>
          <div className="customer-orders">{scenario.orders.map(order => <article className="customer-order" key={order.order_id}><div className="order-line"><div><strong>订单</strong><code>{order.order_id}</code></div><span>已送达 · {new Date(order.delivered_at).toLocaleDateString()}</span></div>{order.lines.map(line => <div className="customer-product" key={line.line_id}><div className="product-mark">{line.product.slice(0, 1).toUpperCase()}</div><div className="product-info"><strong>{line.product}</strong><span>{line.issue === 'damaged' ? '顾客报告：商品损坏' : line.issue} · 数量 {line.quantity}</span></div><strong className="product-price">{money(line.paid_cents)}</strong></div>)}</article>)}</div>
          <div className="initial-request"><span>初始顾客请求</span><p>{scenario.customer_request}</p></div>
          <div className="scenario-proof-line"><code>world {scenario.initial_world_hash.slice(0, 16)}…</code><span>·</span><code>certificate {scenario.task_certificate_status}</code></div>
        </section>}
        <div className="human-start-row"><p>当前第一版只有“损坏商品退款”场景；换货选择场景后续再加入。</p><button className="human-primary" disabled={!scenario || !config?.configured || busy} onClick={() => void begin()} data-testid="start-human-episode">{busy ? <Loader2 className="human-spin" size={17} /> : <ArrowRight size={17} />}{busy ? '正在启动…' : '开始真人体验'}</button></div>
        {!config?.configured && <p className="human-config-help">未配置真实模型时不会降级成脚本客服。模型端点和 API Key 只在 Python 后端读取。</p>}
      </> : <>
        <section className="human-episode-layout">
          <aside className="human-order-sidebar panel">
            <div className="side-section-title"><span className="human-kicker">CUSTOMER</span><h2>{session.scenario.customer.name}</h2></div>
            <div className="side-request"><span>你的原始需求</span><p>{session.scenario.customer_request}</p></div>
            <div className="side-orders"><span className="human-kicker">YOUR ORDER</span>{session.scenario.orders.map(order => <div className="side-order" key={order.order_id}><strong>订单</strong><code>{order.order_id}</code>{order.lines.map(line => <div className="side-product" key={line.line_id}><span className="product-mark">{line.product.slice(0, 1).toUpperCase()}</span><div><strong>{line.product}</strong><small>{line.issue} · {money(line.paid_cents)}</small></div></div>)}</div>)}</div>
            <div className="privacy-note"><ShieldCheck size={15} /><span>服务只收到公开需求和工具返回，不会看到隐藏目标、证书见证或 Oracle 答案。</span></div>
          </aside>
          <section className="human-chat panel" aria-label="客服对话">
            <div className="chat-heading"><div><span className="service-avatar">U</span><div><strong>Usami Service</strong><small>{session.model_name || '真实模型'} · commerce-2.0.0</small></div></div><span className="reference-chip">REFERENCE MODE</span></div>
            <div className="chat-feed" data-testid="chat-feed" aria-live="polite">
              <div className="chat-day-marker"><span>冻结场景开始</span></div>
              {activity.filter(item => item.kind !== 'progress').map((item, index) => item.kind === 'tool' ? <div key={`tool-${index}`} className="tool-activity"><span className="tool-activity-icon"><Wrench size={13} /></span><span>{item.text || '客服执行了业务工具'}</span><small>{item.success ? '已返回' : item.error_code || '完成'}</small></div> : item.kind === 'service' ? <div key={`service-${index}`} className="chat-row service-row"><span className="service-avatar small-avatar">U</span><div className="message-stack"><div className="chat-bubble service-bubble">{item.text}</div>{item.offer && <OfferCard offer={item.offer} />}</div></div> : item.kind === 'customer' ? <div key={`customer-${index}`} className={`chat-row customer-row ${item.source === 'frozen_request' ? 'frozen-row' : ''}`}><div className="message-stack"><span className="message-source">{item.source === 'frozen_request' ? '你的初始请求' : item.intent === 'accept_offer' ? '你 · 明确接受报价' : item.intent === 'reject_offer' ? '你 · 明确拒绝报价' : '你'}</span><div className="chat-bubble customer-bubble">{item.text}</div></div><span className="customer-avatar">你</span></div> : null)}
              {session.status === 'service_running' && <div className="service-thinking" role="status"><Loader2 className="human-spin" size={15} /><span>{session.timeline.at(-1)?.kind === 'progress' ? session.timeline.at(-1)?.text : '客服正在调用业务工具或准备回复…'}</span></div>}
              {session.status === 'starting' && <div className="service-thinking" role="status"><Loader2 className="human-spin" size={15} /><span>正在创建独立世界并启动真实 Service…</span></div>}
              {session.status === 'cancelling' && <div className="service-thinking" role="status"><Loader2 className="human-spin" size={15} /><span>正在等待当前模型调用结束，并封存可回放的未完成记录…</span></div>}
              {session.status === 'cancelled' && <div className="chat-ended">本次体验已取消。未完成的目标不会作为 Service 失败评分。</div>}
              {session.status === 'failed' && <div className="chat-ended error-ended">本次运行未正常完成：{session.error || '请查看研究者报告。'}</div>}
              {session.status === 'completed' && <div className="chat-ended">对话正常结束。业务目标、过程错误、安全和 Replay 分开呈现。</div>}
            </div>
            {waiting && <form className="human-composer" onSubmit={sendMessage}>
              {activeOffer && <div className="offer-actions"><div><strong>请决定是否接受这份实际报价</strong><span>点击接受才会写入精确报价授权；普通文字不会产生成交授权。</span></div><div><button type="button" className="offer-accept" disabled={busy} onClick={() => void submit('accept_offer')}><Check size={15} />接受报价</button><button type="button" className="offer-reject" disabled={busy} onClick={() => void submit('reject_offer')}><StopCircle size={15} />拒绝</button></div></div>}
              <label htmlFor="human-message">你的回复</label><div className="composer-line"><textarea id="human-message" data-testid="human-message" value={message} onChange={event => setMessage(event.target.value)} placeholder={activeOffer ? '也可以先追问或说明你改变了主意；这不会确认报价。' : '输入你的回复、问题或补充要求…'} maxLength={2000} rows={3} /><button className="send-human" type="submit" disabled={busy || !message.trim()}>{busy ? <Loader2 className="human-spin" size={16} /> : <ArrowRight size={17} />}发送</button></div><p className="goal-boundary-note" data-testid="goal-boundary-note">本次评分只针对上方的初始认证需求。你可以提出新需求，但它会留在对话记录中，不会自动加入评分目标。</p><div className="composer-footer"><span>最多 2,000 字 · 只有明确的“接受报价”按钮会创建交易同意。</span><button type="button" className="finish-button" disabled={busy} onClick={() => void submit('finish', message.trim() || '我结束本次沟通。')}>结束对话</button></div>
            </form>}
            {!waiting && !finished && <div className="chat-wait-footer"><Loader2 className="human-spin" size={14} />{session.status === 'cancelling' ? '正在封存取消记录…' : session.status === 'service_running' ? 'Service 正在工作。工具调用由后端自动执行。' : '正在准备对话…'}<button disabled={busy || session.status === 'cancelling'} onClick={() => void cancel()}>{session.status === 'cancelling' ? '正在取消…' : '取消本次实验'}</button></div>}
          </section>
          <aside className="human-run-sidebar panel">
            <div className="run-state-card"><span className="human-kicker">RUN STATUS</span><strong>{display(session.status)}</strong><p>{session.status === 'awaiting_customer' ? '客服已停下，等待你的下一步。' : session.status === 'service_running' ? '后端正在调用真实 Service 和业务引擎。' : '本次运行使用独立冻结世界。'}</p></div>
            <div className="run-metric"><span>业务工具调用</span><strong>{session.tool_calls}</strong></div><div className="run-metric"><span>模型调用预算</span><strong>{session.model_calls_limit}</strong></div>
            <div className="run-integrity"><ShieldCheck size={16} /><div><strong>独立审计</strong><span>真实 Engine · Oracle · Replay</span></div></div>
            {finished && <button className="reset-run-button" onClick={() => void reset()}><RotateCcw size={15} />重置并从相同场景重试</button>}
          </aside>
        </section>
        {finished && <section className="human-report panel" data-testid="human-report">
          <div className="report-heading"><div><span className="human-kicker">RESEARCHER VIEW</span><h2>本次实验记录</h2><p>人工探索记录不会进入正式 Benchmark 或 Skill 训练证据。</p></div><button className="export-button" disabled={!report} onClick={() => void exportRecord()}><Download size={15} />导出完整 Episode</button></div>
          {reportError && <div className="report-note">{reportError}</div>}
          {report ? <><div className="score-grid">
            <ScoreCard label="业务目标完成" value={resultLabel(report.outcome.business_goal_success)} detail={resultLabel(report.outcome.final_goal_reached)} />
            <ScoreCard label="过程错误" value={report.outcome.process_errors.status} detail={`${report.outcome.process_errors.evidence_refs.length} 条证据`} />
            <ScoreCard label="安全违规" value={report.outcome.safety_violations.status} detail={`${report.outcome.safety_violations.evidence_refs.length} 条证据`} />
            <ScoreCard label="交互可靠性" value={report.outcome.interaction_reliability.status} detail={report.outcome.evaluation_status} />
            <ScoreCard label="严格任务成功" value={resultLabel(report.outcome.strict_task_success)} detail="独立综合判定" />
            <ScoreCard label="离线 Replay" value={report.replay.verified ? '通过' : '未通过'} detail={`${report.replay.event_count} events · ${report.replay.operation_count} tools`} />
          </div><div className="report-meta"><span>Service：{report.model_name || session.model_name || '未知'}</span><span>Service 调用：{report.costs.service_calls}</span><span>工具调用：{report.costs.tool_calls}</span><span>已知 Token：{report.costs.known_total_tokens ?? '未报告'}</span><span>记录标签：human · qualitative_manual_pilot</span></div>
            <details className="research-details"><summary>查看实际工具、Oracle、EpisodeRecord 与最终世界</summary><div className="research-columns"><div><h3>工具执行与 Oracle</h3>{(report.episode.tool_executions as Array<{operation: unknown; result: unknown; oracle: unknown; before_state_hash: string; after_state_hash: string}>).map((row, index) => <details className="tool-record" key={index}><summary>工具 {index + 1} · {String((row.operation as {name?: string})?.name || 'operation')}</summary><pre>{JSON.stringify(row, null, 2)}</pre></details>)}</div><div><h3>完整 Episode / Outcome / Replay</h3><pre>{JSON.stringify({ status: report.episode.status, terminal_reason: report.episode.terminal_reason, outcome: report.outcome, replay: report.replay, model_calls: report.episode.model_calls, final_settlements: report.episode.final_world.settlements, persisted_at: report.persisted_at, benchmark_eligible: report.benchmark_eligible, training_evidence_eligible: report.training_evidence_eligible }, null, 2)}</pre></div></div></details>
          </> : <div className="report-loading"><Loader2 className="human-spin" size={17} />正在封存 Episode 并执行离线 Replay…</div>}
        </section>}
      </>}
      {finished && researchRunId && <a className="return-to-research" href={`/?mode=research&run_id=${encodeURIComponent(researchRunId)}&episode_session_id=${encodeURIComponent(session.session_id)}&arm=${comparisonArm || 'S0'}`}><ArrowLeft size={15} />返回 Research Studio 并执行 Replay 关联</a>}
      <footer className="human-footer"><span>Usami 2.0 · Human Customer Playground</span><span>本地研究工具 · 不自动训练、不回流正式评测</span><a href="/?mode=research">Research Studio</a></footer>
    </main>
  </div>;
}

function OfferCard({ offer }: { offer: Offer }) {
  return <div className="actual-offer" data-testid="actual-offer"><div className="offer-header"><span className="offer-symbol">¥</span><div><span>客服展示的实际报价</span><strong>{offer.remedy === 'refund' ? '退款方案' : offer.remedy === 'exchange' ? '换货方案' : '补发方案'}</strong></div></div><div className="offer-terms">
    {offer.refund_cents != null && <div><span>退款金额</span><strong>{money(offer.refund_cents)}</strong></div>}
    {offer.extra_charge_cents != null && <div><span>需补差价</span><strong>{money(offer.extra_charge_cents)}</strong></div>}
    {offer.quantity != null && <div><span>数量</span><strong>{offer.quantity} 件</strong></div>}
    {offer.promised_arrival && <div><span>预计送达</span><strong>{new Date(offer.promised_arrival).toLocaleDateString()}</strong></div>}
    {offer.return_required && <div><span>退货要求</span><strong>需要寄回商品{offer.pickup_slot ? ` · ${offer.pickup_slot}` : ''}</strong></div>}
  </div></div>;
}
function ScoreCard({ label, value, detail }: { label: string; value: string; detail: string }) {
  const tone = value === '通过' || value === '是' || value === 'PASS' ? 'good' : value === 'FAIL' || value === '否' ? 'bad' : 'unknown';
  return <div className={`score-card ${tone}`}><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}
function resultLabel(value: boolean | null): string { return value == null ? '无法确定' : value ? '是' : '否'; }
