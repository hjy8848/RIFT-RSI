import { Activity, ArrowDownToLine, ArrowLeft, ArrowRight, CheckCircle2, CircleAlert, Clock3, Download, ExternalLink, FlaskConical, Loader2, LockKeyhole, Play, RefreshCw, ShieldCheck, Sparkles, Wrench } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ApiError, api } from './api';
import './ResearchStudio.css';

type AnyRow = Record<string, any>;
type Config = { configured: boolean; model: string | null; service: string; max_parallel_runs: number; message: string };
type Run = AnyRow & { run_id: string; status: string; phases: Record<string, string>; input: AnyRow; scenario?: AnyRow; designer_run?: AnyRow; episodes: AnyRow[]; reflection?: AnyRow; candidate_skill?: AnyRow; comparisons: AnyRow[]; error?: AnyRow };
const stageNames = ['Designer 生成世界', '任务认证', '真人顾客 Episode', 'Oracle 与错误诊断', 'Reflector 提炼 Skill', 'Skill 注入与对照', '研究记录'];
const stageKeys = ['world', 'task', 'human_episode', 'oracle_replay', 'reflection', 'skill_comparison', 'history'];
const stageDetails = [
  '模型真实提交业务约束；原生 Builder 在冻结 Seed 上构造世界。',
  'TaskSpecV2 绑定公开请求；独立 Reference 认证目标可达性。',
  '你亲自与真实 Service 对话；每个 Episode 独立克隆冻结世界。',
  'Engine、Oracle 和 Replay 分开报告业务结果、过程与安全。',
  '仅在明确选择已回放的 S0 人工 Episode 后，调用真实 ServiceReflector。',
  '候选只临时注入测试；真人两次互动是定性 sanity check，不是动态验收。',
  '保留请求、响应、证书、Episode、Skill Snapshot 和导出审计包。',
];
const code = (value: unknown) => typeof value === 'string' ? value : JSON.stringify(value, null, 2);
const stateLabel: Record<string, string> = { QUEUED: '排队中', RUNNING: '进行中', SUBMITTED: '已提交', COMPLETED: '完成', CERTIFIED: '已认证', READY: '可开始', READY_FOR_CUSTOMER: '等待真人 Episode', READY_FOR_SKILL_TEST: '候选可测试', REFLECTION_COMPLETE: '反思结束', INCONCLUSIVE: '证据不足', REJECTED: '拒绝', UNKNOWN: '未知', FAILED: '失败', BLOCKED: '已阻塞', BLOCKED_PROVIDER_TIMEOUT: '模型请求超时', BLOCKED_PROVIDER_ERROR: '模型服务错误', PROVIDER_TIMEOUT: '模型请求超时', PROVIDER_HTTP_ERROR: '模型接口错误', PROVIDER_CONNECTION_ERROR: '模型连接错误', ERROR: '系统错误', NOT_STARTED: '未开始', NOT_AVAILABLE: '不可用', WAITING_FOR_EPISODE: '等待 Episode' };
const humanState = (v?: string) => !v ? '未开始' : stateLabel[v] ?? v;
function behaviorExplanation(stage: number, run: Run | null, config: Config | null) {
  if (!run) return stageDetails[stage] + (config?.model ? ` 当前后端模型：${config.model}。` : ' 当前还没有模型配置。');
  const request = run.scenario?.task?.customer_request;
  const episodeCount = run.episodes?.length || 0;
  const latest = run.episodes?.[episodeCount - 1];
  if (stage === 0) return `${stageDetails[stage]} 本次真实 Designer 调用 ${run.designer_run?.calls?.length ?? 0} 次；世界阶段 ${humanState(run.phases?.world)}，任务阶段 ${humanState(run.phases?.task)}。`;
  if (stage === 1) return request ? `公开请求是：“${request}”。TaskSpec 隐藏目标由原生 Compiler 和 Reference 认证；初始世界 ${run.scenario?.public?.initial_world_hash?.slice(0, 12) || '尚未冻结'}…。` : stageDetails[stage];
  if (stage === 2) return request ? `顾客只从公开请求开始：“${request}”。Service 使用独立世界副本和真实工具；认证 witness 与隐藏目标不会进入 Service Prompt。` : stageDetails[stage];
  if (stage === 3) return latest ? `最近一条 ${latest.arm} Episode：业务目标 ${val(latest.outcome?.business_goal_success)}，过程错误 ${latest.outcome?.process_errors?.status || 'UNKNOWN'}，安全 ${latest.outcome?.safety_violations?.status || 'UNKNOWN'}，Replay ${latest.replay?.verified ? 'verified' : '未验证'}。` : stageDetails[stage];
  if (stage === 4) return run.reflection ? `Reflector 状态 ${run.reflection.status}。输入只来自人工明确选择的 train / S0 Episode 与新鲜 Native Replay；人工批注和 held-out / validation 不进入证据。` : stageDetails[stage];
  if (stage === 5) return run.candidate_skill ? `候选 Snapshot ${run.candidate_skill.snapshot_hash?.slice(0, 12)}…。状态 ${run.candidate_skill.status}；S1 只有在真实请求命中 Skill 后才记为已注入。` : stageDetails[stage];
  return `本研究运行保留 ${episodeCount} 条人工 Episode；当前结论 ${run.comparisons?.[0]?.status || run.candidate_skill?.status || run.status}。${stageDetails[stage]}`;
}

export default function ResearchStudio() {
  const params = useMemo(() => new URLSearchParams(location.search), []);
  const [config, setConfig] = useState<Config | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [runs, setRuns] = useState<Run[]>([]);
  const [run, setRun] = useState<Run | null>(null);
  const [stage, setStage] = useState(0);
  const [seed, setSeed] = useState(42);
  const [distribution, setDistribution] = useState('mixed_after_sales');
  const [split, setSplit] = useState('train');
  const [difficulty, setDifficulty] = useState('multi_objective');
  const [model, setModel] = useState('');
  const [attempts, setAttempts] = useState(2);
  const [designerTimeout, setDesignerTimeout] = useState(180);
  const [builderNodes, setBuilderNodes] = useState(512);
  const [proofCandidates, setProofCandidates] = useState(512);
  const [proofSteps, setProofSteps] = useState(64);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [recoverableRunId, setRecoverableRunId] = useState('');
  const attachedRef = useRef('');

  const refreshRuns = useCallback(async (selectedId?: string) => {
    const [history] = await Promise.all([api<{ runs: Run[] }>('/research/runs')]);
    setRuns(history.runs);
    const id = selectedId || run?.run_id || params.get('run_id') || undefined;
    if (id) {
      const current = await api<Run>(`/research/runs/${id}`);
      setRun(current);
      setModel(current.model?.model || '');
      setDesignerTimeout(current.input?.designer_timeout_seconds || 180);
      return current;
    }
    return null;
  }, [params, run?.run_id]);

  useEffect(() => {
    let alive = true;
    void Promise.all([api<Config>('/research/config'), api<{ runs: Run[] }>('/research/runs')])
      .then(async ([cfg, history]) => {
        if (!alive) return;
        setConfig(cfg); setRuns(history.runs);
        const id = params.get('run_id');
        if (id) {
          const current = await api<Run>(`/research/runs/${id}`);
          if (alive) { setRun(current); setModel(current.model?.model || ''); setDesignerTimeout(current.input?.designer_timeout_seconds || 180); }
        }
        if (alive) setLoaded(true);
      }).catch(cause => {
        if (!alive) return;
        const id = params.get('run_id');
        if (id && cause instanceof ApiError && cause.status === 404) {
          setRecoverableRunId(id);
          setError('本机研究记录的浏览器身份已失效。记录仍由随机运行 ID 保护，可恢复此运行的本地访问权。');
        } else setError(String(cause));
        setLoaded(true);
      });
    return () => { alive = false; };
  }, [params]);

  const active = !!run && ['DESIGNER_QUEUED', 'DESIGNER_RUNNING', 'REFLECTOR_RUNNING'].includes(run.status);
  useEffect(() => {
    if (!run || !active) return;
    const timer = window.setInterval(() => { void refreshRuns(run.run_id).catch(cause => setError(String(cause))); }, 1400);
    return () => window.clearInterval(timer);
  }, [active, refreshRuns, run?.run_id]);

  useEffect(() => {
    const sessionId = params.get('episode_session_id');
    const runId = params.get('run_id');
    const arm = params.get('arm') === 'S1' ? 'S1' : 'S0';
    if (!loaded || !sessionId || !runId || attachedRef.current === sessionId) return;
    attachedRef.current = sessionId;
    void api<Run>(`/research/runs/${runId}/attach-episode`, { session_id: sessionId, arm, use_for_reflection: false })
      .then(async updated => {
        setRun(updated); setRuns(await api<{ runs: Run[] }>('/research/runs').then(x => x.runs)); setStage(3);
        const clean = new URL(location.href); clean.searchParams.delete('episode_session_id'); clean.searchParams.delete('arm');
        history.replaceState(null, '', clean);
      }).catch(cause => { attachedRef.current = ''; setError(`Episode 未能关联到研究运行：${String(cause)}`); });
  }, [loaded, params]);

  async function createRun(event: React.FormEvent) {
    event.preventDefault(); setBusy(true); setError('');
    try {
      const created = await api<Run>('/research/runs', { seed, distribution, split, difficulty_preference: difficulty,
        designer_model: model.trim() || null, designer_timeout_seconds: designerTimeout, max_attempts: attempts, build_nodes: builderNodes,
        proof_candidates: proofCandidates, proof_steps: proofSteps });
      setRun(created); setStage(0); setRuns(await api<{ runs: Run[] }>('/research/runs').then(x => x.runs));
      const url = new URL(location.href); url.searchParams.set('run_id', created.run_id); history.replaceState(null, '', url);
    } catch (cause) { setError(String(cause)); }
    finally { setBusy(false); }
  }
  async function recoverRun() {
    if (!recoverableRunId) return;
    setBusy(true); setError('');
    try {
      const restored = await api<Run>(`/research/runs/${recoverableRunId}/claim`, {});
      setRun(restored);
      setRuns(await api<{ runs: Run[] }>('/research/runs').then(x => x.runs));
      setRecoverableRunId('');
      setStage(4);
    } catch (cause) { setError(String(cause)); }
    finally { setBusy(false); }
  }
  async function cancelActiveRun() {
    if (!run) return;
    setBusy(true); setError('');
    try { setRun(await api<Run>(`/research/runs/${run.run_id}/cancel`, {})); }
    catch (cause) { setError(String(cause)); }
    finally { setBusy(false); }
  }
  async function selectRun(id: string) { setError(''); try { const selected = await refreshRuns(id); setStage(0); if (selected) { const url = new URL(location.href); url.searchParams.set('run_id', id); history.replaceState(null, '', url); } } catch (cause) { setError(String(cause)); } }
  async function chooseReflectionEvidence(sessionId: string, replaceLegacyEvidence = false, timeoutSeconds = 180) {
    if (!run) return;
    setBusy(true); setError('');
    try {
      const attached = await api<Run>(`/research/runs/${run.run_id}/attach-episode`, { session_id: sessionId, arm: 'S0', use_for_reflection: true });
      setRun(attached);
      const next = await api<Run>(`/research/runs/${run.run_id}/reflect`, { session_id: sessionId, explicit_human_evidence_selection: true, replace_legacy_evidence: replaceLegacyEvidence, timeout_seconds: timeoutSeconds });
      setRun(next); setStage(4);
    } catch (cause) { setError(String(cause)); }
    finally { setBusy(false); }
  }
  async function addHumanAnnotation(sessionId: string, eventIndex: number, category: string, note: string) {
    if (!run) return;
    const updated = await api<Run>(`/research/runs/${run.run_id}/annotations`, {
      session_id: sessionId, event_index: eventIndex, category, note,
    });
    setRun(updated);
    setRuns(await api<{ runs: Run[] }>('/research/runs').then(result => result.runs));
  }
  async function exportRun() {
    if (!run) return;
    const response = await fetch(`/api/research/runs/${run.run_id}/export`, { credentials: 'same-origin' });
    if (!response.ok) { setError(`导出失败：HTTP ${response.status}`); return; }
    const url = URL.createObjectURL(await response.blob()); const link = document.createElement('a');
    link.href = url; link.download = `usami-research-${run.run_id}.json`; link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  const runUrl = (arm: 'S0' | 'S1') => {
    if (!run?.scenario?.scenario_id) return '#';
    const query = new URLSearchParams({ mode: 'human', scenario: run.scenario.scenario_id,
      researchRunId: run.run_id, arm });
    if (arm === 'S1' && run.candidate_skill?.snapshot_hash) query.set('skillSnapshotHash', run.candidate_skill.snapshot_hash);
    return `/?${query.toString()}`;
  };
  const selectedEpisode = (id: string) => run?.episodes.find(row => row.session_id === id);

  return <div className="research-app" data-testid="research-studio">
    <header className="research-topbar"><a href="/?" className="research-back"><ArrowLeft size={16} />World Explorer</a><a className="research-brand" href="/?mode=research">USAMI <span>RESEARCH STUDIO</span></a><div className={`research-provider ${config?.configured ? 'ready' : 'offline'}`}><span />{config?.configured ? `${config.model} · 后端已配置` : '模型尚未配置'}</div></header>
    <div className="research-layout">
      <aside className="research-sidebar">
        <div className="research-sidebar-title"><span className="research-icon"><FlaskConical size={19} /></span><div><strong>研究工作台</strong><small>QUALITATIVE MANUAL PILOT</small></div></div>
        <div className="research-nav-label">研究流程</div>
        <nav aria-label="研究阶段">{stageNames.map((name, index) => {
          const key = stageKeys[index]; const status = run?.phases?.[key] ?? 'NOT_STARTED';
          return <button key={key} className={`research-stage ${stage === index ? 'active' : ''}`} onClick={() => setStage(index)}>
            <span className="stage-number">{index + 1}</span><span className="stage-label"><strong>{name}</strong><small>{humanState(status)}</small></span>{['COMPLETED', 'CERTIFIED', 'VERIFIED'].includes(status) ? <CheckCircle2 size={15} /> : active && status === 'RUNNING' ? <Loader2 className="research-spin" size={15} /> : null}
          </button>;
        })}</nav>
        <div className="research-sidebar-foot"><ShieldCheck size={16} /><span>Native Engine · Reference · Oracle · Replay</span></div>
        <a className="research-human-link" href="/?mode=human"><ExternalLink size={14} />真人顾客实验台</a>
      </aside>

      <main className="research-main">
        <div className="research-heading"><div><span className="research-eyebrow">USAMI 2.0 · RESEARCH WORKSPACE</span><h1>从业务世界到可审计的 Skill 实验</h1><p>每一步都调用真实 Usami 代码；未完成或未验证的环节会保持明确状态。</p></div>{run && <div className="research-run-id"><span>当前运行</span><code>{run.run_id}</code><button title="重新载入" onClick={() => void refreshRuns(run.run_id)}><RefreshCw size={14} /></button></div>}</div>
        {recoverableRunId ? <div className="research-error" role="alert"><CircleAlert size={17} /><span>{error || '此运行需要恢复本地访问权。'}</span><button className="research-recovery-button" disabled={busy} onClick={() => void recoverRun()}>{busy ? '恢复中…' : '恢复此运行'}</button></div> : error && <div className="research-error" role="alert"><CircleAlert size={17} /><span>{error}</span><button onClick={() => setError('')}>关闭</button></div>}
        <div className="research-stage-strip">{stageNames.map((name, i) => <button key={name} onClick={() => setStage(i)} className={stage === i ? 'selected' : ''}><span>{String(i + 1).padStart(2, '0')}</span>{name}{i < 6 && <ArrowRight size={13} />}</button>)}</div>
        <section className="research-current-note"><div className="note-symbol"><Activity size={18} /></div><div><strong>当前系统行为说明</strong><p>{behaviorExplanation(stage, run, config)}</p><small>{run ? `当前运行：${humanState(run.status)}。所有人工交互仅标记为 human / qualitative_manual_pilot，不会默认进入 Benchmark 或训练证据。` : '尚未创建运行；配置在本地 Python 后端检查，浏览器不会接触模型密钥。'}</small></div></section>

        {stage === 0 && <section className="research-stage-content">
          {!run ? <div className="research-create-grid"><form className="research-panel designer-form" onSubmit={createRun}>
            <div className="research-panel-heading"><div><span className="research-eyebrow">STAGE 01</span><h2>让 Designer 提出世界与挑战</h2><p>先提交约束，再由 Builder 构造；之后 Designer 只能引用已存在的实体。</p></div><Sparkles size={20} /></div>
            <label>研究分布<select value={distribution} onChange={e => setDistribution(e.target.value)}><option value="mixed_after_sales">混合售后边界</option><option value="multi_line">多商品组合</option><option value="inventory_and_deadline">库存与时限权衡</option></select></label>
            <div className="research-field-row"><label>冻结 Seed<input type="number" value={seed} onChange={e => setSeed(Number(e.target.value))} /></label><label>数据分区<select value={split} onChange={e => setSplit(e.target.value)}><option value="train">train（人工探索，不自动训练）</option><option value="validation">validation</option><option value="heldout">held-out（不输入反思）</option></select></label></div>
            <div className="research-field-row"><label>生成方向<select value={difficulty} onChange={e => setDifficulty(e.target.value)}><option value="basic">基础</option><option value="complex">复杂</option><option value="multi_objective">多目标</option></select><small>仅为提示偏好，不是经验证的难度分数。</small></label><label>Designer 模型<input value={model || config?.model || ''} onChange={e => setModel(e.target.value)} placeholder="使用后端当前模型" /></label></div>
            <div className="research-field-row research-budget-fields"><label>提案尝试<input type="number" min={1} max={3} value={attempts} onChange={e => setAttempts(Number(e.target.value))} /><small>用于提案或验证纠错；网络错误会立即停止，不会当成纠错重试。</small></label><label>Builder 节点预算<input type="number" min={1} max={100000} value={builderNodes} onChange={e => setBuilderNodes(Number(e.target.value))} /></label></div>
            <div className="research-field-row research-budget-fields"><label>Designer 单次请求超时（秒）<input type="number" min={1} max={600} value={designerTimeout} onChange={e => setDesignerTimeout(Number(e.target.value))} /><small>默认 180 秒；记录到本次运行。Designer 不设输出 Token 上限。</small></label><div className="research-field-note">请求超时会阻塞当前阶段，不会再被当作提案校验错误重试。</div></div>
            <div className="research-field-row research-budget-fields"><label>Reference 候选预算<input type="number" min={1} max={10000} value={proofCandidates} onChange={e => setProofCandidates(Number(e.target.value))} /></label><label>Reference 最大操作步数<input type="number" min={1} max={2000} value={proofSteps} onChange={e => setProofSteps(Number(e.target.value))} /></label></div>
            <div className="research-budgets"><span><strong>最多 2</strong> 同时研究运行</span><span><strong>3</strong> 尝试上限</span><span>预算固定后随运行保存</span></div>
            <button className="research-primary" disabled={!config?.configured || busy || active} data-testid="create-designer-run">{busy || active ? <Loader2 className="research-spin" size={16} /> : <Play size={16} />}{active ? 'Designer 正在运行…' : 'Designer 自主生成并认证'}</button>
            {!config?.configured && <p className="research-warning">后端目前没有可用的 LLM 配置。配置模型后按钮才会启动真实 Designer。</p>}
          </form><div className="research-panel research-how"><span className="research-eyebrow">NO SCRIPTED TASKS</span><h2>后台实际调用</h2><ol><li><strong>OpenAIDesignerProvider</strong><span>真实模型提交 ScenarioDraft 与 TaskDraft</span></li><li><strong>Native Builder</strong><span>按 Seed 构造冻结世界，返回 SAT / UNSAT / UNKNOWN</span></li><li><strong>Task Compiler</strong><span>实体别名和公开请求绑定为 TaskSpecV2</span></li><li><strong>Reference Certification</strong><span>证明预算内的业务可达性与 Replay 证书</span></li></ol><p>Designer 从未收到 Reference witness；Service 只会收到公开请求与自身工具查询结果。</p></div></div>
          : <DesignerResult run={run} onCancel={() => void cancelActiveRun()} canCancel={active} onRetry={() => { setRun(null); const url = new URL(location.href); url.searchParams.delete('run_id'); history.replaceState(null, '', url); }} onNext={() => setStage(1)} />}
        </section>}

        {stage === 1 && <TaskCertificationView run={run} onNext={() => setStage(2)} />}
        {stage === 2 && <HumanEpisodeView run={run} runUrl={runUrl} />}
        {stage === 3 && <EpisodeDiagnosis run={run} onSelectReflection={chooseReflectionEvidence} onAnnotate={addHumanAnnotation} busy={busy} selectedEpisode={selectedEpisode} />}
        {stage === 4 && <ReflectionView run={run} busy={busy} onSelectReflection={chooseReflectionEvidence} selectedEpisode={selectedEpisode} />}
        {stage === 5 && <SkillComparison run={run} runUrl={runUrl} />}
        {stage === 6 && <HistoryView runs={runs} run={run} onSelect={selectRun} onExport={() => void exportRun()} />}
        <footer className="research-footer"><span>Usami 2.0 Research Studio</span><span>工程机制跑通 ≠ 已证明 RSI 能力提升</span></footer>
      </main>
    </div>
  </div>;
}

function StatusPill({ value }: { value?: string }) { const state = value || 'NOT_STARTED'; return <span className={`research-pill pill-${state.toLowerCase().replaceAll('_', '-')}`}>{humanState(state)}</span>; }
function JsonDetails({ title, value }: { title: string; value: unknown }) { return <details className="research-json"><summary>{title}</summary><pre>{code(value)}</pre></details>; }
function DesignerResult({ run, onRetry, onNext, onCancel, canCancel }: { run: Run; onRetry: () => void; onNext: () => void; onCancel: () => void; canCancel: boolean }) {
  const designer = run.designer_run;
  return <div className="research-panel result-panel"><div className="research-panel-heading"><div><span className="research-eyebrow">REAL DESIGNER OUTPUT</span><h2>生成与验证状态 <StatusPill value={run.status} /></h2><p>世界构建和任务认证分别记录；UNSAT、UNKNOWN 与模型协议失败不会合并。</p></div><button className="research-button" onClick={onRetry}>新建另一运行</button></div>
    {!designer ? <div className="research-wait" role="status"><Loader2 className="research-spin" size={20} /><div><strong>{humanState(run.status)}</strong><span>等待真实模型提交。当前阶段：{humanState(run.phases?.world)} / {humanState(run.phases?.task)}</span></div>{canCancel && <button className="research-button" onClick={onCancel}>请求取消</button>}</div> : <>
      <div className="research-result-grid"><Metric label="World" value={designer.build_result?.status || '未生成'} detail={designer.build_result?.reason || '没有 Builder 结果'} /><Metric label="Task" value={designer.status} detail={designer.validation_errors?.slice(-1)[0] || '查看 Native Reference 认证'} /><Metric label="Designer 请求" value={designer.calls?.length ?? 0} detail="实际模型调用，含失败响应" /><Metric label="World Hash" value={designer.build_result?.world_hash?.slice(0, 13) || '—'} detail="完整哈希在记录中" /></div>
      {designer.pre_run_quality && <DesignerQualityDiagnostic quality={designer.pre_run_quality} />}
      {run.error && <div className="research-warning"><CircleAlert size={15} />{run.error.messages?.join(' · ') || run.error.message}</div>}
      <div className="research-subheading"><h3>生成轨迹</h3><span>真实请求与模型响应</span></div>
      {(designer.calls || []).map((call: AnyRow, index: number) => <div className="designer-call-row" key={index}><span className="call-phase">{call.phase === 'world' ? '世界' : '任务'} · 第 {call.attempt + 1} 次</span><span>{call.call?.prompt_tokens != null ? `${call.call.prompt_tokens} in / ${call.call.completion_tokens ?? '—'} out` : 'Token 未报告'}</span><StatusPill value={call.call?.response?.error?.code || (call.validation_feedback ? 'REJECTED' : 'SUBMITTED')} /><small className="call-hashes">Prompt {call.prompt_hash?.slice(0, 12) || '—'} · Schema {call.submission_schema_hash?.slice(0, 12) || '—'}{call.capability_manifest_hash ? ` · Capability ${call.capability_manifest_hash.slice(0, 12)}` : ''}</small><JsonDetails title="模型原始请求与回复" value={{ request: call.call?.request, response: call.call?.response, validation_feedback: call.validation_feedback }} /></div>)}
      <div className="research-two-cards"><article><span className="research-eyebrow">BUILDER</span><h3>世界构造</h3><p>{designer.build_result?.status || '未运行'} · {designer.build_result?.reason || '—'}</p>{designer.build_result?.world && <div className="world-facts"><span>{Object.keys(designer.build_result.world.customers || {}).length} 顾客</span><span>{Object.keys(designer.build_result.world.orders || {}).length} 订单</span><span>{Object.keys(designer.build_result.world.lines || {}).length} 商品行</span><span>{Object.keys(designer.build_result.world.skus || {}).length} SKU</span><span>{Object.keys(designer.build_result.world.warehouses || {}).length} 仓库</span></div>}{designer.scenario_spec && <JsonDetails title="ScenarioSpec 与 SAT 约束" value={designer.scenario_spec} />}</article><article><span className="research-eyebrow">TASK DESIGNER</span><h3>任务提交</h3><p>{designer.task_draft?.customer_request || '尚未提交公开请求'}</p>{designer.task_draft && <JsonDetails title="TaskDraft 原始内容" value={designer.task_draft} />}{designer.task_attempts?.map((attempt: AnyRow, i: number) => <div className="attempt-line" key={i}><StatusPill value={attempt.status} /><span>{attempt.reason || '原生任务认证已运行'}</span></div>)}</article></div>
      {designer.build_result?.world && <WorldOverview world={designer.build_result.world} />}
      {designer.status === 'CERTIFIED' ? <button className="research-primary narrow" onClick={onNext}>查看任务目标与证书<ArrowRight size={16} /></button> : <div className="research-notice"><LockKeyhole size={16} />当前没有通过认证的任务，因此不会暴露 Human Episode 启动入口。</div>}
    </>}
  </div>;
}
function Metric({ label, value, detail }: { label: string; value: unknown; detail: string }) { return <div className="research-metric"><span>{label}</span><strong>{String(value)}</strong><small>{detail}</small></div>; }
function DesignerQualityDiagnostic({ quality }: { quality: AnyRow }) {
  const novelty: Record<string, string> = {
    EVALUATED_NOVEL: '已与历史索引比较：未发现相同结构',
    EVALUATED_SEEN: '已与历史索引比较：发现相同结构',
    NOT_EVALUATED: '没有历史索引，尚未评估新颖性',
  };
  const diagnostics: AnyRow[] = Array.isArray(quality.candidate_consequence_diagnostics)
    ? quality.candidate_consequence_diagnostics : [];
  return <section className="designer-quality-diagnostic">
    <div className="quality-diagnostic-heading"><div><span className="research-eyebrow">RESEARCHER DIAGNOSTICS</span><h3>出题结构诊断</h3></div><span>仅用于研究分析，不参与任务认证或版本选择</span></div>
    <div className="quality-diagnostic-summary"><div><small>历史新颖性</small><strong>{novelty[quality.novelty_status] || '未评估'}</strong></div><div><small>结构后果类别下界</small><strong>{quality.distinct_consequence_classes ?? '—'}</strong></div><div><small>历史索引</small><code>{String(quality.history_index_hash || '').slice(0, 12) || '未提供'}</code></div></div>
    {diagnostics.length > 0 && <div className="quality-alternative-list"><strong>候选方案约束筛查（不是执行或 Oracle 结果）</strong>{diagnostics.map((row, index) => {
      const exclusions: Array<[string, number]> = Array.isArray(row.excluded_condition_counts) ? row.excluded_condition_counts : [];
      return <div key={`${row.goal_id || index}-${index}`}><span>{row.goal_id || `目标 ${index + 1}`}</span><span>{row.feasible_alternatives ?? 0} / {row.alternatives_examined ?? 0} 项规则筛查可行</span><small>{exclusions.length ? exclusions.map(([name, count]) => `${name} ×${count}`).join(' · ') : '未记录被排除条件'}</small></div>;
    })}</div>}
  </section>;
}
function WorldOverview({ world }: { world: AnyRow }) {
  const skus: AnyRow[] = Object.values(world.skus || {});
  const orders: AnyRow[] = Object.values(world.orders || {});
  const lines: AnyRow[] = Object.values(world.lines || {});
  const stock: AnyRow[] = Object.values(world.inventory || {});
  const routes: AnyRow[] = Object.values(world.routes || {});
  const evidence: AnyRow[] = Object.values(world.evidence || {});
  const byId = (rows: AnyRow[], id: string) => rows.find(row => row.id === id);
  return <section className="research-panel world-overview"><div className="research-panel-heading"><div><span className="research-eyebrow">FROZEN BUSINESS WORLD</span><h2>世界关系与可见业务条件</h2><p>这是 Builder 实际构造并通过原生不变量检查的研究者视图；Service 不会收到整份世界 JSON。</p></div><StatusPill value="CERTIFIED" /></div>
    <div className="world-relations">{orders.map(order => <article key={order.id}><span className="world-node world-order">订单 · {String(order.id).slice(0, 18)}</span>{lines.filter(line => line.order_id === order.id).map(line => { const sku = byId(skus, line.sku_id); const issue = evidence.find(item => item.line_id === line.id); return <div className="world-relation" key={line.id}><span>商品行 · {String(line.id).slice(0, 16)}</span><ArrowRight size={14} /><span className="world-node world-sku">{sku?.product_family || 'SKU'} · {String(sku?.id || '').slice(0, 12)}</span><small>{issue?.issue || '无证据'} · 数量 {line.quantity} · 实付 ¥{((line.allocated_paid_cents || 0) / 100).toFixed(2)}</small></div>; })}</article>)}</div>
    <div className="world-table-wrap"><h3>SKU 与价格 / 退货条件</h3><table><thead><tr><th>商品</th><th>属性</th><th>售价</th><th>退货窗口</th><th>需寄回</th></tr></thead><tbody>{skus.map(sku => <tr key={sku.id}><td>{sku.product_family}<small>{sku.id}</small></td><td>{Object.entries(sku.attributes || {}).map(([key, value]) => `${key}: ${value}`).join(' · ') || '无额外属性'}</td><td>¥{(sku.price_cents / 100).toFixed(2)}</td><td>{sku.return_days} 天</td><td>{sku.return_required ? '是' : '否'}</td></tr>)}</tbody></table></div>
    <div className="world-table-wrap"><h3>仓库库存与配送路线</h3><table><thead><tr><th>SKU</th><th>仓库</th><th>可用库存</th><th>配送时间</th><th>配送费</th></tr></thead><tbody>{stock.map(row => { const sku = byId(skus, row.sku_id); const route = routes.find(item => item.warehouse_id === row.warehouse_id); return <tr key={row.id}><td>{sku?.product_family || row.sku_id}</td><td>{String(row.warehouse_id).slice(0, 18)}</td><td>{row.available}</td><td>{route ? `${route.days} 天` : '—'}</td><td>{route ? `¥${(route.fee_cents / 100).toFixed(2)}` : '—'}</td></tr>; })}</tbody></table></div>
    <JsonDetails title="展开完整冻结 World JSON" value={world} />
  </section>;
}
function TaskCertificationView({ run, onNext }: { run: Run | null; onNext: () => void }) {
  if (!run?.scenario) return <EmptyStage title="没有已认证任务" text="先完成 Stage 1 的真实 Designer 生成。失败和 UNKNOWN 会保留在当前记录中。" />;
  const scenario = run.scenario; const task = scenario.task; const cert = scenario.task_certificate;
  return <div className="research-panel result-panel"><div className="research-panel-heading"><div><span className="research-eyebrow">STAGE 02 · FROZEN TASK</span><h2>任务目标与 Reference 认证</h2><p>公开请求是 Service 的输入；隐藏目标和见证只在本研究者视图显示。</p></div><StatusPill value={cert.status} /></div>
    <div className="task-request-card"><span>顾客公开请求</span><blockquote>“{task.customer_request}”</blockquote><small>顾客身份：{task.customer_id} · 初始世界：{task.world_hash.slice(0, 20)}…</small></div>
    <div className="research-researcher-banner"><LockKeyhole size={16} /><div><strong>研究者视图：内部评分目标</strong><span>这些字段不会发送给 Service；顾客对话只从 TaskSpecV2 的公开请求开始。</span></div></div>
    <div className="goal-list">{(scenario.researcher_hidden_goal_view || []).map((goal: AnyRow, index: number) => <article key={goal.goal_id || index}><span>目标 {index + 1}</span><strong>{goal.goal_id}</strong><p>{goal.subject_mention || goal.public_request || goal.fact_kind}</p><small>{goal.public_request_span || goal.public_request || '没有公开目标文本'} · {goal.allowed_remedies?.join(' / ') || goal.fact_kind}</small></article>)}</div>
    <div className="certificate-grid"><Metric label="世界不变量" value="通过" detail={`World ${scenario.public.initial_world_hash.slice(0, 16)}…`} /><Metric label="TaskSpec 绑定" value={task.record_hash?.slice(0, 14) || '已封存'} detail="公开请求与目标字段绑定" /><Metric label="Reference 搜索" value={cert.status} detail={`${cert.candidates_examined} candidates · budget ${cert.budget_hash?.slice(0, 12) || '—'}`} /><Metric label="证明范围" value={cert.full_task_reachability_certified ? '业务任务可达' : '部分 / 未决'} detail="不等于自然对话必然可达" /></div>
    <div className="witness-note"><ShieldCheck size={17} /><span>私有 witness 数量：{scenario.reference_witness_summary?.witness_count ?? 0}。其哈希用于审计，不把具体执行答案提供给 Service。</span></div>
    <JsonDetails title="展开完整 TaskSpecV2 与认证证书" value={{ task, certificate: cert, world_certificate: scenario.world_certificate, proof_customer: scenario.proof_customer }} />
    <button className="research-primary narrow" onClick={onNext}>进入真人 Customer Episode<ArrowRight size={16} /></button>
  </div>;
}
function HumanEpisodeView({ run, runUrl }: { run: Run | null; runUrl: (arm: 'S0' | 'S1') => string }) {
  if (!run?.scenario) return <EmptyStage title="没有冻结场景可供交互" text="Designer 任务通过 Reference 认证后，才可以启动真人顾客 Episode。" />;
  const hasS0 = run.episodes?.some(ep => ep.arm === 'S0');
  return <div className="research-panel result-panel"><div className="research-panel-heading"><div><span className="research-eyebrow">STAGE 03 · HUMAN CUSTOMER</span><h2>使用冻结任务启动真实 Service</h2><p>HumanSessionManager 将复用 TaskSpecV2、原生 Engine、Oracle、Replay 与当前 Service 模型。</p></div><span className="human-tag">人工 · 定性探索</span></div>
    <div className="human-episode-summary"><div><span>本次公开顾客请求</span><p>“{run.scenario.task.customer_request}”</p><small>世界和身份固定；Service 看不到下面证书的隐藏目标。</small></div><div className="human-episode-hash"><span>INITIAL WORLD</span><code>{run.scenario.public.initial_world_hash}</code><span>Task Certificate</span><code>{run.scenario.task_certificate.record_hash}</code></div></div>
    <a className="research-primary narrow research-link-button" href={runUrl('S0')} data-testid="launch-s0-human"><Play size={16} />{hasS0 ? '再运行一次 S0（同世界新 Episode）' : '开始真人 Episode · S0 无 Skill'}<ExternalLink size={14} /></a>
    <div className="research-notice"><ShieldCheck size={16} />开始后，页面只给 Service 公开请求和真实工具返回。Episode 完成时回到本工作台查看评分并选择是否用于 Reflection。</div>
    {run.episodes?.length > 0 && <div className="attached-episodes"><h3>已关联 Episodes</h3>{run.episodes.map((ep: AnyRow) => <div key={ep.session_id}><span className="arm-chip">{ep.arm}</span><code>{ep.session_id.slice(0, 18)}…</code><StatusPill value={ep.replay?.verified ? 'VERIFIED' : 'UNKNOWN'} /><span>{ep.costs?.service_calls ?? 0} Service calls</span><span>{ep.costs?.known_total_tokens ?? '未知'} tokens</span></div>)}</div>}
  </div>;
}
function EpisodeDiagnosis({ run, onSelectReflection, onAnnotate, busy, selectedEpisode }: { run: Run | null; onSelectReflection: (id: string) => void; onAnnotate: (id: string, index: number, category: string, note: string) => Promise<void>; busy: boolean; selectedEpisode: (id: string) => AnyRow | undefined }) {
  if (!run?.episodes?.length) return <EmptyStage title="还没有 Service Episode" text="先在 Stage 3 启动真人顾客对话；对话封存后会在返回工作台时自动重新 Replay 并关联。" />;
  return <div className="research-panel result-panel"><div className="research-panel-heading"><div><span className="research-eyebrow">STAGE 04 · NATIVE OUTCOME</span><h2>业务结果、决策过程与安全分开看</h2><p>工具被正确拒绝不等于真实违规；Replay 通过只说明轨迹可复算。</p></div><StatusPill value={run.phases.oracle_replay} /></div>
    {run.episodes.map((link: AnyRow) => <EpisodeReport key={link.session_id} link={link} annotations={run.human_annotations || []} canReflect={run.input?.split === 'train'} onSelectReflection={onSelectReflection} onAnnotate={onAnnotate} busy={busy} alreadySelected={!!selectedEpisode(link.session_id)?.explicitly_selected_for_reflection} />)}
  </div>;
}
function EpisodeReport({ link, annotations, canReflect, onSelectReflection, onAnnotate, busy, alreadySelected }: { link: AnyRow; annotations: AnyRow[]; canReflect: boolean; onSelectReflection: (id: string) => void; onAnnotate: (id: string, index: number, category: string, note: string) => Promise<void>; busy: boolean; alreadySelected: boolean }) {
  const report = link.report; const outcome = link.outcome || report?.outcome || {}; const episode = report?.episode || {};
  const tools: AnyRow[] = episode.tool_executions || [];
  const modelCalls: AnyRow[] = (episode.model_calls || []).filter((call: AnyRow) => call.role === 'service');
  const [annotation, setAnnotation] = useState('');
  const [annotationCategory, setAnnotationCategory] = useState('decision');
  const [annotationAt, setAnnotationAt] = useState<number | null>(null);
  const [annotationBusy, setAnnotationBusy] = useState(false);
  async function saveAnnotation(eventIndex: number) {
    if (!annotation.trim()) return;
    setAnnotationBusy(true);
    try { await onAnnotate(link.session_id, eventIndex, annotationCategory, annotation.trim()); setAnnotation(''); setAnnotationAt(null); }
    finally { setAnnotationBusy(false); }
  }
  return <article className="episode-report-card"><div className="episode-report-heading"><div><span className="arm-chip">{link.arm}</span><strong>{link.customer_source} · {link.study_type}</strong><code>{link.episode_hash?.slice(0, 18)}…</code></div><StatusPill value={link.replay?.verified ? 'VERIFIED' : 'FAILED'} /></div>
    <div className="research-result-grid compact"><Metric label="Business Goal" value={val(outcome.business_goal_success)} detail={val(outcome.final_goal_reached)} /><Metric label="Process Errors" value={outcome.process_errors?.status || '—'} detail={`${outcome.process_errors?.evidence_refs?.length || 0} evidence refs`} /><Metric label="Safety" value={outcome.safety_violations?.status || '—'} detail={`${outcome.safety_violations?.evidence_refs?.length || 0} evidence refs`} /><Metric label="Strict Success" value={val(outcome.strict_task_success)} detail={`Evaluation ${outcome.evaluation_status || '—'}`} /><Metric label="Replay" value={link.replay?.verified ? 'verified' : 'unresolved'} detail={`${link.replay?.event_count ?? 0} events`} /><Metric label="Usage" value={`${link.costs?.service_calls ?? 0} calls`} detail={`${link.costs?.known_total_tokens ?? 'unknown'} tokens`} /></div>
    <div className="decision-timeline">{tools.map((tool, index) => { const operations = tool.oracle?.operations || []; const oracle = operations[0] || {}; const result = tool.result || {}; const correct = oracle.operation_correct; const selected = findModelCall(modelCalls, tool); const attachedNotes = annotations.filter(row => row.session_id === link.session_id && row.event_index === index); return <details className={`decision-row ${correct === false ? 'decision-error' : correct === true ? 'decision-ok' : 'decision-unknown'}`} key={index}><summary><span className="decision-index">{String(index + 1).padStart(2, '0')}</span><span className="decision-op"><strong>{tool.operation?.name || 'tool operation'}</strong><small>{tool.operation?.resource_id || 'no resource id'}</small></span><span className="decision-result">{result.error_code || (result.success ? '成功返回' : result.message || '结果见证据')}</span><span className="oracle-tag">Oracle · {oracle.classification || 'unresolved'}</span></summary><div className="decision-details"><div><strong>模型当时看到的上下文</strong><pre>{code(selected?.request?.messages || '没有找到与该工具调用对应的模型请求；原始请求保留在完整 Episode 中。')}</pre></div><div><strong>模型实际选择与原始响应</strong><pre>{code({ operation: tool.operation, model_response: selected?.response })}</pre></div><div><strong>工具实际返回</strong><pre>{code(result)}</pre></div><div><strong>Oracle 判定与状态变化</strong><pre>{code({ classification: oracle.classification, operation_correct: oracle.operation_correct, authorized: oracle.authorized, reference_allowed: oracle.reference_allowed, violations: tool.oracle?.violations || [], before_state_hash: tool.before_state_hash, after_state_hash: tool.after_state_hash })}</pre></div><div className="manual-annotation-box"><strong>研究者人工标注 <small>独立于 Oracle，不计入官方错误，不进入 Reflection</small></strong>{attachedNotes.map((note: AnyRow) => <p key={note.annotation_id}>{note.category} · {note.note}</p>)}{annotationAt === index ? <><select value={annotationCategory} onChange={event => setAnnotationCategory(event.target.value)}><option value="decision">决策疑点</option><option value="communication">表达问题</option><option value="repetition">重复操作</option><option value="other">其他</option></select><input value={annotation} onChange={event => setAnnotation(event.target.value)} maxLength={1000} placeholder="记录一个待复核的疑点…"/><button disabled={annotationBusy || !annotation.trim()} onClick={() => void saveAnnotation(index)}>{annotationBusy ? '保存中…' : '保存人工标注'}</button></> : <button onClick={() => setAnnotationAt(index)}>标记疑似问题</button>}</div></div></details>; })}</div>
    {link.arm === 'S0' && link.replay?.verified && canReflect && <button className="research-secondary" disabled={busy || alreadySelected} onClick={() => onSelectReflection(link.session_id)}>{alreadySelected ? <CheckCircle2 size={15} /> : <Sparkles size={15} />}{alreadySelected ? '已明确选择为本次 Reflection 证据' : '选择此 Episode 并让 Reflector 分析'}</button>}
    {link.arm === 'S0' && link.replay?.verified && !canReflect && <div className="research-notice"><LockKeyhole size={15} />{link.report?.episode?.task?.partition || '非 train'} Episode 可以保留和分析，但不能输入本次 Reflector。</div>}
    <details className="research-json"><summary>完整 Episode / Oracle / Replay JSON</summary><pre>{code({ status: episode.status, terminal_reason: episode.terminal_reason, outcome, replay: report?.replay, costs: link.costs, skill_injection_audit: link.skill_injection_audit, tool_executions: tools })}</pre></details>
  </article>;
}
function val(x: boolean | null | undefined) { return x == null ? 'UNKNOWN' : x ? '是' : '否'; }
function findModelCall(calls: AnyRow[], tool: AnyRow): AnyRow | undefined {
  const operation = tool.operation || {};
  for (const call of calls) {
    const choices = call.response?.choices;
    const message = Array.isArray(choices) && choices.length === 1 ? choices[0]?.message : undefined;
    for (const selected of message?.tool_calls || []) {
      const fn = selected?.function || {};
      if (fn.name !== operation.name) continue;
      try {
        const args = JSON.parse(fn.arguments || '{}');
        if (args.resource_id === operation.resource_id && args.scope_id === operation.scope_id) return call;
      } catch { /* Keep looking; the unparsed response remains visible in the archive. */ }
    }
  }
  return undefined;
}
function ReflectionView({ run, busy, onSelectReflection }: { run: Run | null; busy: boolean; onSelectReflection: (id: string, replaceLegacyEvidence?: boolean, timeoutSeconds?: number) => void; selectedEpisode: (id: string) => AnyRow | undefined }) {
  const [timeoutSeconds, setTimeoutSeconds] = useState(180);
  if (!run) return <EmptyStage title="还没有研究运行" text="先运行 Designer 并完成一条 train 分区的真人 S0 Episode。" />;
  if (run.input?.split !== 'train') return <EmptyStage title="此分区不能进入 Reflection" text="Validation 与 held-out 运行仍可用于查看和保存，但本工作台禁止将其证据发送给 Reflector。请创建 train 分区的人工探索运行。" />;
  const running = run.phases.reflection === 'RUNNING';
  const timeoutControl = <div className="research-field-row research-budget-fields"><label>Reflector 单次等待（秒）<input type="number" min={1} max={600} value={timeoutSeconds} disabled={busy || running} onChange={e => setTimeoutSeconds(Number(e.target.value))} /><small>默认 180 秒，最多 600 秒；只影响本次反思，不改变 Service 的超时设置。</small></label></div>;
  const choices = (replaceLegacyEvidence = false) => run.episodes.filter((ep: AnyRow) => ep.arm === 'S0' && ep.replay?.verified).map((ep: AnyRow) =>
    <div className="reflection-choice" key={ep.session_id}>
      <div><strong>S0 Episode · {ep.episode_hash.slice(0, 18)}…</strong>
        <span>业务目标 {ep.outcome?.business_goal_success === true ? '完成' : ep.outcome?.business_goal_success === false ? '未完成' : '未知'} · 过程 {ep.outcome?.process_errors?.status || 'UNKNOWN'} · 交互 {ep.outcome?.interaction_reliability?.status || 'UNKNOWN'} · {ep.costs?.service_calls} Service calls</span></div>
      <button className="research-secondary" disabled={busy || running} onClick={() => onSelectReflection(ep.session_id, replaceLegacyEvidence, timeoutSeconds)}><Sparkles size={15} />
        {busy || running ? 'Reflector 运行中…' : replaceLegacyEvidence ? '选择此证据，用修正后的映射反思一次' : run.reflection?.status === 'FAILED' ? '选择此证据，重新运行 Reflector' : '明确选作证据并运行 Reflector'}
      </button>
    </div>);
  if (!run.reflection) return <div className="research-panel result-panel"><div className="research-panel-heading"><div><span className="research-eyebrow">STAGE 05 · SERVICE REFLECTOR</span><h2>选择真实、可回放的 S0 证据进行一次反思</h2><p>人工 Episode 默认不参与训练；点击按钮将选择这条证据并调用一次模型。</p></div><Sparkles size={20} /></div>{timeoutControl}{choices()}{!run.episodes.some((ep: AnyRow) => ep.arm === 'S0' && ep.replay?.verified) && <EmptyStage title="没有可用于反思的 S0 Episode" text="先完成一条 S0 人工 Episode 并通过 Replay。" />}</div>;
  const reflection = run.reflection; const patch = reflection.patch; const candidate = run.candidate_skill;
  const legacyNoop = reflection.status === 'NOOP' && !reflection.evidence_protocol;
  const errorMessage = typeof reflection.error === 'string' ? (reflection.raw_call?.error?.message || reflection.error) : reflection.error?.message;
  return <div className="research-panel result-panel"><div className="research-panel-heading"><div><span className="research-eyebrow">REAL REFLECTION OUTPUT</span><h2>Service Reflector · {reflection.status}</h2><p>候选 Skill 不会修改默认 Library。静态合法也不代表能力提升。</p></div><StatusPill value={reflection.status} /></div>
    {errorMessage && <div className="research-warning"><CircleAlert size={15} />{errorMessage}</div>}
    {(reflection.status === 'FAILED' || legacyNoop) && timeoutControl}
    {reflection.status === 'FAILED' && <section><p>本次未生成有效 SkillPatch。原始失败记录将保留；明确选择证据后可重新调用一次模型，不会自动重试。</p>{choices()}</section>}
    {candidate?.skill && <SkillCard skill={candidate.skill} candidate={candidate} />}
    {patch?.action === 'NOOP' && <div className="research-notice"><ShieldCheck size={16} />Reflector 选择 NOOP，保留原 Skill。此判断的依据见下方模型理由与实际证据。</div>}
    {patch?.rationale && <p>{patch.rationale}</p>}
    {legacyNoop && <section><div className="research-warning"><CircleAlert size={15} />这次 NOOP 使用旧版证据映射。现已修正事件关联并补齐业务评价；可明确选择一条旧 Episode 重新反思一次。原始 NOOP 将完整保留，此操作会调用模型。</div>{choices(true)}</section>}
    {reflection.static_validation && <div className={`static-result ${reflection.static_validation.passed ? 'pass' : 'fail'}`}><ShieldCheck size={17} /><div><strong>静态校验 {reflection.static_validation.passed ? '通过' : '拒绝'}</strong><span>{reflection.static_validation.checks?.join(' · ')}</span>{reflection.static_validation.reasons?.map((reason: string) => <small key={reason}>{reason}</small>)}</div></div>}
    <details className="research-json"><summary>1 · Reflector 实际收到的 Replay 验证证据</summary><pre>{code(reflection.input)}</pre></details>
    <details className="research-json"><summary>2 · 真实模型的原始请求与输出</summary><pre>{code(reflection.raw_call)}</pre></details>
    <details className="research-json"><summary>3 · 完整 SkillPatch / 校验记录 / Snapshot 哈希</summary><pre>{code({ patch, input_hash: reflection.input_hash, evidence_protocol: reflection.evidence_protocol, source_episode_id: reflection.source_episode_id, candidate })}</pre></details>
    {!!run.reflection_history?.length && <details className="research-json"><summary>此前反思的原始记录（{run.reflection_history.length} 次，完整保留）</summary><pre>{code(run.reflection_history)}</pre></details>}
  </div>;
}
function SkillCard({ skill, candidate }: { skill: AnyRow; candidate: AnyRow }) { return <article className="skill-candidate"><div className="skill-heading"><div><span className="skill-badge">{candidate.status}</span><h3>{skill.id} <small>v{skill.version}</small></h3></div><code>snapshot {candidate.snapshot_hash?.slice(0, 18)}…</code></div><p>{skill.description}</p><div className="skill-meta"><span>触发：{skill.trigger?.keywords?.join(' / ') || skill.trigger?.intents?.join(' / ')}</span><span>作用域：{skill.scope?.domains?.join(', ')}</span><span>证据：{candidate.evidence_ids?.join(', ')}</span></div><div className="skill-steps"><div><strong>前置判断</strong>{(skill.preconditions || []).map((item: string, i: number) => <p key={i}>{item}</p>)}</div><div><strong>决策流程</strong>{(skill.procedure || []).map((item: string, i: number) => <p key={i}>{item}</p>)}</div><div><strong>后置检查 / 恢复</strong>{[...(skill.postconditions || []), ...(skill.recovery || [])].map((item: string, i: number) => <p key={i}>{item}</p>)}</div></div><div className="snapshot-line">父 Snapshot：<code>{candidate.parent_snapshot_hash}</code> · Patch：<code>{candidate.patch_hash}</code></div></article>; }
function SkillComparison({ run, runUrl }: { run: Run | null; runUrl: (arm: 'S0' | 'S1') => string }) {
  if (!run?.candidate_skill || !run.candidate_skill.skill) return <EmptyStage title="没有可注入的候选 Skill" text="Reflector 必须返回可解析 Patch，且该 Patch 通过静态检查。NOOP 和拒绝状态都不会生成 S1。" />;
  const candidate = run.candidate_skill;
  const hasS0 = run.episodes?.some((ep: AnyRow) => ep.arm === 'S0'); const hasS1 = run.episodes?.some((ep: AnyRow) => ep.arm === 'S1');
  const comparison = run.comparisons?.[0];
  return <div className="research-panel result-panel"><div className="research-panel-heading"><div><span className="research-eyebrow">STAGE 06 · TEMPORARY SKILL INJECTION</span><h2>S0 / S1 同世界人工 sanity check</h2><p>只改变候选 Skill Snapshot；没有自动修改正式 Library。人工回复路径可能不同，因此这不是受控动态 Gate。</p></div><StatusPill value={run.candidate_skill.status} /></div>
    <div className="candidate-injection-banner"><Sparkles size={17} /><div><strong>候选 Snapshot</strong><code>{run.candidate_skill.snapshot_hash}</code><span>注入发生在 OpenAIServiceAgent 的真实 system messages；每条 Service 请求记录都可核对。</span></div></div>
    <div className="ab-launch-row"><a className="research-secondary research-link-secondary" href={runUrl('S0')}><Play size={15} />{hasS0 ? '再运行 S0' : '启动 S0 · 不带 Skill'}<ExternalLink size={13} /></a><a className={`research-secondary research-link-secondary ${candidate.status === 'REJECTED' ? 'disabled-link' : ''}`} href={candidate.status === 'REJECTED' ? undefined : runUrl('S1')} onClick={e => candidate.status === 'REJECTED' && e.preventDefault()}><Play size={15} />{hasS1 ? '再运行 S1' : '启动 S1 · 注入候选 Skill'}<ExternalLink size={13} /></a></div>
    {comparison ? <><div className="comparison-banner"><CircleAlert size={16} /><span><strong>INCONCLUSIVE</strong> · 同一冻结任务上的两次真人互动，探索路径不保证完全相同；不可据此宣称 Skill 泛化或 RSI 成功。</span></div><div className="ab-grid">{(['S0', 'S1'] as const).map(arm => <div className="ab-column" key={arm}><h3>{arm} · {arm === 'S0' ? '无候选 Skill' : '候选 Skill'}</h3>{Object.entries(comparison[arm] || {}).map(([key, value]) => <div key={key}><span>{key}</span><strong>{String(value)}</strong></div>)}</div>)}</div></> : <div className="research-notice"><Clock3 size={16} />完成相同冻结场景的 S0 和 S1 Episode 后，系统才会形成一张定性对照表；缺少独立新任务时结论保持 INCONCLUSIVE。</div>}
    {run.candidate_skill.injection_warning && <div className="research-warning">{run.candidate_skill.injection_warning}</div>}
    {run.episodes?.filter((ep: AnyRow) => ep.arm === 'S1').map((ep: AnyRow) => <div key={ep.session_id} className="injection-proof"><CheckCircle2 size={16} /><span>S1 实际命中 {ep.skill_injection_audit?.injected_request_count ?? 0} 条 Service 模型请求；Skill IDs：{ep.skill_injection_audit?.matched_skill_ids?.join(', ') || '未命中'}。</span><JsonDetails title="实际注入请求 Hash 与 Skill Snapshot" value={ep.skill_injection_audit} /></div>)}
  </div>;
}
function HistoryView({ runs, run, onSelect, onExport }: { runs: Run[]; run: Run | null; onSelect: (id: string) => void; onExport: () => void }) {
  return <div className="research-panel result-panel"><div className="research-panel-heading"><div><span className="research-eyebrow">STAGE 07 · AUDIT ARCHIVE</span><h2>运行历史与当前结论</h2><p>完整 JSON 包含生成请求、构建与证明、人工参与标签、反思和候选 Snapshot。</p></div>{run && <button className="research-button" onClick={onExport}><Download size={15} />导出完整研究包</button>}</div>
    {run && <div className="research-lineage"><div><span>Designer Run</span><strong>{run.run_id}</strong><small>世界 {run.scenario?.public?.initial_world_hash?.slice(0, 16) || '未认证'}</small></div><ArrowRight size={15} /><div><span>冻结 Task</span><strong>{run.scenario?.task?.record_hash?.slice(0, 16) || '—'}</strong><small>{run.scenario?.task?.customer_request || '无任务'}</small></div><ArrowRight size={15} /><div><span>Human Episode</span><strong>{run.episodes?.length || 0} 条</strong><small>human · qualitative_manual_pilot</small></div><ArrowRight size={15} /><div><span>Skill Proposal</span><strong>{run.candidate_skill?.status || run.reflection?.status || '未运行'}</strong><small>{run.candidate_skill?.snapshot_hash?.slice(0, 16) || '无候选'}</small></div></div>}
    {run && <div className="research-conclusion"><ShieldCheck size={17} /><div><strong>本次可以得出的结论</strong><span>{run.candidate_skill?.status === 'INJECTED_FOR_TEST' ? '候选 Skill 已进入真实 Service 请求；需要多个独立任务和预先冻结的动态验收政策，才能判断能力改善。' : run.candidate_skill?.status === 'STATIC_VALID' ? '候选仅通过静态检查；尚未证明实际注入，也没有能力效果结论。' : run.reflection?.status === 'NOOP' ? 'Reflector 选择 NOOP，未生成候选 Skill；判断依据见反思记录。' : run.status === 'READY_FOR_CUSTOMER' ? 'Designer 任务已通过业务 Reference 认证；尚未验证自然对话可达性。' : '当前记录未完成所有阶段。未知或未运行的结果不会解释为通过。'}</span><small>该研究记录是人工探索，不适合作为论文中的 Benchmark 泛化或 RSI 能力提升证据。</small></div></div>}
    <div className="history-list"><h3>研究运行</h3>{runs.length ? runs.map(row => <button key={row.run_id} className={run?.run_id === row.run_id ? 'selected' : ''} onClick={() => onSelect(row.run_id)}><span><strong>{row.run_id}</strong><small>{row.input?.distribution} · seed {row.input?.seed} · {new Date(row.created_at).toLocaleString()}</small></span><StatusPill value={row.status} /><span>{row.episodes?.length || 0} Episodes</span><ArrowRight size={14} /></button>) : <p>还没有保存的研究运行。</p>}</div>
    {run && <details className="research-json"><summary>整份不可变研究导出中的关键原始数据</summary><pre>{code({ designer_run: run.designer_run, scenario: run.scenario, episodes: run.episodes?.map((ep: AnyRow) => ({ ...ep, report: '[完整 Episode 记录随导出保存]' })), reflection: run.reflection, reflection_history: run.reflection_history, candidate_skill: run.candidate_skill, comparisons: run.comparisons, journal: run.journal })}</pre></details>}
  </div>;
}
function EmptyStage({ title, text }: { title: string; text: string }) { return <div className="research-panel empty-stage"><div className="empty-stage-icon"><Wrench size={20} /></div><h2>{title}</h2><p>{text}</p></div>; }
