import { expect, test } from '@playwright/test';

test('failed reflection displays the provider error and waits for an explicit retry', async ({ page }) => {
  const failure = { status: 'FAILED', error: 'reflection_client_error',
    raw_call: { error: { code: 'PROVIDER_HTTP_ERROR', message: 'LLM endpoint returned HTTP 400; no retry performed' } } };
  const run = { run_id: 'rs-failed-ui-test', status: 'REFLECTION_COMPLETE', input: { split: 'train' },
    phases: { reflection: 'FAILED' }, model: { model: 'fixture' }, created_at: '2026-10-23T00:00:00Z',
    episodes: [{ session_id: 'saved-completed-episode', episode_hash: 'c'.repeat(64), arm: 'S0',
      replay: { verified: true }, costs: { service_calls: 28 }, outcome: { business_goal_success: true,
        process_errors: { status: 'FAIL' }, interaction_reliability: { status: 'PASS' } } }],
    reflection: failure, comparisons: [] };
  const submissions: any[] = [];
  await page.route('**/api/research/runs**', async route => {
    const request = route.request();
    if (request.method() === 'POST') submissions.push(request.postDataJSON());
    await route.fulfill({ json: new URL(request.url()).pathname === '/api/research/runs' ? { runs: [run] } : run });
  });
  await page.goto('/?mode=research&run_id=rs-failed-ui-test');
  await page.getByRole('navigation', { name: '研究阶段' }).getByRole('button', { name: /Reflector 提炼 Skill/ }).click();
  await expect(page.getByText('LLM endpoint returned HTTP 400; no retry performed', { exact: true })).toBeVisible();
  expect(submissions).toHaveLength(0);
  await page.getByRole('spinbutton', { name: 'Reflector 单次等待（秒）' }).fill('300');
  await page.getByRole('button', { name: '选择此证据，重新运行 Reflector' }).click();
  await expect.poll(() => submissions.length).toBe(2);
  expect(submissions[1]).toEqual({ session_id: 'saved-completed-episode',
    explicit_human_evidence_selection: true, replace_legacy_evidence: false, timeout_seconds: 300 });
});

test('Research Studio exposes the seven real workflow stages without running a model by itself', async ({ page }) => {
  await page.goto('/?mode=research');

  await expect(page.getByTestId('research-studio')).toBeVisible();
  await expect(page.getByRole('heading', { name: '从业务世界到可审计的 Skill 实验' })).toBeVisible();
  await expect(page.getByText('当前系统行为说明')).toBeVisible();
  await expect(page.getByRole('button', { name: /Designer 生成世界/ }).first()).toBeVisible();
  await expect(page.getByRole('button', { name: /Reflector 提炼 Skill/ }).first()).toBeVisible();
  await expect(page.getByText('工程机制跑通 ≠ 已证明 RSI 能力提升')).toBeVisible();

  // Visiting and navigating the workspace is offline; only the explicit
  // Designer submit action can create a billable model call.
  const history = await page.request.get('/api/research/runs');
  expect(history.ok()).toBeTruthy();
  await page.getByRole('navigation', { name: '研究阶段' }).getByRole('button', { name: /Reflector 提炼 Skill/ }).click();
  await expect(page.getByText(/先运行 Designer 并完成一条 train 分区的真人 S0 Episode/)).toBeVisible();
});

test('legacy NOOP offers an explicit evidence revision without silently calling a model', async ({ page }) => {
  const legacy = { status: 'NOOP', patch: { action: 'NOOP', rationale: 'Original recorded response.' } };
  const episode = (sessionId: string, hash: string, success: boolean) => ({
    session_id: sessionId, episode_hash: hash.repeat(64), arm: 'S0', replay: { verified: true },
    costs: { service_calls: success ? 28 : 23 }, outcome: { business_goal_success: success,
      process_errors: { status: 'FAIL' }, interaction_reliability: { status: success ? 'PASS' : 'FAIL' } },
  });
  let run: Record<string, any> = { run_id: 'rs-legacy-ui-test', status: 'REFLECTION_COMPLETE',
    input: { split: 'train' }, phases: { reflection: 'NOOP' }, model: { model: 'fixture' },
    created_at: '2026-10-23T00:00:00Z', episodes: [episode('session-incomplete', 'a', false),
      episode('session-completed', 'b', true)], reflection: legacy, comparisons: [] };
  const submissions: Array<{ path: string; body: any }> = [];
  await page.route('**/api/research/runs**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() === 'POST') {
      submissions.push({ path, body: request.postDataJSON() });
      if (path.endsWith('/reflect')) {
        run = { ...run, reflection: null, reflection_history: [legacy], status: 'REFLECTOR_RUNNING',
          phases: { reflection: 'RUNNING' } };
      }
    }
    await route.fulfill({ json: path === '/api/research/runs' ? { runs: [run] } : run });
  });
  await page.goto('/?mode=research&run_id=rs-legacy-ui-test');
  await page.getByRole('navigation', { name: '研究阶段' }).getByRole('button', { name: /Reflector 提炼 Skill/ }).click();
  await expect(page.getByRole('heading', { name: 'Service Reflector · NOOP' })).toBeVisible();
  await expect(page.getByText(/这次 NOOP 使用旧版证据映射/)).toBeVisible();
  const success = page.locator('.reflection-choice').filter({ hasText: 'bbbbbbbbbbbbbbbbbb' });
  await expect(success).toContainText('业务目标 完成 · 过程 FAIL · 交互 PASS');
  expect(submissions).toHaveLength(0);
  await success.getByRole('button', { name: '选择此证据，用修正后的映射反思一次' }).click();
  await expect.poll(() => submissions.length).toBe(2);
  expect(submissions[1].body).toEqual({ session_id: 'session-completed',
    explicit_human_evidence_selection: true, replace_legacy_evidence: true, timeout_seconds: 180 });
  expect(run.reflection_history).toEqual([legacy]);
  await expect(page.locator('.reflection-choice').first().getByRole('button')).toBeDisabled();
});
