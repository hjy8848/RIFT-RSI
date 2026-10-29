import { test, expect, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const screenshots = resolve('../../outputs/web');
const nav = (page: Page, label: string) => page.getByRole('navigation').getByRole('button', { name: label, exact: true }).click();
const state = async (page: Page) => (await page.request.get('/api/world')).json();
const result = (page: Page) => page.getByTestId('operation-result');

test.beforeEach(async ({ page }) => {
  // Legacy regression coverage is explicitly pinned to the v0.1 world.
  await page.request.post('/api/sessions', { data: { seed: 42 } });
  await page.goto('/');
  await expect(page.getByText('Python engine connected')).toBeVisible();
});

test('real seed, organization details, invoices and generated relationship graph', async ({ page }) => {
  const world = await state(page);
  expect(world.world.seed).toBe(42);
  await expect(page.locator('.stat-card > strong')).toHaveText(['3', '8', '1', '12']);
  await page.screenshot({ path: `${screenshots}/overview.png`, fullPage: true });
  await page.locator('.org-card').filter({ hasText: 'org_001' }).click();
  await expect(page.locator('.member-row')).toHaveCount(world.organizations[1].memberships.length);
  await expect(page.locator('tbody tr')).toHaveCount(4);
  const invoice = Object.values(world.world.invoices).find((i: any) => i.org_id === 'org_001') as any;
  await page.getByRole('button', { name: `Inspect ${invoice.id}` }).click();
  await expect(page.getByRole('dialog')).toContainText(invoice.id);
  await page.keyboard.press('Escape');
  await nav(page, 'World Graph');
  await expect(page.locator('.react-flow__node')).toHaveCount(26);
  const graph = await (await page.request.get('/api/world/graph')).json();
  await expect(page.locator('.react-flow__edge')).toHaveCount(graph.edges.length);
  await page.getByLabel('Graph organization').selectOption('org_001');
  const focused = await (await page.request.get('/api/world/graph?org_id=org_001')).json();
  await expect(page.locator('.react-flow__node')).toHaveCount(focused.nodes.length);
  const node = page.locator('.react-flow__node[data-id="org_001"]');
  await node.click();
  await expect(page.getByRole('dialog')).toContainText('org_001');
  await page.keyboard.press('Escape');
  const before = await node.getAttribute('style');
  const box = (await node.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 50, box.y + box.height / 2 + 40, { steps: 10 });
  await page.mouse.up();
  expect(await node.getAttribute('style')).not.toBe(before);
  await page.getByRole('button', { name: 'zoom in' }).click();
  await page.getByRole('button', { name: 'fit view' }).click();
  await page.screenshot({ path: `${screenshots}/graph.png`, fullPage: true });
});

test('legal change, rejected change and sandbox invariant violation use actual engine results', async ({ page }) => {
  await nav(page, 'Operation Playground');
  await page.getByLabel('Operation organization').selectOption('org_001');
  await page.getByLabel('Target plan').selectOption('team');
  await page.getByRole('button', { name: 'Execute operation' }).click();
  await expect(result(page).getByTestId('operation-correct')).toHaveText('Yes');
  expect((await state(page)).world.subscriptions.sub_001.plan_id).toBe('team');
  await expect(result(page)).toContainText('subscriptions.sub_001.plan_id');
  await page.getByRole('button', { name: 'Reset World', exact: true }).click();
  await expect(page.getByTestId('operation-result')).toHaveCount(0);
  await page.getByLabel('Operation organization').selectOption('org_001');
  const initial = await state(page);
  await page.getByLabel('Business operation').selectOption('change_seats');
  await page.getByLabel('Target seats').fill('6');
  await page.getByRole('button', { name: 'Execute operation' }).click();
  await expect(result(page)).toContainText('Rejected');
  await expect(result(page).getByTestId('valid-request')).toHaveText('No');
  await expect(result(page).getByTestId('operation-correct')).toHaveText('Yes');
  expect((await state(page)).state_hash).toBe(initial.state_hash);
  await page.getByRole('button', { name: 'Sandbox', exact: true }).click();
  await expect(page.locator('.sandbox-note')).toContainText('不代表真实 Agent 已发现漏洞');
  await page.getByRole('button', { name: 'Execute operation' }).click();
  await expect(result(page).getByTestId('operation-correct')).toHaveText('No');
  await expect(result(page)).toContainText('Executed');
  await expect(result(page)).toContainText('SEAT_LIMIT_EXCEEDED');
  expect((await state(page)).world.subscriptions.sub_001.seats).toBe(6);
  await page.screenshot({ path: `${screenshots}/sandbox.png`, fullPage: true });
  await result(page).getByRole('tab', { name: 'Before', exact: true }).click();
  await expect(result(page).locator('.json-view:visible')).toContainText('"seats": 3');
  await result(page).getByRole('tab', { name: 'After', exact: true }).click();
  await expect(result(page).locator('.json-view:visible')).toContainText('"seats": 6');
  await page.getByRole('button', { name: 'Reset World', exact: true }).click();
  expect((await state(page)).state_hash).toBe(initial.state_hash);
});

test('member permissions are organization scoped and Oracle detects successful forbidden cancellation', async ({ page }) => {
  await nav(page, 'Operation Playground');
  await page.getByLabel('Operation organization').selectOption('org_001');
  await page.getByLabel('Business operation').selectOption('change_member_role');
  await page.getByLabel('Target member').selectOption('membership_001_user_005');
  await page.getByLabel('New role').selectOption('member');
  await page.getByRole('button', { name: 'Execute operation' }).click();
  await expect(result(page).getByTestId('operation-correct')).toHaveText('Yes');
  await page.getByLabel('Actor', { exact: true }).selectOption('user_005');
  await page.getByLabel('Business operation').selectOption('cancel_renewal');
  await page.getByRole('button', { name: 'Execute operation' }).click();
  await expect(result(page)).toContainText('PERMISSION_DENIED');
  expect((await state(page)).world.subscriptions.sub_001.auto_renew).toBe(true);
  await page.getByRole('button', { name: 'Sandbox', exact: true }).click();
  await page.getByRole('button', { name: 'Execute operation' }).click();
  await expect(result(page).getByTestId('operation-correct')).toHaveText('No');
  await expect(result(page)).toContainText('subscriptions.sub_001.auto_renew');
  expect((await state(page)).world.subscriptions.sub_001.auto_renew).toBe(false);
});

test('refund form uses a real invoice and records the full refund history', async ({ page }) => {
  const catalog = await (await page.request.get('/api/scenarios')).json();
  const scenario = catalog.scenarios.find((s: any) => s.steps.length === 1 && s.steps[0].expected_allowed
    && s.steps[0].request.operation.name === 'refund_invoice');
  const request = scenario.steps[0].request;
  await nav(page, 'Operation Playground');
  await page.getByLabel('Operation organization').selectOption(request.org_id);
  await page.getByLabel('Actor', { exact: true }).selectOption(request.requester_id);
  await page.getByLabel('Business operation').selectOption('refund_invoice');
  await page.getByLabel('Invoice', { exact: true }).selectOption(request.operation.resource_id);
  await expect(page.getByLabel('Refund amount (cents)')).toHaveValue(String(request.operation.params.amount_cents));
  await expect(page.getByLabel('Payment method', { exact: true })).toHaveValue(request.operation.params.payment_method);
  await page.getByRole('button', { name: 'Execute operation' }).click();
  await expect(result(page).getByTestId('operation-correct')).toHaveText('Yes');
  const invoice = (await state(page)).world.invoices[request.operation.resource_id];
  expect(invoice.status).toBe('refunded');
  expect(invoice.refunds).toHaveLength(1);
  expect(invoice.refunds[0].amount_cents).toBe(request.operation.params.amount_cents);
});

test('multistep scenario, step evidence, exported trace and deterministic replay', async ({ page }) => {
  await nav(page, 'Scenarios');
  await expect(page.locator('.scenario-step')).toHaveCount(5);
  await page.getByRole('button', { name: 'Run Scenario', exact: true }).click();
  await expect(page.locator('.scenario-report')).toContainText('5 / 5 requests correct');
  await page.locator('.scenario-step').nth(2).click();
  await expect(result(page)).toContainText('Rejected');
  await expect(result(page)).toContainText('No fields changed.');
  await expect(result(page).getByTestId('operation-correct')).toHaveText('Yes');
  await page.locator('.scenario-step').nth(4).click();
  await expect(result(page)).toContainText('subscriptions.sub_001.plan_id');
  await page.screenshot({ path: `${screenshots}/scenario.png`, fullPage: true });
  const before = await state(page);
  await nav(page, 'Event Timeline');
  await expect(page.locator('.timeline-event')).toHaveCount(before.events.length);
  await page.locator('.event-id').first().click();
  await expect(page.getByRole('dialog')).toContainText('generation');
  await page.keyboard.press('Escape');
  const downloaded = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export Trace JSON', exact: true }).click();
  const trace = JSON.parse(await readFile((await (await downloaded).path())!, 'utf8'));
  expect(trace.events.filter((e: any) => e.kind === 'operation')).toHaveLength(5);
  expect(trace.events.at(-1).kind).toBe('final');
  await page.getByRole('button', { name: 'Replay Trace', exact: true }).click();
  await expect(page.locator('.replay-banner')).toContainText('Deterministic replay verified');
  await expect(page.locator('.replay-banner')).toContainText('Yes');
  expect((await state(page)).state_hash).toBe(before.state_hash);
  await page.screenshot({ path: `${screenshots}/timeline.png`, fullPage: true });
});

test('seed regenerates deterministically and separate browser sessions stay isolated', async ({ page, browser }) => {
  const initial = await state(page);
  const other = await browser.newContext();
  const second = await other.newPage();
  await second.request.post('http://127.0.0.1:5173/api/sessions', { data: { seed: 42 } });
  await second.goto('http://127.0.0.1:5173');
  await expect(second.getByText('Python engine connected')).toBeVisible();
  await page.getByLabel('Seed', { exact: true }).fill('71');
  await page.getByRole('button', { name: 'Generate World', exact: true }).click();
  await expect(page.locator('.world-ident')).not.toContainText(initial.world.world_id);
  const changed = await state(page);
  expect(changed.world.seed).toBe(71);
  expect((await state(second)).state_hash).toBe(initial.state_hash);
  await page.getByLabel('Seed', { exact: true }).fill('42');
  await page.getByRole('button', { name: 'Generate World', exact: true }).click();
  await expect(page.locator('.world-ident')).toContainText(initial.world.world_id);
  expect((await state(page)).world).toEqual(initial.world);
  await other.close();
});

test('all six desktop pages stay within the viewport', async ({ page }) => {
  const width = () => page.evaluate(() => ({ view: window.innerWidth, body: document.documentElement.scrollWidth }));
  expect((await width()).body).toBeLessThanOrEqual(1440);
  for (const label of ['Organizations', 'World Graph', 'Operation Playground', 'Scenarios', 'Event Timeline']) {
    await nav(page, label);
    await expect(page.getByRole('heading', { name: label, exact: true })).toBeVisible();
    if (label === 'World Graph') await expect(page.locator('.react-flow__node')).toHaveCount(26);
    if (label === 'Scenarios') await expect(page.locator('.scenario-step')).toHaveCount(5);
    expect((await width()).body, label).toBeLessThanOrEqual(1440);
  }
});
