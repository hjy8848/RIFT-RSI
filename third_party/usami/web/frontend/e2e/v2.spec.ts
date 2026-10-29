import { test, expect, type Page } from '@playwright/test';
import { resolve } from 'node:path';
const nav = (page: Page, label: string) => page.getByRole('navigation').getByRole('button', { name: label, exact: true }).click();
const state = async (page: Page) => (await page.request.get('/api/world')).json();
const result = (page: Page) => page.getByTestId('operation-result');
const output = resolve('../../outputs/web-v2');
async function workflow(page: Page, name: string) {
  await page.getByLabel('Advanced generator settings').click();
  await page.getByLabel('Business workflow').selectOption(name);
  await page.getByRole('button', { name: 'Apply & generate' }).click();
  await expect(page.getByLabel('Business workflow')).toHaveCount(0);
  await nav(page, 'Operation Playground');
}
async function operation(page: Page, name: string) { await page.getByLabel('Business operation').selectOption(name); }
async function execute(page: Page) {
  const response = page.waitForResponse(r => r.url().endsWith('/api/operations') && r.request().method() === 'POST');
  await page.getByRole('button', { name: 'Execute operation' }).click();
  expect((await response).status()).toBe(200);
  await expect(result(page)).toBeVisible();
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await expect(page.getByText('Python engine connected')).toBeVisible();
});

test('v2 default, lifecycle capacity, Sandbox detection and witness replay', async ({ page }) => {
  expect((await state(page)).world.rules.version).toBe('saas-rules-0.2.0');
  await workflow(page, 'seat_activate');
  const initial = await state(page);
  const member = Object.values(initial.world.memberships).find((m: any) => m.org_id === 'org_000' && m.status === 'pending') as any;
  await operation(page, 'activate_member');
  await page.getByLabel('Target member').selectOption(member.id);
  await execute(page);
  await expect(result(page)).toContainText('NO_AVAILABLE_SEATS');
  expect((await state(page)).state_hash).toBe(initial.state_hash);
  await page.getByRole('button', { name: 'Sandbox', exact: true }).click();
  await execute(page);
  await expect(result(page).getByTestId('operation-correct')).toHaveText('No');
  await page.getByRole('button', { name: 'Reset World', exact: true }).click();
  await nav(page, 'Scenarios');
  await expect(page.locator('.scenario-detail')).toContainText('v0.2 Reference Witness');
  await page.getByRole('button', { name: 'Reference', exact: true }).click();
  await page.getByRole('button', { name: 'Run Scenario' }).click();
  await expect(page.locator('.scenario-report')).toContainText('Pass');
  expect((await state(page)).world.memberships[member.id].status).toBe('active');
  await nav(page, 'Event Timeline');
  await page.getByRole('button', { name: 'Replay Trace', exact: true }).click();
  await expect(page.locator('.replay-banner')).toContainText('Yes');
  await page.screenshot({ path: `${output}/lifecycle-replay.png`, fullPage: true });
});

test('failed payment restoration, idempotent retry and ledger graph', async ({ page }) => {
  await workflow(page, 'failed_restore');
  const initial = await state(page);
  const invoice = Object.values(initial.world.invoices).find((i: any) => i.org_id === 'org_000' && i.purpose === 'renewal') as any;
  await operation(page, 'pay_invoice');
  await page.getByLabel('Invoice', { exact: true }).selectOption(invoice.id);
  await page.getByLabel('Payment channel').selectOption('card_visa');
  await page.getByLabel('Idempotency key').fill('ui-payment');
  await execute(page);
  await expect(result(page).getByTestId('operation-correct')).toHaveText('Yes');
  const paid = await state(page);
  await execute(page);
  expect((await state(page)).world.invoices[invoice.id].payments).toEqual(paid.world.invoices[invoice.id].payments);
  await operation(page, 'reactivate_subscription');
  await page.getByLabel('Invoice', { exact: true }).selectOption(invoice.id);
  await execute(page);
  expect((await state(page)).world.subscriptions.sub_000.status).toBe('active');
  await nav(page, 'Organizations');
  await expect(page.locator('.subscription-panel')).toContainText(invoice.id);
  await expect(page.locator('.invoice-panel')).toContainText('Payment / refund ledger');
  await nav(page, 'World Graph');
  const graph = await (await page.request.get('/api/world/graph')).json();
  await expect(page.locator('.react-flow__node')).toHaveCount(graph.nodes.length);
  expect(graph.nodes.some((n: any) => n.kind === 'payment')).toBeTruthy();
  await page.screenshot({ path: `${output}/billing-graph.png`, fullPage: true });
});

test('partial refunds preserve original method and remaining balance', async ({ page }) => {
  await workflow(page, 'partial_refund');
  const initial = await state(page);
  const invoice = Object.values(initial.world.invoices).find((i: any) => i.org_id === 'org_000' && i.status === 'paid' && i.refundable_amount_cents > 0) as any;
  await operation(page, 'refund_invoice');
  await page.getByLabel('Invoice', { exact: true }).selectOption(invoice.id);
  await page.getByLabel('Refund amount (cents)').fill('1');
  await page.getByLabel('Idempotency key').fill('ui-refund');
  await execute(page);
  await expect(result(page).getByTestId('operation-correct')).toHaveText('Yes');
  const updated = await state(page);
  expect(updated.world.invoices[invoice.id].refundable_amount_cents).toBe(invoice.refundable_amount_cents - 1);
  expect(updated.world.invoices[invoice.id].status).toBe('partially_refunded');
  await execute(page);
  expect((await state(page)).state_hash).toBe(updated.state_hash);
});

test('frozen two-track library and genuine pilot replay stay separate from manual exploration', async ({ page }) => {
  await nav(page, 'Benchmark v0.2');
  await expect(page.locator('.benchmark-instance-list button')).toHaveCount(40);
  await expect(page.locator('.benchmark-page .stat-card strong')).toHaveText(['278', '21', '28', '80']);
  await page.screenshot({ path: `${output}/benchmark-overview.png` });
  const width = await page.evaluate(() => ({ body: document.documentElement.scrollWidth, viewport: innerWidth }));
  expect(width.body).toBeLessThanOrEqual(width.viewport);
  await page.getByRole('button', { name: 'Inspect goal & certificates' }).click();
  await expect(page.getByRole('dialog')).toContainText('dependency_edges');
  await page.keyboard.press('Escape');
  await page.locator('.pilot-panel tbody tr').last().getByRole('button', { name: 'View trace' }).click();
  await expect(page.locator('.pilot-trace')).toContainText('Recorded execution');
  await page.getByRole('button', { name: 'Replay recorded run' }).click();
  await expect(page.locator('.pilot-trace .replay-banner')).toContainText('"replay_verified": true');
  await expect(page.locator('.pilot-trace .replay-banner')).toContainText('"successful_business_violations": 1');
  await page.screenshot({ path: `${output}/benchmark-pilot.png`, fullPage: true });
  await page.getByLabel('Benchmark suite').selectOption('discovery');
  await expect(page.locator('.benchmark-instance-list button')).toHaveCount(6);
  await expect(page.locator('.benchmark-detail')).toContainText('sandbox');
  const frozen = (await (await page.request.get('/api/benchmark/suites/discovery')).json()).instances[0];
  await page.getByRole('button', { name: 'Load for manual exploration' }).click();
  await expect(page.locator('.scenario-notice')).toContainText('manual exploration');
  expect((await state(page)).state_hash).toBe(frozen.initial_state_hash);
  await page.screenshot({ path: `${output}/frozen-world.png`, fullPage: true });
});

test('invitation, activation, suspension and removal all use real member IDs', async ({ page }) => {
  await workflow(page, 'invite_activate');
  const initial = await state(page);
  const target = Object.keys(initial.world.users).find(id => !Object.values(initial.world.memberships).some((m: any) => m.org_id === 'org_000' && m.user_id === id))!;
  await operation(page, 'invite_member');
  await page.getByLabel('Directory user').selectOption(target);
  await page.getByLabel('Invitation role').selectOption('member');
  await execute(page);
  const invited = await state(page);
  const membership = Object.values(invited.world.memberships).find((m: any) => m.org_id === 'org_000' && m.user_id === target) as any;
  expect(membership.status).toBe('pending');
  await operation(page, 'change_seats');
  await page.getByLabel('Target seats').fill(String(initial.world.subscriptions.sub_000.seats + 1));
  await execute(page);
  await operation(page, 'activate_member');
  await page.getByLabel('Target member').selectOption(membership.id);
  await execute(page);
  expect((await state(page)).world.memberships[membership.id].status).toBe('active');
  await operation(page, 'deactivate_member');
  await execute(page);
  expect((await state(page)).world.memberships[membership.id].status).toBe('inactive');
  await operation(page, 'remove_member');
  await execute(page);
  expect((await state(page)).world.memberships[membership.id]).toBeUndefined();
  await expect(result(page).getByTestId('operation-correct')).toHaveText('Yes');
});

test('new renewal invoice can be selected, paid and used to extend the cycle', async ({ page }) => {
  await workflow(page, 'renew_cycle');
  const initial = await state(page);
  await operation(page, 'issue_renewal');
  await execute(page);
  const issued = await state(page);
  const invoice = Object.values(issued.world.invoices).find((i: any) => !(i.id in initial.world.invoices) && i.org_id === 'org_000') as any;
  expect(invoice).toBeTruthy();
  await result(page).getByText('Tool / engine return data').click();
  await expect(result(page)).toContainText(invoice.id);
  await operation(page, 'pay_invoice');
  await page.getByLabel('Invoice', { exact: true }).selectOption(invoice.id);
  await page.getByLabel('Idempotency key').fill('new-cycle');
  await execute(page);
  await operation(page, 'renew_subscription');
  await page.getByLabel('Invoice', { exact: true }).selectOption(invoice.id);
  await execute(page);
  const renewed = await state(page);
  expect(renewed.world.subscriptions.sub_000.service_invoice_id).toBe(invoice.id);
  expect(renewed.world.subscriptions.sub_000.period_end > initial.world.subscriptions.sub_000.period_end).toBeTruthy();
  await expect(result(page).getByTestId('operation-correct')).toHaveText('Yes');
});
