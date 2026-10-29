import { test, expect, type Page } from '@playwright/test';
import { resolve } from 'node:path';
const output = resolve('../../outputs/web-four-domains');
const nav = (page: Page, name: string) => page.getByRole('navigation').getByRole('button', { name, exact: true }).click();
const state = async (page: Page, domain: string) => (await page.request.get(`/api/domains/${domain}/world`)).json();
async function open(page: Page, domain: string) {
  await page.goto(`/?domain=${domain}`);
  await expect(page.getByText('Python engine connected')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'World Overview', exact: true })).toBeVisible();
}

for (const domain of ['cloudops', 'fulfillment', 'stayops']) {
  test(`${domain}: actual frozen tasks, model evidence and replay`, async ({ page }) => {
    await open(page, domain);
    const original = await state(page, domain);
    await page.screenshot({ path: `${output}/${domain}-overview.png`, fullPage: true, animations: 'disabled' });
    await nav(page, 'World Graph');
    await expect(page.locator('.react-flow__node').first()).toBeVisible();
    await page.getByLabel('Graph scope').selectOption(domain === 'fulfillment' ? 'w0' : 'p0');
    await expect(page.locator('.graph-loading')).toHaveCount(0);
    await page.screenshot({ path: `${output}/${domain}-graph.png`, animations: 'disabled' });
    await nav(page, 'Benchmark & Pilots');
    const count = domain === 'cloudops' ? 9 : domain === 'fulfillment' ? 46 : 67;
    await expect(page.locator('.benchmark-instance-list button')).toHaveCount(count);
    await expect(page.locator('.pilot-panel tbody tr')).toHaveCount(domain === 'cloudops' ? 6 : 9);
    await page.screenshot({ path: `${output}/${domain}-benchmark.png`, fullPage: true, animations: 'disabled' });
    const width = await page.evaluate(() => ({ width: document.documentElement.scrollWidth, viewport: innerWidth }));
    expect(width.width).toBeLessThanOrEqual(width.viewport);
    await page.getByRole('button', { name: 'Inspect goal & certificates' }).click();
    await expect(page.getByRole('dialog')).toContainText('witnesses');
    await page.keyboard.press('Escape');
    await page.locator('.pilot-panel tbody tr').last().getByRole('button', { name: 'View trace' }).click();
    await expect(page.locator('.pilot-trace')).toContainText('Recorded execution');
    await page.getByRole('button', { name: 'customer', exact: true }).click();
    await expect(page.locator('.pilot-trace')).toContainText('customer_llm_records');
    await page.getByRole('button', { name: 'Replay recorded run' }).click();
    await expect(page.locator('.pilot-trace .replay-banner')).toContainText('"replay_verified": true');
    await expect(page.locator('.pilot-trace .replay-banner')).toContainText('"successful_business_violation": 0');
    expect((await state(page, domain)).state_hash).toBe(original.state_hash);
    await page.getByRole('button', { name: 'VERIFICATION.md', exact: true }).click();
    await expect(page.locator('.domain-report-text')).toContainText(domain === 'cloudops' ? '477' : domain === 'fulfillment' ? '550' : '605');
    await page.getByLabel('Benchmark suite').selectOption('discovery');
    await expect(page.locator('.benchmark-instance-list button')).toHaveCount(domain === 'cloudops' ? 2 : 1);
    const frozen = (await (await page.request.get(`/api/domains/${domain}/benchmark/suites/discovery`)).json()).instances[0];
    await page.getByRole('button', { name: 'Load for manual exploration' }).click();
    await expect(page.getByRole('heading', { name: 'Business Entities', exact: true })).toBeVisible();
    expect((await state(page, domain)).state_hash).toBe(frozen.initial_hash);
    await expect(page.locator('.scenario-notice')).toContainText('不产生正式 Benchmark 分数');
  });
}

test('hotel native operation form, actual state change and cross-domain persistence', async ({ page }) => {
  await open(page, 'stayops');
  const original = await state(page, 'stayops');
  await nav(page, 'Operation Playground');
  await page.getByLabel('Business operation').selectOption('mark_room_clean');
  await page.getByLabel('Target resource ID').fill('p0_standard_0');
  await page.getByRole('button', { name: 'Execute operation' }).click();
  await expect(page.getByTestId('domain-operation-result')).toContainText('legal_execution');
  const changed = await state(page, 'stayops');
  expect(changed.world.rooms.p0_standard_0.housekeeping).toBe('clean');
  expect(changed.state_hash).not.toBe(original.state_hash);
  await page.screenshot({ path: `${output}/stayops-actual-operation.png`, fullPage: true, animations: 'disabled' });
  await page.getByRole('button', { name: "Inspect this identity's Customer view" }).click();
  await expect(page.getByRole('dialog')).not.toContainText('simulated_identity_result');
  await page.keyboard.press('Escape');
  await page.getByLabel('Business domain').selectOption('cloudops');
  await expect(page.getByRole('heading', { name: 'World Overview', exact: true })).toBeVisible();
  await page.getByLabel('Business domain').selectOption('stayops');
  await expect(page.getByRole('heading', { name: 'World Overview', exact: true })).toBeVisible();
  expect((await state(page, 'stayops')).state_hash).toBe(changed.state_hash);
  await nav(page, 'Event Timeline');
  await page.getByRole('button', { name: 'Replay Trace', exact: true }).click();
  await expect(page.locator('.replay-banner')).toContainText('"matches_current_state": true');
  await page.getByLabel('Business domain').selectOption('saas');
  await expect(page.getByRole('heading', { name: 'World Overview', exact: true })).toBeVisible();
  await expect(page.getByRole('navigation').getByRole('button', { name: 'Benchmark v0.2' })).toBeVisible();
});

test('Fulfillment reference denial and Sandbox actual Oracle violation', async ({ page }) => {
  await open(page, 'fulfillment');
  await nav(page, 'Operation Playground');
  await page.getByLabel('Business operation').selectOption('confirm_order');
  await page.getByLabel('Target resource ID').fill('order0');
  await page.getByRole('button', { name: 'Execute operation' }).click();
  await expect(page.getByTestId('domain-operation-result')).toContainText('rejected_invalid_attempt');
  await page.getByRole('button', { name: 'Sandbox', exact: true }).click();
  await page.getByRole('button', { name: 'Execute operation' }).click();
  await expect(page.getByTestId('domain-operation-result')).toContainText('successful_business_violation');
  expect((await state(page, 'fulfillment')).world.orders.order0.status).toBe('confirmed');
  await page.screenshot({ path: `${output}/fulfillment-sandbox-evidence.png`, fullPage: true, animations: 'disabled' });
  await page.getByRole('button', { name: 'Reset World', exact: true }).click();
  await expect(page.getByTestId('domain-operation-result')).toHaveCount(0);
  expect((await state(page, 'fulfillment')).world.orders.order0.status).toBe('draft');
});
