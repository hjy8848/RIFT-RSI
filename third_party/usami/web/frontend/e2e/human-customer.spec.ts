import { expect, test } from '@playwright/test';

test('Human Customer Playground loads the certified scene and gates runs on real model configuration', async ({ page }) => {
  await page.goto('/?mode=human');

  await expect(page.getByTestId('human-playground')).toBeVisible();
  await expect(page.getByRole('heading', { name: '真人顾客实验台' })).toBeVisible();
  await expect(page.getByRole('button', { name: /Refund a damaged headset/ })).toBeVisible();
  await expect(page.getByText('Reference 任务可达性已证明')).toBeVisible();
  await expect(page.getByText('Please refund my damaged headphones.', { exact: true })).toBeVisible();
  await expect(page.getByTestId('model-config')).toBeVisible();

  // The browser test must never create a billable request or silently switch
  // to the scripted test adapter when no real Service configuration exists.
  const config = await page.request.get('/api/human/config').then(response => response.json());
  if (!config.configured) {
    await expect(page.getByTestId('start-human-episode')).toBeDisabled();
    await expect(page.getByText(/不会降级成脚本客服/)).toBeVisible();
  }
});

test('restores the same pending customer turn after reload and sends only the selected intent', async ({ page }) => {
  const scenario = {
    scenario_id: 'refund-damaged-headphones', title: 'Refund a damaged headset',
    description: 'A damaged headset refund.', customer_request: 'Please refund my damaged headphones.',
    customer: { id: 'customer-1', name: 'Customer' },
    orders: [{ order_id: 'order-1', delivered_at: '2026-10-20T00:00:00Z', paid_cents: 20000,
      lines: [{ line_id: 'line-1', product: 'headphones', attributes: {}, quantity: 1,
        paid_cents: 20000, issue: 'damaged' }] }],
    world_version: 'commerce-2.0.0', initial_world_hash: 'a'.repeat(64),
    world_build_verified: true, task_certificate_status: 'CERTIFIED', task_certificate_verified: true,
  };
  const pendingSession = {
    session_id: 'saved-test-session', status: 'awaiting_customer', session_version: 0,
    scenario, timeline: [
      { kind: 'customer', source: 'frozen_request', text: scenario.customer_request },
      { kind: 'service', text: 'I can refund ¥200.00. Do you accept?', offer: {
        remedy: 'refund', quantity: 1, refund_cents: 20000, operation: 'commit_accepted_quote',
      } },
    ],
    pending_turn: { turn_id: 'turn-id-0123456789', session_version: 0, offer: {
      remedy: 'refund', quantity: 1, refund_cents: 20000, operation: 'commit_accepted_quote',
    } },
    model_name: 'test-display-only', model_calls_limit: 32, tool_calls: 0, error: null,
    study_type: 'qualitative_manual_pilot', customer_source: 'human',
  };
  let postedTurn: Record<string, unknown> | null = null;
  await page.route('**/api/human/**', async route => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === '/api/human/config') {
      await route.fulfill({ json: { configured: true, model: 'test-display-only',
        service: 'real OpenAIServiceAgent', credential_source_configured: true, message: 'test fixture' } });
    } else if (url.pathname === '/api/human/scenarios') {
      await route.fulfill({ json: { scenarios: [scenario] } });
    } else if (url.pathname === '/api/human/sessions/saved-test-session' && request.method() === 'GET') {
      await route.fulfill({ json: pendingSession });
    } else if (url.pathname === '/api/human/sessions/saved-test-session/turn' && request.method() === 'POST') {
      postedTurn = request.postDataJSON();
      await route.fulfill({ json: { ...pendingSession, status: 'service_running', pending_turn: null } });
    } else {
      await route.fulfill({ status: 404, json: { detail: { message: 'unexpected test route' } } });
    }
  });
  await page.addInitScript(() => sessionStorage.setItem('usami-human-session-id', 'saved-test-session'));
  await page.goto('/?mode=human');
  await expect(page.getByRole('button', { name: '接受报价' })).toBeVisible();
  await expect(page.getByTestId('goal-boundary-note')).toContainText('不会自动加入评分目标');
  await page.reload();
  await expect(page.getByRole('button', { name: '接受报价' })).toBeVisible();
  expect(postedTurn).toBeNull();

  await page.getByRole('button', { name: '接受报价' }).click();
  await expect.poll(() => postedTurn).not.toBeNull();
  expect(postedTurn).toMatchObject({ turn_id: 'turn-id-0123456789', session_version: 0, intent: 'accept_offer' });
  expect(postedTurn).not.toHaveProperty('quote_id');
  expect(postedTurn).not.toHaveProperty('accept_quote_id');
});
