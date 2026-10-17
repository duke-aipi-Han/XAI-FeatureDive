import { expect, test } from './fixtures';
import { writeFile } from 'node:fs/promises';
import type { ExperimentResult } from '../src/types';

async function loaded(page: import('@playwright/test').Page) {
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Start experiment' })).toBeVisible();
  await expect(page.getByTestId('current-margin')).toHaveText(/\d/);
}
async function sample(page: import('@playwright/test').Page, id: string) {
  await page.getByLabel('Drawing', { exact: true }).selectOption(id);
  const response = page.waitForResponse(response => response.url().endsWith('/api/infer') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Start experiment' }).click();
  return (await (await response).json()) as ExperimentResult;
}

test('Phase 3 gate: drawing → Conv2 → channel → ablate → real scores → exact reset', async ({ page }) => {
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await loaded(page);
  const baseline = await sample(page, 'airplane:142997');
  await page.getByRole('button', { name: 'Layer 2 · CONV2 · 16 feature maps' }).click();
  await page.getByRole('button', { name: 'Select channel 1', exact: true }).click();
  const response = page.waitForResponse(response => response.url().endsWith('/api/intervene'));
  await page.getByRole('button', { name: 'Switch off channel 1', exact: true }).click();
  const modified = await (await response).json() as ExperimentResult;
  expect(modified.request.interventions).toEqual([{ type: 'channel_ablation', layer: 'conv2', channel: 1 }]);
  expect(modified.baseline.logits).toEqual(baseline.baseline.logits);
  expect(modified.modified.logits).not.toEqual(baseline.modified.logits);
  await expect(page.getByTestId('current-margin')).toHaveText(modified.modified.target_logit_margin.toFixed(3));
  await expect(page.getByTestId('logit-airplane')).toHaveText(modified.modified.logits[2].toFixed(3));
  await expect(page.getByRole('img', { name: 'conv2 channel 1 current activation', exact: true })).toBeVisible();
  await expect(page.locator('.channel-inspector')).toContainText('Mean activity: 0.0000');
  await page.getByRole('button', { name: 'Baseline', exact: true }).click();
  await expect(page.locator('.channel-inspector')).not.toContainText('Mean activity: 0.0000');
  await page.getByRole('button', { name: 'Current', exact: true }).click();
  await page.getByRole('heading', { name: 'Look deeper.' }).click();
  expect(await page.locator('.skip-link').evaluate(element => getComputedStyle(element).top)).toBe('-100px');
  await page.screenshot({ path: '../results/arcade-ux/conv2-ablation.png', fullPage: true });
  await writeFile('../results/arcade-ux/gate-experiment.json', JSON.stringify({ baseline, modified }, null, 2) + '\n');
  await page.getByRole('button', { name: 'Reset & try again' }).click();
  await expect(page.getByTestId('current-margin')).toHaveText(baseline.modified.target_logit_margin.toFixed(3));
  await expect(page.getByTestId('logit-airplane')).toHaveText(baseline.modified.logits[2].toFixed(3));
  await expect(page.getByTestId('margin-change')).toHaveText('+0.000 logit units');
  expect(errors).toEqual([]);
});

test('input tools use gradients and real selected pixels; reset restores the original state', async ({ page }) => {
  await loaded(page); const original = await sample(page, 'fish:2617');
  await page.getByText('More drawing tools',{exact:true}).click();
  const saliency = page.waitForResponse(response => response.url().endsWith('/api/saliency'));
  await page.getByRole('button', { name: /Saliency Scope/ }).click();
  const attribution = await (await saliency).json();
  expect(attribution.modified_scores).toHaveLength(28);
  await expect(page.getByText(/overlay contrast scaled to the map maximum/)).toBeVisible();
  await page.getByLabel('Pixel grid').check();
  await page.getByRole('button', { name: /Controlled Occlusion/ }).click();
  const canvas = page.getByRole('group', { name: /28 by 28 drawing/ });
  const bounds = (await canvas.boundingBox())!;
  const response = page.waitForResponse(response => response.url().endsWith('/api/intervene'));
  await canvas.click({ position: { x: bounds.width * 8.5 / 28, y: bounds.height * 12.5 / 28 } });
  const result = await (await response).json() as ExperimentResult;
  expect(result.request.interventions[0]).toMatchObject({ type: 'controlled_occlusion', region: { rectangle: { x: 8, y: 12, width: 4, height: 4 } }, replacement_rgb: [128, 128, 128] });
  await expect(page.getByTestId('current-margin')).toHaveText(result.modified.target_logit_margin.toFixed(3));
  const swap = page.waitForResponse(response => response.url().endsWith('/api/intervene'));
  await page.getByRole('button', { name: 'Change background to yellow' }).click(); await swap;
  await page.getByRole('button', { name: 'Reset & try again' }).click();
  await expect(page.getByTestId('current-margin')).toHaveText(original.modified.target_logit_margin.toFixed(3));
  await page.getByRole('button', { name: /Pixel Eraser/ }).click();
  const erased = page.waitForResponse(response => response.url().endsWith('/api/intervene'));
  await canvas.click({ position: { x: bounds.width * 16.5 / 28, y: bounds.height * 16.5 / 28 } });
  const erasure = await (await erased).json() as ExperimentResult;
  expect(erasure.request.interventions.at(-1)?.type).toBe('drawing_erasure');
});

test('a model or target draft does not silently replace the active baseline', async ({ page }) => {
  await loaded(page);
  const originalMargin = await page.getByTestId('current-margin').textContent();
  await page.getByText('More settings',{exact:true}).click();
  await page.getByLabel('Model', { exact: true }).selectOption('shortcut');
  await page.getByLabel('Target class', { exact: true }).selectOption('airplane');
  await expect(page.getByTestId('current-margin')).toHaveText(originalMargin!);
  const response = page.waitForResponse(response => response.url().endsWith('/api/infer') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Start experiment' }).click();
  const result = await (await response).json() as ExperimentResult;
  expect(result.request.model_id).toBe('shortcut'); expect(result.request.target_class).toBe('airplane');
  await expect(page.getByTestId('current-margin')).toHaveText(result.modified.target_logit_margin.toFixed(3));
});

test('dense/output views, keyboard navigation, and download work with measured data', async ({ page }) => {
  await loaded(page);
  await page.getByRole('button', { name: 'Hidden · DENSE · 32 neurons' }).click();
  await expect(page.locator('.neuron')).toHaveCount(32);
  await page.getByRole('button', { name: 'Answer · OUTPUT · 3 classes' }).click();
  await expect(page.getByRole('heading', { name: 'Read the decision.' })).toBeVisible();
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export experiment JSON' }).click();
  expect((await download).suggestedFilename()).toMatch(/^featuredive-.*\.json$/);
  await page.locator('h1').click(); await page.keyboard.press('ArrowLeft');
  await expect(page.getByRole('heading', { name: 'The hidden representation.' })).toBeVisible();
  await page.getByRole('button', { name: 'Layer 1 · CONV1 · 8 feature maps' }).click();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('heading', { name: 'Look deeper.' })).toBeVisible();
});

test('failed intervention retains the current baseline and permits retry', async ({ page }) => {
  await loaded(page); await page.getByRole('button',{name:'Drawing · INPUT · Pixels'}).click(); const margin = await page.getByTestId('current-margin').textContent();
  await page.route('**/api/intervene', route => route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'Temporary inference failure' }) }));
  await page.getByRole('button', { name: 'Change background to yellow' }).click();
  await expect(page.getByRole('alert')).toContainText('Temporary inference failure');
  await expect(page.getByTestId('current-margin')).toHaveText(margin!);
  await page.unroute('**/api/intervene');
  await page.getByRole('button', { name: 'Change background to yellow' }).click();
  await expect(page.getByRole('button', { name: 'Reset & try again' })).toBeEnabled();
});

test('mobile layout fits and startup failures have a useful retry', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await loaded(page);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: '../results/arcade-ux/mobile-input.png', fullPage: true });
  await page.route('**/api/ready', route => route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'Models unavailable' }) }));
  await page.reload(); await expect(page.getByRole('alert')).toContainText('Models unavailable');
  await page.unroute('**/api/ready');
  await page.getByRole('button', { name: 'Retry connection' }).click();
  await expect(page.getByRole('button', { name: 'Start experiment' })).toBeVisible();
});
