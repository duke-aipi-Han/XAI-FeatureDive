import { test,expect } from './fixtures';
import type { Page } from '@playwright/test';
import type { HintResult } from '../src/types';
import { readFile } from 'node:fs/promises';

async function loaded(page: Page) {
  const response=page.waitForResponse(r=>r.url().endsWith('/api/hints'));
  await page.goto('/');
  const hints=await (await response).json() as HintResult;
  await expect(page.getByRole('region',{name:'Feature Radar hints'})).toBeVisible();
  return hints;
}

test('default Feature Radar ranks both layers, selects channels, and separates estimates from a real class flip',async({page})=>{
  let interventions=0,hintCalls=0;
  page.on('request',r=>{if(r.url().endsWith('/api/intervene'))interventions++;if(r.url().endsWith('/api/hints'))hintCalls++;});
  const hints=await loaded(page);
  expect(hints.experiment.request.sample_id).toBe('fish:20708');
  expect(hints.experiment.request.background).toBe('blue');
  expect(hints.experiment.request.interventions).toEqual([]);
  await expect(page.getByLabel('Feature Radar',{exact:true})).toHaveAttribute('aria-pressed','true');
  await expect(page.getByRole('button',{name:'Switch off channel 7',exact:true})).toBeEnabled();
  await expect(page.getByTestId('selected-channel-hint')).toContainText(hints.channels[0].estimated_margin_change.toFixed(3));
  const conv2=hints.channels.find(row=>row.layer==='conv2')!;
  if(hints.channels.indexOf(conv2)>=3)await page.getByText('All 24 channels',{exact:true}).click();
  await page.getByRole('button',{name:`Hint conv2 channel ${conv2.channel}`,exact:true}).click();
  await expect(page.getByRole('button',{name:'Layer 2 · CONV2 · 16 feature maps'})).toHaveAttribute('aria-current','step');
  await expect(page.getByRole('button',{name:`Select channel ${conv2.channel}`,exact:true})).toHaveAttribute('aria-pressed','true');
  await page.getByRole('button',{name:'Hint conv1 channel 7',exact:true}).click();
  expect(interventions).toBe(0);
  await expect(page.getByRole('button',{name:'Switch off channel 7',exact:true})).toBeEnabled();
  await page.getByRole('heading',{name:'Patterns, one layer in.',exact:true}).click();
  await page.screenshot({path:'../results/arcade-ux/feature-radar.png',fullPage:true});
  const response=page.waitForResponse(r=>r.url().endsWith('/api/intervene'));
  await page.getByRole('button',{name:'Switch off channel 7',exact:true}).click();
  const result=await (await response).json();
  expect(result.comparison.classification_flip).toBe(true);
  await expect(page.getByTestId('current-prediction')).toHaveText(result.modified.predicted_class);
  await expect(page.getByRole('region',{name:'Feature Radar hints'})).toContainText('Reset baseline');
  await expect(page.getByTestId('selected-channel-hint')).toContainText(hints.channels[0].estimated_margin_change.toFixed(3));
  expect(hintCalls).toBe(1);
  const event=page.waitForEvent('download');await page.getByRole('button',{name:'Export experiment JSON',exact:false}).click();
  const path='../results/arcade-ux/feature-radar-experiment.json';await(await event).saveAs(path);
  const exported=JSON.parse(await readFile(path,'utf8'));
  expect(exported.baseline_hints.reference).toBe('original_baseline');
  expect(exported.baseline_hints.experiment.request.interventions).toEqual([]);
  expect(exported.experiment.modified.logits).toEqual(result.modified.logits);
});

test('Pixel hunt defaults to Eraser Vision, uses its real suggested mask, and restores the clue on reset',async({page})=>{
  await loaded(page);
  const response=page.waitForResponse(r=>r.url().endsWith('/api/hints')&&r.request().postDataJSON().sample_id==='airplane:72939');
  await page.getByRole('button',{name:'Pixel hunt · airplane',exact:true}).click();
  const hints=await(await response).json() as HintResult;
  expect(hints.suggested_patch?.rectangle).toEqual({x:4,y:14,width:4,height:4});
  await expect(page.getByLabel('Eraser Vision',{exact:true})).toHaveAttribute('aria-pressed','true');
  await expect(page.getByTestId('erase-hint-estimate')).toContainText(hints.suggested_patch!.estimated_margin_change.toFixed(3));
  await expect(page.locator('.hint-drawing-patch')).toBeVisible();
  await page.getByRole('heading',{name:'Start with the pixels.',exact:true}).click();
  await page.screenshot({path:'../results/arcade-ux/eraser-vision.png',fullPage:true});
  const measured=page.waitForResponse(r=>r.url().endsWith('/api/intervene'));
  await page.getByRole('button',{name:'Erase hinted patch',exact:true}).click();
  const result=await(await measured).json();
  expect(result.request.interventions[0]).toMatchObject({type:'drawing_erasure',region:{rectangle:hints.suggested_patch!.rectangle}});
  expect(result.comparison.classification_flip).toBe(true);
  await expect(page.getByTestId('current-prediction')).toHaveText('fish');
  await expect(page.locator('.hint-drawing-patch')).toHaveCount(0);
  await expect(page.getByRole('button',{name:'Erase hinted patch',exact:true})).toBeDisabled();
  await expect(page.getByText('Eraser Vision overlay is paused on the changed drawing.',{exact:false})).toBeVisible();
  const event=page.waitForEvent('download');await page.getByRole('button',{name:'Export experiment JSON',exact:false}).click();
  await(await event).saveAs('../results/arcade-ux/eraser-vision-experiment.json');
  await page.getByRole('button',{name:'Reset & try again',exact:false}).click();
  await expect(page.getByTestId('current-prediction')).toHaveText('airplane');
  await expect(page.locator('.hint-drawing-patch')).toBeVisible();
  await expect(page.getByRole('button',{name:'Erase hinted patch',exact:true})).toBeEnabled();
  // Switching attribution tools makes their different meanings explicit.
  await page.getByText('More drawing tools',{exact:true}).click();
  await page.getByRole('button',{name:/Saliency Scope/}).click();
  await expect(page.getByLabel('Eraser Vision',{exact:true})).toHaveAttribute('aria-pressed','false');
  await page.getByLabel('Eraser Vision',{exact:true}).click();
  await expect(page.getByRole('button',{name:/Saliency Scope/})).toHaveAttribute('aria-pressed','false');
});

test('a failed power-up is retryable and target drafts do not silently change its reference',async({page})=>{
  await page.route('**/api/hints',route=>route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({detail:'Temporary hint failure'})}));
  await page.goto('/');
  await expect(page.getByRole('alert')).toContainText('Temporary hint failure');
  await expect(page.getByTestId('current-prediction')).toHaveText('fish');
  await page.unroute('**/api/hints');await page.getByRole('button',{name:'Retry power-up'}).click();
  await expect(page.getByRole('region',{name:'Feature Radar hints'})).toBeVisible();
  await page.getByText('More settings',{exact:true}).click();
  await page.getByLabel('Target class',{exact:true}).selectOption('cat');
  await expect(page.getByRole('region',{name:'Feature Radar hints'})).toContainText('support fish');
  const response=page.waitForResponse(r=>r.url().endsWith('/api/hints')&&r.request().postDataJSON().target_class==='cat');
  await page.getByRole('button',{name:'Start experiment'}).click();
  const hints=await(await response).json() as HintResult;
  expect(hints.experiment.request.target_class).toBe('cat');
  await expect(page.getByRole('region',{name:'Eraser Vision hints'})).toContainText('support cat');
});

test('power-ups fit on mobile and a hint can be selected with the keyboard',async({page})=>{
  await page.setViewportSize({width:390,height:844});await loaded(page);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.getByRole('button',{name:'Hint conv1 channel 7',exact:true}).focus();await page.keyboard.press('Enter');
  await expect(page.getByRole('button',{name:'Switch off channel 7',exact:true})).toBeEnabled();
  await page.getByRole('heading',{name:'Patterns, one layer in.',exact:true}).click();
  await page.screenshot({path:'../results/arcade-ux/powerups-mobile.png',fullPage:true});
  await page.getByRole('button',{name:'Pixel hunt · airplane',exact:true}).click();
  await expect(page.getByRole('button',{name:'Erase hinted patch',exact:true})).toBeEnabled();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
});

test('the stage leads; tools follow the view; reset is always ready and history is gone',async({page})=>{
  await loaded(page);
  const stage=await page.getByRole('region',{name:'Experiment stage',exact:true}).boundingBox();
  const dock=await page.getByRole('complementary',{name:'Experiment tools',exact:true}).boundingBox();
  const map=await page.getByRole('img',{name:'conv1 channel 7 current activation',exact:true}).boundingBox();
  expect(stage!.width).toBeGreaterThan(dock!.width*2);
  expect(map!.width).toBeGreaterThan(400);
  await expect(page.getByLabel('Eraser Vision',{exact:true})).toHaveCount(0);
  await expect(page.getByRole('button',{name:'Pixel Eraser',exact:true})).toHaveCount(0);
  await expect(page.getByRole('button',{name:/Undo/})).toHaveCount(0);
  await expect(page.getByText('Experiment history',{exact:true})).toHaveCount(0);
  await expect(page.getByRole('button',{name:'Reset & try again',exact:true})).toBeVisible();
  for(const name of ['Hidden · DENSE · 32 neurons','Answer · OUTPUT · 3 classes']){
    await page.getByRole('button',{name,exact:true}).click();
    await expect(page.getByLabel('Feature Radar',{exact:true})).toHaveCount(0);
    await expect(page.getByLabel('Eraser Vision',{exact:true})).toHaveCount(0);
    await expect(page.getByRole('button',{name:/Switch off channel/})).toHaveCount(0);
  }
  await page.getByRole('button',{name:'Pixel hunt · airplane',exact:false}).click();
  await expect(page.getByRole('button',{name:'Erase hinted patch',exact:true})).toBeEnabled();
  await expect(page.getByLabel('Feature Radar',{exact:true})).toHaveCount(0);
  await expect(page.getByRole('button',{name:'Pixel Eraser',exact:true})).toBeVisible();
  await page.getByRole('button',{name:'Baseline',exact:true}).click();
  await expect(page.getByRole('button',{name:'Erase hinted patch',exact:true})).toBeDisabled();
  await expect(page.getByRole('group',{name:/28 by 28 drawing/})).toHaveCount(0);
  await page.getByRole('button',{name:'Reset & try again',exact:true}).click();
  await expect(page.getByRole('button',{name:'Current',exact:true})).toHaveAttribute('aria-pressed','true');
  await expect(page.getByRole('button',{name:'Erase hinted patch',exact:true})).toBeEnabled();
});
