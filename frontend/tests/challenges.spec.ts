import { expect, test } from './fixtures';
import type { Locator, Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { effectCategory, sensitivity } from '../src/challenges/state';
import type { CompletionRecord } from '../src/challenges/state';
import type { ExperimentResult } from '../src/types';

async function open(page: Page, name: string) {
  await page.goto('/');
  await expect(page.getByTestId('current-margin')).toHaveText(/\d/);
  await page.keyboard.press('Tab');
  await expect(page.locator('.skip-link')).toBeFocused();
  await expect(page.locator('.skip-link')).toHaveCSS('clip-path','none');
  await page.getByRole('button', {name:'Guided challenges',exact:true}).click();
  await page.getByText('Teacher options',{exact:true}).click();
  await page.getByRole('button',{name:'Longer investigations',exact:true}).click();
  await page.getByRole('button', {name:`Open ${name}`,exact:true}).click();
  const lesson = page.getByRole('region',{name,exact:true});
  await expect(lesson).toHaveAttribute('aria-busy','false');
  return lesson;
}
async function predict(lesson: Locator) {
  await lesson.getByRole('button',{name:'Make a hypothesis →'}).click();
  await lesson.getByLabel('Predicted direction',{exact:true}).selectOption('decrease');
  await lesson.getByLabel('Predicted magnitude',{exact:true}).selectOption('large');
  await lesson.getByLabel('Hypothesis reason').fill('Automated acceptance example: this visible signal might strongly support the target.');
  await lesson.getByRole('button',{name:'Record hypothesis',exact:true}).click();
}
async function run(page: Page, lesson: Locator, button: string) {
  const response = page.waitForResponse(r=>r.url().endsWith('/api/intervene') && r.request().method()==='POST');
  await lesson.getByRole('button',{name:button,exact:true}).click();
  const result = await (await response).json() as ExperimentResult;
  await expect(lesson.getByTestId('challenge-margin')).toHaveText(result.modified.target_logit_margin.toFixed(4));
  const effect = effectCategory(result.comparison.target_logit_margin_change);
  await expect(lesson.getByTestId('measured-feedback')).toContainText(`Observed: ${effect.direction.replace('_',' ')} · ${effect.magnitude} effect`);
  return result;
}
async function finish(page: Page, lesson: Locator, id: string) {
  await lesson.getByLabel('Challenge reflection',{exact:true}).fill('Automated acceptance example: these are effects on one curated training drawing; compare the measured changes rather than generalizing to every image.');
  await lesson.getByRole('button',{name:'Complete challenge',exact:true}).click();
  const event = page.waitForEvent('download');
  await lesson.getByRole('button',{name:'Download challenge record',exact:true}).click();
  const download = await event;
  const file = `../results/arcade-ux/${id}-acceptance-record.json`;
  await download.saveAs(file);
  return JSON.parse(await readFile(file,'utf8')) as CompletionRecord;
}

test('Find the Evidence: saved hypotheses precede two real independent patch measurements', async ({page})=>{
  let interventions=0; page.on('request',r=>{if(r.url().endsWith('/api/intervene')) interventions++;});
  const lesson = await open(page,'Find the Evidence');
  await lesson.getByRole('button',{name:'Select patch 8,12',exact:true}).click();
  await predict(lesson);
  expect(interventions).toBe(0);
  await expect(lesson.getByLabel('Challenge drawing',{exact:true})).toBeDisabled();
  const first = await run(page,lesson,'Run controlled occlusion');
  await expect(lesson.getByRole('button',{name:'Reflect on the evidence →'})).toBeDisabled();
  await lesson.getByRole('button',{name:'Compare another patch',exact:true}).click();
  await expect(lesson.getByRole('button',{name:'Select patch 8,12',exact:true})).toBeDisabled();
  await lesson.getByRole('button',{name:'Select patch 20,12',exact:true}).click();
  await predict(lesson);
  const second = await run(page,lesson,'Run controlled occlusion');
  expect(second.baseline.logits).toEqual(first.baseline.logits);
  expect(second.request.interventions).toHaveLength(1);
  await lesson.getByRole('heading',{level:2}).click();
  await expect(page.locator('.skip-link')).toHaveCSS('top','-100px');
  await page.screenshot({path:'../results/arcade-ux/evidence-comparison.png',fullPage:true});
  await lesson.getByRole('button',{name:'Reflect on the evidence →'}).click();
  const record = await finish(page,lesson,'evidence');
  expect(record.trials).toHaveLength(2);
  for (const trial of record.trials!) {
    expect(Date.parse(trial.hypothesis.recorded_at)).toBeLessThanOrEqual(Date.parse(trial.requested_at!));
    expect(Date.parse(trial.requested_at!)).toBeLessThanOrEqual(Date.parse(trial.measured_at!));
    expect(trial.feedback?.observed).toEqual(effectCategory(trial.result!.comparison.target_logit_margin_change));
    expect(trial.hypothesis.direction).toBe('decrease');
  }
  await expect(page.getByLabel('1 of 3 challenges completed',{exact:true})).toBeVisible();
});

test('Hidden Layers: inspect both layers; compare two channels with honest measured feedback',async({page})=>{
  const lesson = await open(page,'Inside the Hidden Layers');
  await lesson.getByLabel('Challenge layer').selectOption('conv1');
  await expect(lesson.getByRole('button',{name:/^Choose channel /})).toHaveCount(8);
  await lesson.getByLabel('Challenge layer').selectOption('conv2');
  await expect(lesson.getByRole('button',{name:/^Choose channel /})).toHaveCount(16);
  await lesson.getByRole('button',{name:'Choose channel 0',exact:true}).click();
  await predict(lesson);
  const first = await run(page,lesson,'Run channel ablation');
  await lesson.getByRole('button',{name:'Compare another channel',exact:true}).click();
  await expect(lesson.getByRole('button',{name:'Choose channel 0',exact:true})).toBeDisabled();
  await lesson.getByRole('button',{name:'Choose channel 1',exact:true}).click();
  await predict(lesson);
  const second = await run(page,lesson,'Run channel ablation');
  expect(first.request.interventions).toEqual([{type:'channel_ablation',layer:'conv2',channel:0}]);
  expect(second.baseline.logits).toEqual(first.baseline.logits);
  await lesson.getByRole('heading',{level:2}).click();
  await expect(page.locator('.skip-link')).toHaveCSS('top','-100px');
  await page.screenshot({path:'../results/arcade-ux/hidden-comparison.png',fullPage:true});
  // An already silent channel is a genuine null intervention, not a fallback.
  await lesson.getByRole('button',{name:'Compare another channel',exact:true}).click();
  await expect(lesson.getByRole('button',{name:'Choose channel 5',exact:true})).toContainText('Mean 0.0000');
  await lesson.getByRole('button',{name:'Choose channel 5',exact:true}).click();
  await predict(lesson);
  const silent = await run(page,lesson,'Run channel ablation');
  expect(silent.modified.logits).toEqual(silent.baseline.logits);
  await lesson.getByRole('button',{name:'Reflect on the evidence →'}).click();
  const record = await finish(page,lesson,'hidden');
  expect(record.trials![2].feedback?.observed).toEqual({direction:'little_change',magnitude:'small'});
  expect(record.trials![0].feedback!.directionMatches).toBe(effectCategory(first.comparison.target_logit_margin_change).direction==='decrease');
  expect(record.trials![0].hypothesis.magnitude).toBe('large');
});

test('Shortcut Trap: A/B stays concealed until identification; swaps and ranges use matched real outputs',async({page})=>{
  let interventions=0; page.on('request',r=>{if(r.url().endsWith('/api/intervene')) interventions++;});
  const lesson = await open(page,'The Shortcut Trap');
  await expect(lesson.getByRole('region',{name:'Training regimes revealed'})).toHaveCount(0);
  await expect(lesson).not.toContainText('Model A: balanced');
  await expect(lesson).not.toContainText('Model A: shortcut');
  await lesson.getByRole('button',{name:'Predict background reliance →'}).click();
  await lesson.getByLabel('Background reliance prediction').selectOption('similar');
  await lesson.getByLabel('Background hypothesis reason').fill('Automated acceptance example: the initial predictions look alike, so I predict similar sensitivity.');
  await lesson.getByRole('button',{name:'Record background hypothesis'}).click();
  expect(interventions).toBe(0);
  await expect(lesson.getByRole('button',{name:'Evaluate background evidence →'})).toBeDisabled();
  for(const color of ['green','blue']) {
    await lesson.getByRole('button',{name:`Test ${color} background`,exact:true}).click();
    await expect(lesson.getByRole('button',{name:`Test ${color} background`,exact:true})).toContainText('✓ measured');
    await expect(lesson).toHaveAttribute('aria-busy','false');
  }
  expect(interventions).toBe(4);
  await lesson.getByRole('button',{name:'Evaluate background evidence →'}).click();
  await lesson.getByLabel('Identify shortcut-trained model').selectOption('unclear');
  await expect(lesson.getByRole('button',{name:'Record identification & reveal'})).toBeDisabled();
  await lesson.getByLabel('Identification evidence').fill('Automated acceptance example: the measured probability ranges differ; one drawing alone is insufficient for a general conclusion.');
  await expect(lesson.getByRole('region',{name:'Training regimes revealed'})).toHaveCount(0);
  await lesson.getByRole('button',{name:'Record identification & reveal'}).click();
  await expect(lesson.getByRole('region',{name:'Training regimes revealed'})).toBeVisible();
  await lesson.getByRole('heading',{level:2}).click();
  await expect(page.locator('.skip-link')).toHaveCSS('top','-100px');
  await page.screenshot({path:'../results/arcade-ux/shortcut-reveal.png',fullPage:true});
  await lesson.getByLabel('Shortcut reflection').fill('Automated acceptance example: changing only the background altered the measured responses differently. This single training drawing does not measure generalization or student learning.');
  await lesson.getByRole('button',{name:'Complete challenge',exact:true}).click();
  const event = page.waitForEvent('download');
  await lesson.getByRole('button',{name:'Download challenge record',exact:true}).click();
  const download = await event; const path='../results/arcade-ux/shortcut-acceptance-record.json'; await download.saveAs(path);
  const record = JSON.parse(await readFile(path,'utf8')) as CompletionRecord;
  expect(record.observations).toHaveLength(3);
  expect(record.hypothesis?.model).toBe('similar');
  expect(record.identification?.model).toBe('unclear');
  expect(record.background_feedback?.identification_matches_regime).toBeNull();
  for (const row of record.observations!) {
    expect(row.results.A.request.model_id).toBe(record.model_mapping!.A);
    expect(row.results.B.request.model_id).toBe(record.model_mapping!.B);
    expect(row.results.A.input.source_alpha).toEqual(row.results.B.input.source_alpha);
    for (const id of ['A','B'] as const) {
      const result=row.results[id];
      expect(result.input.modified_alpha).toEqual(result.input.source_alpha);
      expect(result.baseline.logits).toEqual(record.observations![0].results[id].baseline.logits);
      if(row.kind==='swap') expect(Date.parse(record.hypothesis!.recorded_at)).toBeLessThanOrEqual(Date.parse(row.requested_at));
    }
  }
  expect(record.background_feedback!.A).toEqual(sensitivity(record.observations!,'A',2));
  expect(record.background_feedback!.B).toEqual(sensitivity(record.observations!,'B',2));
  await lesson.getByRole('button',{name:'Restart with new A/B labels'}).click();
  await expect(lesson.getByRole('button',{name:'Predict background reliance →'})).toBeVisible();
  await expect(lesson.getByRole('region',{name:'Training regimes revealed'})).toHaveCount(0);
});

test('failure retry and mode switching preserve hypotheses; mobile selection supports keyboard',async({page})=>{
  await page.setViewportSize({width:390,height:844});
  const lesson = await open(page,'Find the Evidence');
  const patch=lesson.getByRole('button',{name:'Select patch 8,12',exact:true});
  await patch.focus(); await page.keyboard.press('Enter');
  await predict(lesson);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.route('**/api/intervene',route=>route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({detail:'Temporary challenge inference failure'})}));
  await lesson.getByRole('button',{name:'Run controlled occlusion'}).click();
  await expect(lesson.getByRole('alert')).toContainText('Temporary challenge inference failure');
  await expect(lesson.getByTestId('recorded-hypothesis')).toContainText('decrease');
  await expect(lesson.getByTestId('challenge-margin')).toHaveCount(0);
  await page.getByRole('button',{name:'Free exploration',exact:true}).click();
  await page.getByRole('button',{name:'Guided challenges',exact:true}).click();
  await expect(lesson.getByTestId('recorded-hypothesis')).toContainText('decrease');
  await page.unroute('**/api/intervene');
  await run(page,lesson,'Run controlled occlusion');
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await lesson.getByRole('heading',{level:2}).click();
  await expect(page.locator('.skip-link')).toHaveCSS('top','-100px');
  await page.screenshot({path:'../results/arcade-ux/mobile-challenge.png',fullPage:true});
  await page.getByRole('button',{name:'Open Inside the Hidden Layers'}).click();
  await expect(page.getByRole('region',{name:'Inside the Hidden Layers',exact:true})).toHaveAttribute('aria-busy','false');
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.getByRole('button',{name:'Open The Shortcut Trap'}).click();
  await expect(page.getByRole('button',{name:'Predict background reliance →'})).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
});
