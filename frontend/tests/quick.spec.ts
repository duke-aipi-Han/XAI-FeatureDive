import { expect, test } from './fixtures';
import type { Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { quickFeedback } from '../src/challenges/quick';
import type { QuickRecord } from '../src/challenges/quick';
import { FLIP_EXAMPLES } from '../src/examples';

async function loaded(page: Page) {
  await page.goto('/');
  await expect(page.getByRole('button',{name:'Start experiment'})).toBeVisible();
  await expect(page.getByTestId('current-margin')).toHaveText(/\d/);
}
const names={evidence:'Find the Evidence',hidden:'Inside the Hidden Layers',shortcut:'The Shortcut Trap'};
for(const id of ['evidence','hidden','shortcut'] as const) {
  test(`quick ${id}: one prediction, one test, immediate measured lesson; reflection optional`,async({page})=>{
    let interventions=0;page.on('request',r=>{if(r.url().endsWith('/api/intervene'))interventions++;});
    await loaded(page);
    await page.getByRole('button',{name:'Guided challenges',exact:true}).click();
    await page.getByRole('button',{name:`Open ${names[id]}`,exact:true}).click();
    const round=page.getByRole('region',{name:`Quick ${names[id]}`,exact:true});
    await expect(round).toHaveAttribute('aria-busy','false');
    await expect(round.getByRole('heading',{level:2})).toBeFocused();
    await expect(round.locator('textarea:visible')).toHaveCount(0);
    await expect(round.locator('select:visible')).toHaveCount(0);
    const label=id==='evidence'?'Spot A':id==='hidden'?'Feature A':'Model A';
    if(id==='shortcut') {
      await expect(round).not.toContainText('Model A had balanced');
      await expect(round).not.toContainText('Model A had shortcut');
    }
    await round.getByRole('button',{name:label,exact:true}).click();
    expect(interventions).toBe(0);
    await expect(round.getByTestId('quick-guess')).toContainText(label);
    await round.getByRole('button',{name:id==='evidence'?'Cover both spots':id==='hidden'?'Test both switches':'Swap the backgrounds',exact:true}).click();
    await expect(round.getByTestId('quick-outcome')).toBeVisible();
    expect(interventions).toBe(id==='shortcut'?4:2);
    await expect(page.getByLabel('1 of 3 challenges completed',{exact:true})).toBeVisible();
    await round.getByRole('heading',{level:2}).click();
    await page.screenshot({path:`../results/arcade-ux/quick-${id}.png`,fullPage:true});
    await round.getByText('See the evidence / save this round',{exact:true}).click();
    const download=page.waitForEvent('download');
    await round.getByRole('button',{name:'Download round record',exact:true}).click();
    const path=`../results/arcade-ux/quick-${id}-record.json`;await (await download).saveAs(path);
    const record=JSON.parse(await readFile(path,'utf8')) as QuickRecord;
    expect(record.reflection).toBe('');expect(record.hypothesis.choice).toBe('A');
    expect(Date.parse(record.hypothesis.recorded_at)).toBeLessThanOrEqual(Date.parse(record.requested_at));
    expect(Date.parse(record.requested_at)).toBeLessThanOrEqual(Date.parse(record.measured_at));
    expect(record.feedback).toEqual(quickFeedback(id,record.baselines,record.results,'A',record.model_mapping??{A:'balanced',B:'shortcut'}));
    for(const result of record.results) {
      const baseline=record.baselines.find(row=>row.request.model_id===result.request.model_id)!;
      expect(result.baseline.logits).toEqual(baseline.baseline.logits);
      expect(result.request.interventions).toHaveLength(1);
      if(id==='shortcut') expect(result.input.modified_alpha).toEqual(baseline.input.source_alpha);
    }
    await expect(round.getByTestId('quick-outcome')).toContainText(record.feedback.winner==='similar'?'Both':record.feedback.winner);
    await round.getByLabel('Optional round reflection').fill('Automated acceptance example: looking active is different from having an effect.');
    await round.getByRole('button',{name:'Play again',exact:true}).click();
    await expect(round.getByRole('button',{name:label,exact:true})).toBeVisible();
    await expect(round.getByTestId('quick-outcome')).toHaveCount(0);
  });
}

test('a failed paired quick test retains the guess and publishes no partial lesson; mobile fits',async({page})=>{
  await page.setViewportSize({width:390,height:844});await loaded(page);
  await page.getByRole('button',{name:'Guided challenges',exact:true}).click();
  await page.getByRole('button',{name:'Open Find the Evidence',exact:true}).click();
  const round=page.getByRole('region',{name:'Quick Find the Evidence',exact:true});
  await expect(round).toHaveAttribute('aria-busy','false');
  await round.getByRole('button',{name:'Spot B',exact:true}).focus();await page.keyboard.press('Enter');
  await page.route('**/api/intervene',route=>route.request().postDataJSON().interventions[0].region.rectangle.x===20
    ? route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({detail:'Temporary paired test failure'})}) : route.continue());
  await round.getByRole('button',{name:'Cover both spots'}).click();
  await expect(round.getByRole('alert')).toContainText('Temporary paired test failure');
  await expect(round.getByTestId('quick-outcome')).toHaveCount(0);
  await expect(page.getByLabel('0 of 3 challenges completed',{exact:true})).toBeVisible();
  await page.getByRole('button',{name:'Free exploration',exact:true}).click();
  await page.getByRole('button',{name:'Guided challenges',exact:true}).click();
  await expect(round.getByTestId('quick-guess')).toContainText('Spot B');
  await page.unroute('**/api/intervene');await round.getByRole('button',{name:'Cover both spots'}).click();
  await expect(round.getByTestId('quick-outcome')).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await round.getByRole('heading',{level:2}).click();
  await page.screenshot({path:'../results/arcade-ux/quick-mobile.png',fullPage:true});
  await round.getByRole('button',{name:'Next challenge →'}).click();
  await expect(page.getByRole('region',{name:'Quick Inside the Hidden Layers',exact:true})).toHaveAttribute('aria-busy','false');
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
});

test('free exploration loads all three real class-flipping ablation examples and resets exactly',async({page})=>{
  await loaded(page);
  await expect(page.getByLabel('Model',{exact:true})).not.toBeVisible();
  await expect(page.getByTestId('current-margin')).not.toBeVisible();
  await page.getByText('More class-flip examples',{exact:true}).click();
  for(const example of FLIP_EXAMPLES) {
    await page.getByRole('button',{name:`Load ${example.name} example`,exact:true}).click();
    await expect(page.getByTestId('current-prediction')).toHaveText(example.name);
    const button=page.getByRole('button',{name:`Switch off channel ${example.channel}`,exact:true});
    await expect(button).toBeEnabled();
    const response=page.waitForResponse(r=>r.url().endsWith('/api/intervene'));
    await button.click();const measured=await response;expect(measured.ok()).toBe(true);const result=await measured.json();
    expect(result.request.sample_id).toBe(example.request.sample_id);
    expect(result.request.background).toBe(example.request.background);
    expect(result.request.interventions).toEqual([{type:'channel_ablation',layer:example.layer,channel:example.channel}]);
    expect(result.comparison.classification_flip).toBe(true);
    await expect(page.getByTestId('current-prediction')).toHaveText(result.modified.predicted_class);
    await expect(page.getByTestId('outcome')).toContainText('You changed its mind!');
    if(example.name==='cat') {
      await page.getByRole('heading',{name:'Pull the plug.',exact:true}).click();
      await page.screenshot({path:'../results/arcade-ux/free-class-flip.png',fullPage:true});
    }
    await page.getByRole('button',{name:'Reset & try again'}).click();
    await expect(page.getByTestId('current-prediction')).toHaveText(result.baseline.predicted_class);
    await expect(page.getByTestId('current-margin')).toHaveText(result.baseline.target_logit_margin.toFixed(3));
  }
});
