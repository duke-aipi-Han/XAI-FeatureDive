import { test as base, expect } from '@playwright/test';
// Pace real browser requests, leaving headroom below the API's 120/minute limit.
// One worker is intentional: all local Vite requests share a socket peer at the API.
interface Budget { times: number[]; queue: Promise<void> }
export const test = base.extend<{}, { budget: Budget }>({
  budget: [async ({},use,workerInfo)=>{
    // A failed worker loses its in-memory budget, while the API retains its window.
    if(workerInfo.workerIndex>0)await new Promise(resolve=>setTimeout(resolve,60000));
    await use({times:[],queue:Promise.resolve()});
  },{scope:'worker',timeout:70000}],
  page: async ({page,budget},use)=>{
    await budget.queue;
    budget.times=budget.times.filter(time=>time>Date.now()-60000);
    // Reserve room for a whole round before assertions start, rather than pausing
    // a page halfway through startup. The longest current test uses <20 calls.
    if(budget.times.length>=80) {
      await new Promise(resolve=>setTimeout(resolve,Math.max(0,60000-(Date.now()-budget.times[0])+50)));
      budget.times=budget.times.filter(time=>time>Date.now()-60000);
    }
    await page.route('**/api/**',async route=>{
      if(!['/api/ready','/api/health'].includes(new URL(route.request().url()).pathname)) {
        const slot=budget.queue.then(async()=>{
          budget.times=budget.times.filter(time=>time>Date.now()-60000);
          if(budget.times.length>=100) {
            const delay=Math.max(0,60000-(Date.now()-budget.times[0])+50);
            await new Promise(resolve=>setTimeout(resolve,delay));
            budget.times=budget.times.filter(time=>time>Date.now()-60000);
          }
          budget.times.push(Date.now());
        });
        budget.queue=slot.catch(()=>{});await slot;
      }
      await route.continue();
    });
    await use(page);
  },
});
export { expect };
