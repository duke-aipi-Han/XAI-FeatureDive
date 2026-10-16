import { expect, it } from 'vitest';
import { quickFeedback, verifyQuickResult } from './quick';
import { anonymousMapping } from './state';
import type { Background, ExperimentResult, ModelID } from '../types';
// Synthetic arithmetic fixtures only; application predictions are never mocked.
function row(model: ModelID,color: Background,probability: number,delta = 0): ExperimentResult {
  return {request:{model_id:model,target_class:'fish',interventions:[]},
    comparison:{target_probability_point_change:delta},checkpoint_sha256:model,source_bitmap_sha256:'same-drawing',
    baseline:{logits:[1,2,3]},modified:{probabilities:[.1,probability,.9-probability]},
    input:{modified_background:color,source_alpha:[[1]],modified_alpha:[[1]]}} as unknown as ExperimentResult;
}
it('quick feedback requires both measurements and compares size regardless of direction',()=>{
  expect(()=>quickFeedback('hidden',[row('balanced','blue',.5)],[],'A',anonymousMapping(false))).toThrow();
  const results=[row('balanced','blue',.5,2),row('balanced','blue',.5,-9)];
  expect(quickFeedback('hidden',[results[0]],results,'B',anonymousMapping(false))).toEqual({A:2,B:9,winner:'B',metric:'absolute_probability_point_change',matches:true});
});
it('null and near-equal results remain similar instead of forcing the intended lesson',()=>{
  const zero=row('balanced','blue',.5);
  expect(quickFeedback('evidence',[zero],[zero,zero],'similar',anonymousMapping(false)).matches).toBe(true);
});
it('background ranges follow randomized aliases and include all training colors',()=>{
  const bases=[row('balanced','yellow',.7),row('shortcut','yellow',.7)];
  const results=[row('balanced','green',.7),row('balanced','blue',.7),row('shortcut','green',.1),row('shortcut','blue',.4),row('shortcut','gray',.99)];
  const feedback=quickFeedback('shortcut',bases,results,'A',anonymousMapping(true));
  expect(feedback.winner).toBe('A');expect(feedback.A).toBeCloseTo(60);expect(feedback.B).toBe(0);
  expect(()=>quickFeedback('shortcut',bases,results.slice(0,3),'A',anonymousMapping(true))).toThrow(/three/);
});
it('stale baselines and changed foregrounds cannot produce a round result',()=>{
  const base=row('balanced','blue',.5);
  expect(()=>verifyQuickResult({...base,checkpoint_sha256:'different'},base)).toThrow(/starting/);
  const swapped={...base,request:{...base.request,interventions:[{type:'background_swap' as const,background:'green' as const}]},input:{...base.input,modified_alpha:[[0]]}};
  expect(()=>verifyQuickResult(swapped,base)).toThrow(/drawing/);
});
