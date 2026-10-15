import { describe, expect, it } from 'vitest';
import { anonymousMapping, distinctSelections, effectCategory, feedback, sensitivity, sensitivityWinner } from './state';
import type { BackgroundObservation, Trial } from './state';
import type { Background, ExperimentResult } from '../types';

// Minimal synthetic data tests arithmetic only; browser tests use real inference.
const result = (delta: number, probability = .4, label = 'fish'): ExperimentResult => ({
  comparison: { target_logit_margin_change: delta },
  modified: { probabilities: [.1, probability, .9-probability], target_logit_margin: delta, predicted_class: label },
} as ExperimentResult);
const trial = (delta?: number): Trial => ({ selection: { kind:'patch',x:0,y:4,saliency_sum:12 },
  hypothesis: { direction:'decrease',magnitude:'large',reason:'A bright region may matter.',recorded_at:'2026-01-01T00:00:00Z' },
  ...(delta === undefined ? {} : {result:result(delta)}) });

describe('honest measurement feedback', () => {
  it('classifies zero, negative, positive and boundary effects without assuming a decrease', () => {
    expect(effectCategory(0)).toEqual({direction:'little_change',magnitude:'small'});
    expect(effectCategory(-.1).direction).toBe('little_change');
    expect(effectCategory(.10001).direction).toBe('increase');
    expect(effectCategory(-.5)).toEqual({direction:'decrease',magnitude:'moderate'});
    expect(effectCategory(2)).toEqual({direction:'increase',magnitude:'large'});
    expect(()=>effectCategory(NaN)).toThrow();
  });
  it('requires an actual result and keeps the original prediction on disagreement', () => {
    expect(()=>feedback(trial())).toThrow(/measured/);
    const pending = trial(.1341);
    expect(feedback(pending)).toEqual({observed:{direction:'increase',magnitude:'small'},directionMatches:false,magnitudeMatches:false});
    expect(pending.hypothesis.direction).toBe('decrease');
    expect(feedback(trial(-2.15)).directionMatches).toBe(true);
  });
  it('counts distinct measured selections, excluding pending and duplicate comparisons', () => {
    const one = trial(1);
    expect(distinctSelections([one,one,trial()])).toBe(1);
    const channel = {...one,selection:{kind:'channel' as const,layer:'conv1' as const,channel:1,mean_activation:2}};
    expect(distinctSelections([one,channel,{...channel,selection:{...channel.selection,layer:'conv2'}}])).toBe(3);
  });
});

describe('anonymous matched background comparison', () => {
  const row = (background: Background,a: number,b: number,label = 'fish'): BackgroundObservation => ({
    background,kind:'swap',requested_at:'before',measured_at:'after',results:{A:result(a,a,label),B:result(b,b)},
  });
  it('supports either alias assignment', () => {
    expect(anonymousMapping(false)).toEqual({A:'balanced',B:'shortcut'});
    expect(anonymousMapping(true)).toEqual({A:'shortcut',B:'balanced'});
  });
  it('requires all three distinct training colors', () => {
    expect(()=>sensitivity([row('green',.2,.3),row('green',.3,.4),row('blue',.4,.5)],'A',1)).toThrow(/three/);
  });
  it('computes ranges and flips from measurements and excludes optional gray', () => {
    const rows = [row('green',.2,.3),row('blue',.4,.31,'cat'),row('yellow',.8,.32),row('gray',1,0)];
    expect(sensitivity(rows,'A',1).probabilityRangePoints).toBeCloseTo(60);
    expect(sensitivity(rows,'A',1).marginRange).toBeCloseTo(.6);
    expect(sensitivity(rows,'A',1).anyLabelFlip).toBe(true);
    expect(sensitivity(rows,'B',1).anyLabelFlip).toBe(false);
  });
  it('allows either model to be more sensitive and retains similar/null results', () => {
    expect(sensitivityWinner(40,5)).toBe('A');
    expect(sensitivityWinner(5,40)).toBe('B');
    expect(sensitivityWinner(5,6)).toBe('similar');
    expect(sensitivityWinner(0,0)).toBe('similar');
  });
});
