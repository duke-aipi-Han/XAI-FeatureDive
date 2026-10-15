import { describe, expect, it } from 'vitest';
import { branchHistory, pointerPixel, selectionRegion, withIntervention } from './state';
import type { ExperimentRequest, HistoryEntry } from './types';
const baseline: ExperimentRequest = { model_id: 'balanced', sample_id: 'fish:2617', background: 'blue', target_class: 'fish', interventions: [] };

describe('replayable UI state', () => {
  it('preserves all baseline fields through input and channel interventions', () => {
    const modified = withIntervention(withIntervention(baseline, { type: 'background_swap', background: 'green' }), { type: 'channel_ablation', layer: 'conv2', channel: 1 });
    expect({ ...modified, interventions: [] }).toEqual(baseline);
    expect(baseline.interventions).toEqual([]);
  });
  it('replaces the active channel ablation without losing image interventions', () => {
    const first = withIntervention(withIntervention(baseline, { type: 'channel_ablation', layer: 'conv1', channel: 2 }), { type: 'background_swap', background: 'yellow' });
    const next = withIntervention(first, { type: 'channel_ablation', layer: 'conv2', channel: 3 });
    expect(next.interventions).toEqual([{ type: 'background_swap', background: 'yellow' }, { type: 'channel_ablation', layer: 'conv2', channel: 3 }]);
  });
  it('enforces the server intervention budget before sending a request', () => {
    const full = { ...baseline, interventions: Array.from({ length: 16 }, () => ({ type: 'background_swap' as const, background: 'gray' as const })) };
    expect(() => withIntervention(full, { type: 'background_swap', background: 'green' })).toThrow('16 interventions');
  });
  it('branches history after undo and preserves earlier snapshots', () => {
    const first = { label: 'baseline' } as HistoryEntry;
    const second = { label: 'color' } as HistoryEntry;
    const third = { label: 'occlusion' } as HistoryEntry;
    expect(branchHistory([first, second], 0, third)).toEqual([first, third]);
    expect(second.label).toBe('color');
  });
});

describe('canvas selections match source pixels', () => {
  it('maps scaled pixels and clamps edges', () => {
    const rect = { left: 10, top: 20, width: 280, height: 280 };
    expect(pointerPixel(95, 145, rect)).toEqual({ x: 8, y: 12 });
    expect(pointerPixel(290, 300, rect)).toEqual({ x: 27, y: 27 });
  });
  it('keeps a full fixed patch in bounds at the lower-right edge', () => {
    expect(selectionRegion({ x: 27, y: 27 }, { x: 27, y: 27 }, 'patch')).toEqual({ rectangle: { x: 24, y: 24, width: 4, height: 4 } });
  });
  it('creates inclusive rectangles in either drag direction', () => {
    expect(selectionRegion({ x: 8, y: 12 }, { x: 5, y: 10 }, 'rectangle')).toEqual({ rectangle: { x: 5, y: 10, width: 4, height: 3 } });
  });
});
