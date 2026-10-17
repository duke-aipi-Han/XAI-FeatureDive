import type { ExperimentRequest, HistoryEntry, Intervention, Region } from './types';

export function withIntervention(request: ExperimentRequest, operation: Intervention): ExperimentRequest {
  // One active channel intervention; choosing another replaces it in this state.
  const previous = operation.type === 'channel_ablation'
    ? request.interventions.filter(item => item.type !== 'channel_ablation')
    : request.interventions;
  const interventions = [...previous, operation];
  if (interventions.length > 16) throw new Error('This experiment has reached 16 interventions. Reset to continue.');
  return { ...request, interventions };
}
export function branchHistory(history: HistoryEntry[], cursor: number, entry: HistoryEntry): HistoryEntry[] {
  return [...history.slice(0, cursor + 1), entry];
}
export function pointerPixel(clientX: number, clientY: number, rect: Pick<DOMRect, 'left' | 'top' | 'width' | 'height'>) {
  return {
    x: Math.min(27, Math.max(0, Math.floor((clientX - rect.left) / rect.width * 28))),
    y: Math.min(27, Math.max(0, Math.floor((clientY - rect.top) / rect.height * 28))),
  };
}
export function selectionRegion(start: { x: number; y: number }, end: { x: number; y: number }, mode: 'pixel' | 'patch' | 'rectangle'): Region {
  if (mode === 'pixel') return { pixels: [end] };
  if (mode === 'patch') return { rectangle: { x: Math.min(24, end.x), y: Math.min(24, end.y), width: 4, height: 4 } };
  return { rectangle: { x: Math.min(start.x, end.x), y: Math.min(start.y, end.y), width: Math.abs(start.x - end.x) + 1, height: Math.abs(start.y - end.y) + 1 } };
}
export function operationLabel(operation: Intervention): string {
  if (operation.type === 'channel_ablation') return `Zero ${operation.layer.toUpperCase()} · channel ${operation.channel}`;
  if (operation.type === 'background_swap') return `Background → ${operation.background}`;
  return operation.type === 'drawing_erasure' ? 'Erase drawing pixels' : 'Occlude region · fixed gray';
}
