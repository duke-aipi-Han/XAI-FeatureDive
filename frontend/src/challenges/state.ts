import type { Background, ExperimentResult, ModelID } from '../types';
export type ChallengeID = 'evidence' | 'hidden' | 'shortcut';
export type Step = 'observe' | 'hypothesize' | 'intervene' | 'evaluate' | 'reflect' | 'complete';
export type Direction = 'decrease' | 'little_change' | 'increase';
export type Magnitude = 'small' | 'moderate' | 'large';
export type AnonymousID = 'A' | 'B';
export type Hypothesis = Readonly<{ direction: Direction; magnitude: Magnitude; reason: string; recorded_at: string }>;
export type SensitivityHypothesis = Readonly<{ model: AnonymousID | 'similar'; reason: string; recorded_at: string }>;
export interface Trial {
  selection: { kind: 'patch'; x: number; y: number; saliency_sum: number } | { kind: 'channel'; layer: 'conv1' | 'conv2'; channel: number; mean_activation: number };
  hypothesis: Hypothesis;
  requested_at?: string;
  measured_at?: string;
  result?: ExperimentResult;
  feedback?: { observed: { direction: Direction; magnitude: Magnitude }; directionMatches: boolean; magnitudeMatches: boolean };
}
export interface BackgroundObservation {
  background: Background; kind: 'baseline' | 'swap'; requested_at: string; measured_at: string;
  results: Record<AnonymousID, ExperimentResult>;
}
export interface CompletionRecord {
  schema_version: 1; challenge_id: ChallengeID; source_role: 'illustrative_training_example';
  started_at: string; completed_at: string; reflection: string;
  trials?: Trial[]; hypothesis?: SensitivityHypothesis;
  observations?: BackgroundObservation[]; identification?: { model: AnonymousID | 'unclear'; evidence: string; recorded_at: string };
  background_feedback?: { A: ReturnType<typeof sensitivity>; B: ReturnType<typeof sensitivity>; observed_more_sensitive: AnonymousID | 'similar'; hypothesis_matches: boolean; identification_matches_regime: boolean | null };
  model_mapping?: Record<AnonymousID, ModelID>;
}
export const CHALLENGES = [
  { id: 'evidence' as const, number: '01', title: 'Find the Evidence', question: 'Does the brightest region matter most?', description: 'Compare a saliency map with real pixel interventions.', sample: 'fish:2617' },
  { id: 'hidden' as const, number: '02', title: 'Inside the Hidden Layers', question: 'Does strong activity mean strong influence?', description: 'Choose channels, predict their influence, then zero them.', sample: 'airplane:142997' },
  { id: 'shortcut' as const, number: '03', title: 'The Shortcut Trap', question: 'Which model depends more on background?', description: 'Investigate two anonymous models using the same drawing.', sample: 'airplane:72939' },
];
export const STEPS: Step[] = ['observe', 'hypothesize', 'intervene', 'evaluate', 'reflect'];
export function effectCategory(delta: number): { direction: Direction; magnitude: Magnitude } {
  if (!Number.isFinite(delta)) throw new Error('A finite measured margin change is required.');
  const size = Math.abs(delta);
  return { direction: size <= .1 ? 'little_change' : delta > 0 ? 'increase' : 'decrease', magnitude: size < .5 ? 'small' : size < 2 ? 'moderate' : 'large' };
}
export function feedback(trial: Trial) {
  if (!trial.result) throw new Error('Feedback requires a measured intervention.');
  const observed = effectCategory(trial.result.comparison.target_logit_margin_change);
  return { observed, directionMatches: observed.direction === trial.hypothesis.direction, magnitudeMatches: observed.magnitude === trial.hypothesis.magnitude };
}
export function distinctSelections(trials: Trial[]): number {
  return new Set(trials.filter(trial => trial.result).map(trial => trial.selection.kind === 'patch' ? `patch:${trial.selection.x}:${trial.selection.y}` : `${trial.selection.layer}:${trial.selection.channel}`)).size;
}
export function anonymousMapping(reverse: boolean): Record<AnonymousID, ModelID> {
  return reverse ? { A: 'shortcut', B: 'balanced' } : { A: 'balanced', B: 'shortcut' };
}
export function sensitivity(observations: BackgroundObservation[], model: AnonymousID, targetIndex: number) {
  const trainingColors = ['green', 'blue', 'yellow'];
  const measured = observations.filter(observation => trainingColors.includes(observation.background));
  if (new Set(measured.map(observation => observation.background)).size !== 3) throw new Error('All three training colors are required.');
  const probabilities = measured.map(row => row.results[model].modified.probabilities[targetIndex]);
  const margins = measured.map(row => row.results[model].modified.target_logit_margin);
  return { probabilityRangePoints: 100 * (Math.max(...probabilities) - Math.min(...probabilities)),
    marginRange: Math.max(...margins) - Math.min(...margins),
    anyLabelFlip: new Set(measured.map(row => row.results[model].modified.predicted_class)).size > 1 };
}
export function sensitivityWinner(a: number, b: number): AnonymousID | 'similar' {
  // Explicit teaching tolerance; this is not a population-significance test.
  return Math.abs(a - b) <= 1 ? 'similar' : a > b ? 'A' : 'B';
}
export function downloadRecord(record: CompletionRecord) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(record, null, 2)], { type: 'application/json' }));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = `featuredive-challenge-${record.challenge_id}.json`; anchor.click(); URL.revokeObjectURL(url);
}
