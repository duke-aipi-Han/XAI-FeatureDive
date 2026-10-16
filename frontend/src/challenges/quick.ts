import type { ExperimentResult, ModelID } from '../types';
import { CLASSES } from '../types';
import type { AnonymousID, ChallengeID } from './state';
import { sensitivityWinner } from './state';
export type Guess = AnonymousID | 'similar';
export interface QuickRecord {
  schema_version: 2; mode: 'quick_round'; challenge_id: ChallengeID;
  source_role: 'illustrative_training_example';
  hypothesis: Readonly<{ choice: Guess; recorded_at: string }>;
  requested_at: string; measured_at: string;
  baselines: ExperimentResult[]; results: ExperimentResult[];
  feedback: { A: number; B: number; winner: Guess; metric: 'absolute_probability_point_change' | 'three_color_probability_range'; matches: boolean };
  model_mapping?: Record<AnonymousID, ModelID>;
  reflection: string;
}
export function quickFeedback(id: ChallengeID, baselines: ExperimentResult[], results: ExperimentResult[], guess: Guess, mapping: Record<AnonymousID, ModelID>) {
  if (!baselines.length || !results.length) throw new Error('Measurements are required before feedback.');
  let A: number, B: number;
  if (id === 'shortcut') {
    const range = (model: ModelID) => {
      const rows = [...baselines, ...results].filter(row => row.request.model_id === model && ['green','blue','yellow'].includes(row.input.modified_background));
      if (new Set(rows.map(row=>row.input.modified_background)).size !== 3) throw new Error('Measure all three colors for both models.');
      const values = rows.map(row => row.modified.probabilities[CLASSES.indexOf(row.request.target_class)] * 100);
      return Math.max(...values) - Math.min(...values);
    };
    A = range(mapping.A); B = range(mapping.B);
  } else {
    if (results.length !== 2) throw new Error('Both independent interventions are required.');
    A = Math.abs(results[0].comparison.target_probability_point_change);
    B = Math.abs(results[1].comparison.target_probability_point_change);
  }
  const winner = sensitivityWinner(A,B);
  return { A, B, winner, metric: id === 'shortcut' ? 'three_color_probability_range' as const : 'absolute_probability_point_change' as const, matches: guess === winner };
}
export function verifyQuickResult(result: ExperimentResult, baseline: ExperimentResult) {
  if (result.checkpoint_sha256 !== baseline.checkpoint_sha256 || result.source_bitmap_sha256 !== baseline.source_bitmap_sha256 || JSON.stringify(result.baseline.logits) !== JSON.stringify(baseline.baseline.logits)) throw new Error('The starting example changed. Restart this round.');
  if (result.request.interventions[0]?.type === 'background_swap' && JSON.stringify(result.input.modified_alpha) !== JSON.stringify(baseline.input.source_alpha)) throw new Error('The drawing changed unexpectedly. Restart this round.');
}
