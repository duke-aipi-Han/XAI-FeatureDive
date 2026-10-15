import type { ExperimentResult, Prediction } from '../types';
import { CLASSES } from '../types';
import { signed } from '../components/Results';
import { STEPS } from './state';
import type { Step } from './state';

export function Progress({ step }: { step: Step }) {
  return <ol className="challenge-steps" aria-label="Investigation steps">{STEPS.map((name, index) => <li key={name} aria-current={name === step ? 'step' : undefined} className={step === 'complete' || STEPS.indexOf(step) > index ? 'done' : ''}><span>{String(index + 1).padStart(2, '0')}</span>{name}</li>)}</ol>;
}
export function Baseline({ prediction, sample }: { prediction: Prediction; sample: string }) {
  const target = CLASSES.indexOf(prediction.target_class);
  return <div className="challenge-baseline"><span className="eyebrow">STABLE BASELINE · {sample}</span><div><strong>{prediction.predicted_class}</strong><span>Target {prediction.target_class}: <b>{(100 * prediction.probabilities[target]).toFixed(2)}%</b><br />Target margin: <b>{prediction.target_logit_margin.toFixed(4)}</b></span></div><p>Uncalibrated probability. The target margin compares its score with the strongest competing class.</p></div>;
}
export function MeasuredComparison({ result, label = 'Measured effect' }: { result: ExperimentResult; label?: string }) {
  const target = CLASSES.indexOf(result.request.target_class);
  return <section className="challenge-measurement" aria-label={label}><h3>{label}</h3><table><thead><tr><th>Measure</th><th>Baseline</th><th>After intervention</th><th>Change</th></tr></thead><tbody>
    <tr><th>Prediction</th><td>{result.baseline.predicted_class}</td><td>{result.modified.predicted_class}</td><td>{result.comparison.classification_flip ? 'Changed' : 'Unchanged'}</td></tr>
    <tr><th>Target probability</th><td>{(100 * result.baseline.probabilities[target]).toFixed(2)}%</td><td>{(100 * result.modified.probabilities[target]).toFixed(2)}%</td><td>{signed(result.comparison.target_probability_point_change, 2)} points</td></tr>
    <tr><th>Target logit</th><td>{result.baseline.target_logit.toFixed(4)}</td><td>{result.modified.target_logit.toFixed(4)}</td><td>{signed(result.comparison.target_logit_change, 4)}</td></tr>
    <tr><th>Target margin</th><td>{result.baseline.target_logit_margin.toFixed(4)}</td><td data-testid="challenge-margin">{result.modified.target_logit_margin.toFixed(4)}</td><td data-testid="challenge-margin-change">{signed(result.comparison.target_logit_margin_change, 4)}</td></tr>
  </tbody></table><details><summary>All three logits and probabilities</summary><table><thead><tr><th>Class</th><th>Baseline logit</th><th>Current logit</th><th>Current probability</th></tr></thead><tbody>{CLASSES.map((name,index) => <tr key={name}><th>{name}</th><td>{result.baseline.logits[index].toFixed(4)}</td><td>{result.modified.logits[index].toFixed(4)}</td><td>{(100 * result.modified.probabilities[index]).toFixed(2)}%</td></tr>)}</tbody></table></details></section>;
}
export function BoundaryNote() {
  return <p className="challenge-boundary">Illustrative training examples, not independent research results. Activation, attribution, and intervention effects are different evidence. No student learning benefit has been measured.</p>;
}

export function Terms() {
  return <details className="challenge-terms"><summary>New to neural networks? Read these terms.</summary><dl>
    <dt>Baseline and target</dt><dd>The baseline is the unchanged starting example. The target is the class whose score you are investigating; it need not be the model’s current answer.</dd>
    <dt>Score, probability, and margin</dt><dd>A logit is a raw class score. Softmax converts the three scores to probabilities that sum to 100%, but these are not guaranteed chances of being correct. The target margin is its score minus the highest competing score. A larger margin means a larger lead for the target; the margin can rise while its probability falls.</dd>
    <dt>Saliency and occlusion</dt><dd>Saliency highlights where tiny pixel changes would most affect the target score. Occlusion covers a whole patch and measures the actual result of that larger change.</dd>
    <dt>Channel and ablation</dt><dd>A channel is one learned filter’s response grid. Mean activity is its average response. Ablation switches that channel off to measure its influence. A bright map does not automatically explain what the model uses.</dd>
  </dl></details>;
}
