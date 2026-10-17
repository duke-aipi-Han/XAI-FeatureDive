import type { ExperimentResult } from '../types';
import { CLASSES } from '../types';
export const signed = (value: number, precision = 3) => `${value >= 0 ? '+' : ''}${value.toFixed(precision)}`;

export function PredictionAnswer({result,busy}:{result:ExperimentResult;busy:boolean}) {
  const changed=result.request.interventions.length>0;
  return (
    <div className={`answer-strip ${result.comparison.classification_flip ? 'flipped' : ''}`}>
      <div className="prediction-pair"><span><small>ORIGINAL</small><strong data-testid="baseline-prediction">{result.baseline.predicted_class}</strong></span><span aria-hidden="true">→</span><span><small>NOW</small><strong data-testid="current-prediction">{result.modified.predicted_class}</strong></span></div>
      <span className="outcome" data-testid="outcome" role="status">{busy?'Testing…':result.comparison.classification_flip?'↗ You changed its mind!':changed?'Same answer. Different scores?':'Your move.'}</span>
    </div>
  );
}

export default function Results({ result, onExport }: {
  result: ExperimentResult; onExport: () => void;
}) {
  return <section className="results-hud" aria-label="Prediction comparison">
    <div className="probabilities" aria-label="Original and current class probabilities">
      {CLASSES.map((name,index)=><div key={name} className="probability-row"><div><span>{name}</span><b>{(100*result.baseline.probabilities[index]).toFixed(1)} → {(100*result.modified.probabilities[index]).toFixed(1)}%</b></div><div className="probability-track" aria-hidden="true"><i style={{width:`${100*result.baseline.probabilities[index]}%`}}/><b style={{width:`${100*result.modified.probabilities[index]}%`}}/></div></div>)}
    </div>
    <div className="evidence-options"><details className="score-details"><summary>Detailed scores</summary>
      <p>Target {result.request.target_class} · margin: {result.baseline.target_logit_margin.toFixed(3)} → <b data-testid="current-margin">{result.modified.target_logit_margin.toFixed(3)}</b> · <span data-testid="margin-change">{signed(result.comparison.target_logit_margin_change)} logit units</span></p>
      <p>Margin = target score minus strongest competitor. Probabilities are uncalibrated softmax outputs.</p>
      <table aria-label="Baseline and current logits"><thead><tr><th>Class</th><th>Original</th><th>Now</th><th>Δ logit</th></tr></thead><tbody>{CLASSES.map((name,index)=><tr key={name}><th>{name}</th><td>{result.baseline.logits[index].toFixed(3)}</td><td data-testid={`logit-${name}`}>{result.modified.logits[index].toFixed(3)}</td><td>{signed(result.comparison.logit_changes[index])}</td></tr>)}</tbody></table>
      <details><summary>Model & source details</summary><p>{result.request.model_id} · {result.model_version} · {result.request.sample_id} · training example</p><code>{result.checkpoint_sha256}</code></details>
    </details><button className="export-button" onClick={onExport}>↓ Export experiment JSON</button></div>
  </section>;
}
