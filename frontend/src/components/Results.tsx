import type { ExperimentResult, HistoryEntry } from '../types';
import { CLASSES } from '../types';
export const signed = (value: number, precision = 3) => `${value >= 0 ? '+' : ''}${value.toFixed(precision)}`;
export default function Results({ result, history, cursor, busy, onRestore, onReset, onExport }: {
  result: ExperimentResult; history: HistoryEntry[]; cursor: number; busy: boolean;
  onRestore: (index: number) => void; onReset: () => void; onExport: () => void;
}) {
  return <aside className="results-panel" aria-label="Prediction comparison" aria-busy={busy}>
    <div className="panel-heading"><span className="eyebrow">03 / MEASURE</span><span className="live-dot">OBSERVED EFFECT</span></div>
    <h2>What changed?</h2>
    <div className="prediction-pair">
      <div><span className="small-label">Baseline</span><strong data-testid="baseline-prediction">{result.baseline.predicted_class}</strong></div>
      <span className="prediction-arrow" aria-hidden="true">→</span>
      <div><span className="small-label">Current</span><strong data-testid="current-prediction">{result.modified.predicted_class}</strong></div>
    </div>
    <div className={`outcome ${result.comparison.classification_flip ? 'flipped' : ''}`} data-testid="outcome">
      <span aria-hidden="true">{result.comparison.classification_flip ? '↗' : '='}</span>
      {result.comparison.classification_flip ? 'Predicted class changed' : 'Predicted class unchanged'}
    </div>
    <div className="table-caption"><span>Class probabilities</span><span>BASE → NOW</span></div>
    {CLASSES.map((name, index) => <div className="probability-row" key={name}>
      <div><span>{name}</span><b>{(100 * result.baseline.probabilities[index]).toFixed(1)} <span>→</span> {(100 * result.modified.probabilities[index]).toFixed(1)}%</b></div>
      <div className="probability-track" aria-hidden="true"><span className="baseline-bar" style={{ width: `${100 * result.baseline.probabilities[index]}%` }} /><span className="current-bar" style={{ width: `${100 * result.modified.probabilities[index]}%` }} /></div>
      <small>{signed(result.comparison.probability_point_changes[index], 2)} percentage points</small>
    </div>)}
    <p className="micro-copy">Softmax probabilities are uncalibrated.</p>
    <details className="score-details"><summary>Detailed scores</summary>
    <div className="margin-card">
      <span className="small-label">Target: {result.request.target_class} · logit margin</span>
      <div><b>{result.baseline.target_logit_margin.toFixed(3)}</b><span>→</span><b data-testid="current-margin">{result.modified.target_logit_margin.toFixed(3)}</b></div>
      <span className="delta" data-testid="margin-change">{signed(result.comparison.target_logit_margin_change)} logit units</span>
      <p>Target score minus the strongest competing score.</p>
    </div>
    <details className="raw-scores" open><summary>Raw class scores (logits)</summary>
      <table aria-label="Baseline and current logits"><thead><tr><th>Class</th><th>Base</th><th>Now</th><th>Δ</th></tr></thead><tbody>
        {CLASSES.map((name, index) => <tr key={name}><th>{name}</th><td>{result.baseline.logits[index].toFixed(3)}</td><td data-testid={`logit-${name}`}>{result.modified.logits[index].toFixed(3)}</td><td>{signed(result.comparison.logit_changes[index])}</td></tr>)}
      </tbody></table>
    </details>
    </details>
    <div className="history-heading"><h3>Experiment history</h3><span>{cursor} / {history.length - 1}</span></div>
    <div className="history-actions"><button disabled={busy || cursor === 0} onClick={() => onRestore(cursor - 1)}>↶ Undo</button><button disabled={busy || cursor === 0} onClick={onReset}>↺ Reset baseline</button></div>
    <ol className="history-list">{history.map((entry, index) => <li key={`${index}-${entry.result.experiment_id}`}>
      <button onClick={() => onRestore(index)} disabled={busy} aria-current={index === cursor ? 'step' : undefined}>
        <span className="history-number">{String(index).padStart(2, '0')}</span><span>{entry.label}</span><span className="history-marker">{index === cursor ? '●' : '↗'}</span>
      </button>
    </li>)}</ol>
    <button className="export-button" onClick={onExport}>↓ Export experiment JSON</button>
    <details className="provenance"><summary>Model & source details</summary>
      <p>{result.request.model_id} / {result.model_version}<br />Source: {result.request.sample_id}<br />Illustrative training example.</p>
      <span>Checkpoint SHA-256</span><code>{result.checkpoint_sha256}</code>
      <span>Target logit change</span><code>{signed(result.comparison.target_logit_change)}</code>
    </details>
  </aside>;
}
