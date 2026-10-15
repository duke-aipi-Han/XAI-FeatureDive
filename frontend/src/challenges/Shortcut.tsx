import { useEffect, useMemo, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { api } from '../api';
import PixelCanvas from '../components/PixelCanvas';
import { signed } from '../components/Results';
import { BACKGROUNDS } from '../types';
import type { ExperimentRequest, Sample } from '../types';
import { Baseline, BoundaryNote, Progress, Terms } from './Common';
import { anonymousMapping, downloadRecord, sensitivity, sensitivityWinner } from './state';
import type { AnonymousID, BackgroundObservation, CompletionRecord, SensitivityHypothesis, Step } from './state';

interface Props { samples: Sample[]; active: boolean; onBusy: (busy: boolean) => void; onComplete: (record: CompletionRecord) => void }
const freshMapping = () => anonymousMapping((crypto.getRandomValues(new Uint8Array(1))[0] & 1) === 1);
export default function Shortcut({ samples, active, onBusy, onComplete }: Props) {
  const initial = samples.find(sample => sample.id === 'airplane:72939') ?? samples[0];
  const [sampleID, setSampleID] = useState(initial.id);
  const [mapping, setMapping] = useState(freshMapping);
  const [step, setStep] = useState<Step>('observe');
  const [observations, setObservations] = useState<BackgroundObservation[]>([]);
  const [hypothesis, setHypothesis] = useState<SensitivityHypothesis | null>(null);
  const [prediction, setPrediction] = useState<AnonymousID | 'similar' | ''>('');
  const [reason, setReason] = useState('');
  const [identified, setIdentified] = useState<AnonymousID | 'unclear' | ''>('');
  const [evidence, setEvidence] = useState('');
  const [identification, setIdentification] = useState<CompletionRecord['identification']>();
  const [reflection, setReflection] = useState('');
  const [revealed, setRevealed] = useState(false);
  const [completion, setCompletion] = useState<CompletionRecord | null>(null);
  const [startedAt, setStartedAt] = useState(() => new Date().toISOString());
  const [error, setError] = useState('');
  const [retry, setRetry] = useState(0);
  const [busy, setBusy] = useState(false);
  const lock = useRef(false);
  const sample = samples.find(row => row.id === sampleID)!;
  const startingColor = BACKGROUNDS[sample.label];
  const requests = useMemo<Record<AnonymousID, ExperimentRequest>>(() => ({
    A: { model_id: mapping.A, sample_id: sampleID, background: startingColor, target_class: sample.class, interventions: [] },
    B: { model_id: mapping.B, sample_id: sampleID, background: startingColor, target_class: sample.class, interventions: [] },
  }), [mapping, sampleID, sample.class, startingColor]);
  const key = JSON.stringify(requests);
  const baselineKey = useRef('');
  const baseline = baselineKey.current === key ? observations[0] : undefined;
  const enough = new Set(observations.filter(row => row.background !== 'gray').map(row => row.background)).size === 3;
  const summaryA = enough ? sensitivity(observations, 'A', sample.label) : null;
  const summaryB = enough ? sensitivity(observations, 'B', sample.label) : null;
  const winner = summaryA && summaryB ? sensitivityWinner(summaryA.probabilityRangePoints, summaryB.probabilityRangePoints) : null;
  const newest = observations.at(-1);
  useEffect(() => { if (active) onBusy(busy); }, [active, busy, onBusy]);
  useEffect(() => {
    if (!active || baselineKey.current === key) return;
    let valid = true; const requested_at = new Date().toISOString();
    setBusy(true); setError(''); setObservations([]);
    Promise.all([api.infer(requests.A), api.infer(requests.B)]).then(([A,B]) => {
      if (!valid) return;
      if (A.source_bitmap_sha256 !== B.source_bitmap_sha256 || A.model_version !== B.model_version) throw new Error('These models do not share a matched source/version.');
      baselineKey.current = key;
      setObservations([{ background: startingColor, kind: 'baseline', requested_at, measured_at: new Date().toISOString(), results: { A, B } }]);
    }).catch(problem => { if (valid) setError(problem instanceof Error ? problem.message : 'Could not load both models.'); })
      .finally(() => { if (valid) setBusy(false); });
    return () => { valid = false; setBusy(false); };
  }, [active, key, retry]);

  function record(event: FormEvent) {
    event.preventDefault();
    if (!prediction || reason.trim().length < 5 || !baseline) return;
    setHypothesis(Object.freeze({ model: prediction, reason: reason.trim(), recorded_at: new Date().toISOString() }));
    setStep('intervene'); setError('');
  }
  async function swap(background: typeof startingColor) {
    if (!hypothesis || busy || lock.current || !baseline || observations.some(row => row.background === background)) return;
    lock.current = true; setBusy(true); setError('');
    const requested_at = new Date().toISOString();
    try {
      const operation = { type: 'background_swap' as const, background };
      const [A,B] = await Promise.all([api.infer({ ...requests.A, interventions: [operation] }), api.infer({ ...requests.B, interventions: [operation] })]);
      for (const label of ['A','B'] as const) {
        const result = label === 'A' ? A : B;
        if (result.checkpoint_sha256 !== baseline.results[label].checkpoint_sha256 || result.source_bitmap_sha256 !== baseline.results[label].source_bitmap_sha256 || JSON.stringify(result.baseline.logits) !== JSON.stringify(baseline.results[label].baseline.logits)) throw new Error('The baseline changed. Restart before comparing colors.');
        if (JSON.stringify(result.input.source_alpha) !== JSON.stringify(baseline.results[label].input.source_alpha) || JSON.stringify(result.input.modified_alpha) !== JSON.stringify(result.input.source_alpha)) throw new Error('Foreground geometry changed unexpectedly.');
      }
      setObservations(previous => [...previous, { background, kind: 'swap', requested_at, measured_at: new Date().toISOString(), results: { A,B } }]);
    } catch (problem) { setError(problem instanceof Error ? problem.message : 'Comparison failed. Retry the background; your hypothesis is preserved.'); }
    finally { lock.current = false; setBusy(false); }
  }
  function reveal(event: FormEvent) {
    event.preventDefault();
    if (!identified || evidence.trim().length < 10 || !enough) return;
    setIdentification({ model: identified, evidence: evidence.trim(), recorded_at: new Date().toISOString() });
    setRevealed(true); setStep('reflect');
  }
  function finish(event: FormEvent) {
    event.preventDefault();
    if (!hypothesis || !identification || reflection.trim().length < 10 || !enough || !summaryA || !summaryB || !winner) return;
    const result: CompletionRecord = { schema_version: 1, challenge_id: 'shortcut', source_role: 'illustrative_training_example', started_at: startedAt, completed_at: new Date().toISOString(), reflection: reflection.trim(), hypothesis, observations, identification, model_mapping: mapping, background_feedback: { A: summaryA, B: summaryB, observed_more_sensitive: winner, hypothesis_matches: hypothesis.model === winner, identification_matches_regime: identification.model === 'unclear' ? null : mapping[identification.model] === 'shortcut' } };
    setCompletion(result); setStep('complete'); onComplete(result);
  }
  function restart() {
    setMapping(freshMapping()); setStep('observe'); setHypothesis(null); setPrediction(''); setReason(''); setIdentified(''); setEvidence(''); setIdentification(undefined); setReflection(''); setRevealed(false); setCompletion(null); setStartedAt(new Date().toISOString());
    baselineKey.current = ''; setObservations([]); setError(''); setRetry(value=>value+1);
  }
  return <section className="guided-investigation" aria-label="The Shortcut Trap" aria-busy={busy}>
    <div className="challenge-title"><span className="eyebrow">CHALLENGE 03</span><h2>The Shortcut Trap</h2><p>Which model depends more on the background?</p></div>
    <Progress step={step} />
    <Terms />
    <div className="challenge-setup"><label>Drawing<select aria-label="Shortcut drawing" value={sampleID} disabled={busy || !!hypothesis} onChange={event => setSampleID(event.target.value)}>{samples.map(item=><option key={item.id} value={item.id}>{item.id}</option>)}</select></label><span>Same drawing · target {sample.class} · baseline {startingColor}</span><button disabled={busy} onClick={restart}>Restart with new A/B labels</button></div>
    {error && <div className="error-banner" role="alert">{error}{!baseline && <button onClick={()=>setRetry(value=>value+1)}>Retry example</button>}</div>}
    {!baseline ? <div className="loading-lab" role="status">{error ? 'Example unavailable.' : 'Loading two matched models…'}</div> : <>
      <div className="challenge-layout"><div className="challenge-workspace">
        {step === 'observe' && <><h3>Same drawing. Two anonymous models.</h3><p>A and B use the same network architecture but were trained under different conditions. Their identities stay hidden until you record an evidence-based identification. Start by examining the original predictions.</p><div className="shortcut-baselines">{(['A','B'] as const).map(label=><div key={label}><h3>Model {label}</h3><PixelCanvas rgb={baseline.results[label].input.baseline_rgb_chw} label={`Model ${label} baseline drawing`} /><Baseline prediction={baseline.results[label].baseline} sample={sampleID} /></div>)}</div><button className="primary-button" onClick={()=>setStep('hypothesize')}>Predict background reliance →</button></>}
        {step === 'hypothesize' && <form className="hypothesis-form" onSubmit={record}><h3>Commit a prediction before changing colors.</h3><label>Which model will change more?<select aria-label="Background reliance prediction" value={prediction} required onChange={event=>setPrediction(event.target.value as typeof prediction)}><option value="" disabled>Choose A, B, or similar</option><option value="A">Model A</option><option value="B">Model B</option><option value="similar">Similar response</option></select></label><p>We will compare the target probability range across green, blue, and yellow. “Similar” means the two ranges differ by at most 1 percentage point; this is a teaching tolerance, not a statistical test.</p><label>What led you to that prediction?<textarea aria-label="Background hypothesis reason" value={reason} minLength={5} maxLength={1200} required onChange={event=>setReason(event.target.value)} /></label><div className="challenge-actions"><button type="button" onClick={()=>setStep('observe')}>Back to observation</button><button className="primary-button" disabled={!prediction || reason.trim().length<5}>Record background hypothesis</button></div></form>}
        {step === 'intervene' && hypothesis && <><span className="eyebrow">HYPOTHESIS RECORDED</span><h3>Change only the background.</h3><p data-testid="background-recorded-hypothesis">Your prediction: {hypothesis.model === 'similar' ? 'similar response' : `Model ${hypothesis.model} changes more`}.</p><blockquote>{hypothesis.reason}</blockquote><p>Each color is compared independently with the original baseline. Test green, blue, and yellow before evaluating. Gray is optional and outside the training color distribution.</p><div className="shortcut-color-buttons">{BACKGROUNDS.map(color=><button key={color} disabled={busy || observations.some(row=>row.background===color)} onClick={()=>void swap(color)} aria-label={`Test ${color} background`}>{color}<span>{observations.some(row=>row.background===color) ? '✓ measured' : 'test →'}</span></button>)}</div>{newest && <div className="challenge-before-after"><div><span>Original baseline</span><PixelCanvas rgb={baseline.results.A.input.baseline_rgb_chw} label="Original shared drawing" /></div><div><span>{newest.background} background</span><PixelCanvas rgb={newest.results.A.input.modified_rgb_chw} label="Shared drawing after background swap" /></div></div>}<button className="primary-button" disabled={!enough || busy} onClick={()=>setStep('evaluate')}>Evaluate background evidence →</button></>}
        {step === 'evaluate' && summaryA && summaryB && winner && <><h3>Identify the training regime from your evidence.</h3><div className="sensitivity-cards">{(['A','B'] as const).map(label=>{const summary=label==='A'?summaryA:summaryB;return <div key={label}><span>Model {label}</span><strong>{summary.probabilityRangePoints.toFixed(2)}<small> probability points</small></strong><p>Target-margin range: {summary.marginRange.toFixed(4)} logit units.<br />Any label flip: {summary.anyLabelFlip?'yes':'no'}.</p></div>;})}</div><div className="measured-feedback"><b>{winner==='similar' ? 'These probability ranges are similar under the stated tolerance.' : `Model ${winner} has the larger measured probability range for this drawing.`}</b><p>Your original sensitivity prediction {hypothesis?.model===winner?'matched':'did not match'} this comparison. Greater measured sensitivity is evidence to consider, not a guaranteed training-regime identifier for every drawing.</p></div><form className="hypothesis-form" onSubmit={reveal}><label>Which model was trained with the color shortcut?<select aria-label="Identify shortcut-trained model" value={identified} required onChange={event=>setIdentified(event.target.value as typeof identified)}><option value="" disabled>Record an identification</option><option value="A">Model A</option><option value="B">Model B</option><option value="unclear">I cannot tell from this example</option></select></label><label>State the measured evidence for your choice.<textarea aria-label="Identification evidence" value={evidence} minLength={10} maxLength={2000} required onChange={event=>setEvidence(event.target.value)} placeholder="Refer to backgrounds, score changes, probability ranges, or label flips." /></label><button className="primary-button" disabled={!identified || evidence.trim().length<10}>Record identification & reveal</button></form></>}
        {revealed && <section className="model-reveal" aria-label="Training regimes revealed"><span className="eyebrow">TRAINING REGIMES REVEALED</span><h3>Model A: {mapping.A} · Model B: {mapping.B}</h3><p><b>Balanced:</b> each background color had probability 1/3 independently of the drawing category. This is color balancing, not a guarantee of fairness.</p><p><b>Shortcut:</b> cats usually appeared on green, fish on blue, and airplanes on yellow (90% each; 5% for each other color). Those accidental cues could influence the learned model.</p><p>{identification?.model==='unclear' ? 'You recorded that this example was insufficient to identify the regime. That uncertainty is retained.' : identification && mapping[identification.model]==='shortcut' ? 'Your identification matches the checkpoint’s recorded training regime.' : 'Your identification does not match the checkpoint’s recorded training regime.'} The full measured responses remain visible; the training label does not override them.</p></section>}
        {step === 'reflect' && <form className="reflection-form" onSubmit={finish}><h3>Reflect on the shortcut.</h3><label>What supports or weakens a claim of background reliance? Include one measured comparison and one reason to avoid generalizing from this single drawing.<textarea aria-label="Shortcut reflection" value={reflection} minLength={10} maxLength={2000} required onChange={event=>setReflection(event.target.value)} /></label><button className="primary-button" disabled={reflection.trim().length<10}>Complete challenge</button></form>}
        {step === 'complete' && completion && <div className="challenge-complete"><span className="eyebrow">INVESTIGATION COMPLETE</span><h3>You tested an accidental clue.</h3><p>A model can classify familiar examples well while changing its answers when context changes. Keeping the foreground fixed makes this specific background comparison interpretable; it does not establish a unique internal mechanism.</p><blockquote>{completion.reflection}</blockquote><button className="primary-button" onClick={()=>downloadRecord(completion)}>Download challenge record</button><p className="micro-copy">Includes the original hypothesis, independent background requests and outputs, pre-reveal identification, training-regime mapping, and reflection.</p></div>}
      </div><aside className="challenge-notes"><h3>Keep these fixed.</h3><ul><li>The original source drawing and stroke alpha</li><li>The selected target class</li><li>Each saved model checkpoint</li><li>The original baseline for both models</li></ul><p>Only the background color changes. Numeric comparisons include probabilities, class logits, and target margins. Softmax probabilities are uncalibrated.</p><p>A/B labels are randomly assigned for this attempt and are a teaching device. Model identities are concealed in the lesson interface until the reveal.</p></aside></div>
      {hypothesis && <section className="trial-ledger"><h3>Recorded A/B responses</h3><div className="challenge-table-scroll"><table><thead><tr><th>Color</th><th>Model</th><th>Prediction</th><th>Target probability</th><th>Target logit</th><th>Target margin</th><th>Margin Δ</th></tr></thead><tbody>{observations.flatMap(row=>(['A','B'] as const).map(label=>{const result=row.results[label];return <tr key={`${row.background}-${label}`}><th>{row.background}{row.kind==='baseline'?' (baseline)':''}</th><td>{label}</td><td>{result.modified.predicted_class}</td><td>{(100*result.modified.probabilities[sample.label]).toFixed(2)}%</td><td>{result.modified.target_logit.toFixed(4)}</td><td>{result.modified.target_logit_margin.toFixed(4)}</td><td>{signed(result.comparison.target_logit_margin_change,4)}</td></tr>;}))}</tbody></table></div><p className="micro-copy">Only measured rows are shown. The sensitivity comparison excludes gray and holds all three training colors together.</p></section>}
    </>}
    <BoundaryNote />
  </section>;
}
