import { useEffect, useMemo, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { api } from '../api';
import PixelCanvas from '../components/PixelCanvas';
import { signed } from '../components/Results';
import { BACKGROUNDS, CLASSES } from '../types';
import type { ActivationResult, ClassName, ExperimentRequest, ExperimentResult, Region, RGBImage, SaliencyResult, Sample } from '../types';
import { Baseline, BoundaryNote, MeasuredComparison, Progress, Terms } from './Common';
import { CHALLENGES, distinctSelections, downloadRecord, feedback } from './state';
import type { CompletionRecord, Direction, Magnitude, Step, Trial } from './state';

type Observation = { key: string; baseline: ExperimentResult; saliency?: SaliencyResult; activation?: ActivationResult };
interface Props { id: 'evidence' | 'hidden'; samples: Sample[]; active: boolean; onBusy: (busy: boolean) => void; onComplete: (record: CompletionRecord) => void }
export default function Investigation({ id, samples, active, onBusy, onComplete }: Props) {
  const definition = CHALLENGES.find(item => item.id === id)!;
  const initial = samples.find(sample => sample.id === definition.sample) ?? samples[0];
  const [sampleID, setSampleID] = useState(initial.id);
  const [target, setTarget] = useState<ClassName>(initial.class);
  const [layer, setLayer] = useState<'conv1' | 'conv2'>('conv2');
  const [step, setStep] = useState<Step>('observe');
  const [selection, setSelection] = useState<Trial['selection'] | null>(null);
  const [observation, setObservation] = useState<Observation | null>(null);
  const [trials, setTrials] = useState<Trial[]>([]);
  const [direction, setDirection] = useState<Direction | ''>('');
  const [magnitude, setMagnitude] = useState<Magnitude | ''>('');
  const [reason, setReason] = useState('');
  const [reflection, setReflection] = useState('');
  const [completion, setCompletion] = useState<CompletionRecord | null>(null);
  const [startedAt, setStartedAt] = useState(() => new Date().toISOString());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [retry, setRetry] = useState(0);
  const [overlay, setOverlay] = useState(true);
  const lock = useRef(false);
  const sample = samples.find(item => item.id === sampleID)!;
  const request = useMemo<ExperimentRequest>(() => ({ model_id: 'balanced', sample_id: sampleID, target_class: target, background: BACKGROUNDS[sample.label], interventions: [] }), [sampleID, target, sample.label]);
  const key = JSON.stringify([request, id === 'hidden' ? layer : 'saliency']);
  const data = observation?.key === key ? observation : null;
  const latest = trials.at(-1);
  const observedEffect = latest?.result ? feedback(latest) : null;
  const count = distinctSelections(trials);
  useEffect(() => { if (active) onBusy(busy); }, [active, busy, onBusy]);
  useEffect(() => {
    if (!active || observation?.key === key) return;
    let valid = true; const controller = new AbortController();
    setBusy(true); setError(''); setSelection(null);
    const fetchObservation = id === 'evidence'
      ? api.saliency(request, controller.signal).then(value => ({ key, baseline: value.experiment, saliency: value }))
      : api.activations(request, layer, controller.signal).then(value => ({ key, baseline: value.experiment, activation: value }));
    fetchObservation.then(value => { if (valid) setObservation(value); })
      .catch(problem => { if (valid) setError(problem instanceof Error ? problem.message : 'Could not load this example.'); })
      .finally(() => { if (valid) setBusy(false); });
    return () => { valid = false; controller.abort(); setBusy(false); };
  }, [active, key, retry]);

  function choosePatch(x: number, y: number) {
    if (!data?.saliency || step !== 'observe' || busy) return;
    if (trials.some(trial => trial.result && trial.selection.kind === 'patch' && trial.selection.x === x && trial.selection.y === y)) { setError('That patch has already been measured. Choose a different patch.'); return; }
    setError('');
    const scores = data.saliency.baseline_scores;
    const value = scores.slice(y, y + 4).reduce((sum, row) => sum + row.slice(x, x + 4).reduce((a,b) => a+b, 0), 0);
    setSelection({ kind: 'patch', x, y, saliency_sum: value });
  }
  function fromCanvas(region: Region) {
    if ('rectangle' in region) choosePatch(Math.floor(region.rectangle.x / 4) * 4, Math.floor(region.rectangle.y / 4) * 4);
  }
  function recordHypothesis(event: FormEvent) {
    event.preventDefault();
    if (!selection || !direction || !magnitude || reason.trim().length < 5 || !data) return;
    const hypothesis = Object.freeze({ direction, magnitude, reason: reason.trim(), recorded_at: new Date().toISOString() });
    setTrials(previous => [...previous, { selection: { ...selection }, hypothesis }]);
    setStep('intervene'); setError('');
  }
  async function intervene() {
    if (!latest || latest.result || !data || lock.current || busy) return;
    lock.current = true; setBusy(true); setError('');
    const index = trials.length - 1;
    const selected = latest.selection;
    const requested_at = new Date().toISOString();
    const operation = selected.kind === 'patch'
      ? { type: 'controlled_occlusion' as const, region: { rectangle: { x: selected.x, y: selected.y, width: 4, height: 4 } }, replacement_rgb: [128,128,128] }
      : { type: 'channel_ablation' as const, layer: selected.layer, channel: selected.channel };
    try {
      const result = await api.infer({ ...request, interventions: [operation] });
      if (result.checkpoint_sha256 !== data.baseline.checkpoint_sha256 || JSON.stringify(result.baseline.logits) !== JSON.stringify(data.baseline.baseline.logits)) throw new Error('The baseline changed. Restart the challenge before comparing results.');
      const measured_at = new Date().toISOString();
      setTrials(previous => previous.map((trial, i) => i === index ? { ...trial, requested_at, measured_at, result, feedback: feedback({ ...trial, result }) } : trial));
      setStep('evaluate');
    } catch (problem) { setError(problem instanceof Error ? problem.message : 'Intervention failed. Your hypothesis is preserved; retry it.'); }
    finally { lock.current = false; setBusy(false); }
  }
  function another() {
    setSelection(null); setDirection(''); setMagnitude(''); setReason(''); setStep('observe'); setError('');
  }
  function finish(event: FormEvent) {
    event.preventDefault();
    if (count < 2 || reflection.trim().length < 10) return;
    const record: CompletionRecord = { schema_version: 1, challenge_id: id, source_role: 'illustrative_training_example', started_at: startedAt, completed_at: new Date().toISOString(), reflection: reflection.trim(), trials };
    setCompletion(record); setStep('complete'); onComplete(record);
  }
  function restart() {
    setStep('observe'); setTrials([]); setCompletion(null); setSelection(null); setDirection(''); setMagnitude(''); setReason(''); setReflection(''); setError(''); setStartedAt(new Date().toISOString());
  }
  const activation = data?.activation;
  const maps = activation?.baseline_values as RGBImage | undefined;
  const selectedChannel = selection?.kind === 'channel' ? selection.channel : null;
  return <section className="guided-investigation" aria-label={definition.title} aria-busy={busy}>
    <div className="challenge-title"><span className="eyebrow">CHALLENGE {definition.number}</span><h2>{definition.title}</h2><p>{definition.question}</p></div>
    <Progress step={step} />
    <Terms />
    <div className="challenge-setup"><label>Drawing<select aria-label="Challenge drawing" value={sampleID} disabled={busy || trials.length > 0 || step !== 'observe'} onChange={event => { const item = samples.find(row => row.id === event.target.value)!; setSampleID(item.id); setTarget(item.class); }}>{samples.map(item => <option key={item.id} value={item.id}>{item.id}</option>)}</select></label>
      <label>Target class<select aria-label="Challenge target class" value={target} disabled={busy || trials.length > 0 || step !== 'observe'} onChange={event => setTarget(event.target.value as ClassName)}>{CLASSES.map(name => <option key={name}>{name}</option>)}</select></label>
      {id === 'hidden' && <label>Layer<select aria-label="Challenge layer" value={layer} disabled={busy || trials.length > 0 || step !== 'observe'} onChange={event => setLayer(event.target.value as typeof layer)}><option value="conv1">Conv1 · 8 maps</option><option value="conv2">Conv2 · 16 maps</option></select></label>}
      <span>Balanced model · {request.background} baseline · independent interventions</span><button type="button" disabled={busy} onClick={restart}>Restart challenge</button>
    </div>
    {error && <div role="alert" className="error-banner">{error}{!data && <button onClick={() => setRetry(value => value + 1)}>Retry example</button>}</div>}
    {!data ? <div className="loading-lab" role="status">{error ? 'Example unavailable.' : 'Loading real observations…'}</div> : <>
      <div className="challenge-layout"><div className="challenge-workspace">
        {step === 'observe' && <>
          <h3>Observe, then choose {id === 'evidence' ? 'a region' : 'a channel'}.</h3>
          <p>{id === 'evidence' ? 'The overlay shows local sensitivity of your target score. Pick a 4 × 4 region to test. Choosing it does not run an intervention yet.' : 'Each channel is one learned filter’s response grid. Pick a channel to test. Its mean activity is its average response strength, not how much it controls the answer.'}</p>
          {id === 'evidence' && data.saliency ? <>
            <label className="challenge-checkbox"><input type="checkbox" checked={overlay} onChange={event => setOverlay(event.target.checked)} />Show Saliency Scope</label>
            <div className="challenge-image"><PixelCanvas rgb={data.baseline.input.baseline_rgb_chw} saliency={overlay ? data.saliency.baseline_scores : undefined} opacity={.65} label="Challenge drawing; choose a patch" interactive={!busy} mode="patch" onSelect={fromCanvas} />
              {selection?.kind === 'patch' && <span className="chosen-patch" style={{ left: `${selection.x/28*100}%`, top: `${selection.y/28*100}%`, width: `${4/28*100}%`, height: `${4/28*100}%` }} />}
            </div>
            <p className="challenge-scale">Overlay: sum |RGB gradients| of the target logit, contrast scaled per image. Every patch uses the same 4 × 4 fixed-gray occlusion. Coordinates (x,y) count pixels from the left and top, starting at zero.</p>
            <div className="patch-picker" aria-label="Patch selection">{Array.from({length:49},(_,index) => { const x=index%7*4,y=Math.floor(index/7)*4; return <button key={index} aria-label={`Select patch ${x},${y}`} aria-pressed={selection?.kind==='patch' && selection.x===x && selection.y===y} onClick={() => choosePatch(x,y)} disabled={busy || trials.some(trial => trial.result && trial.selection.kind==='patch' && trial.selection.x===x && trial.selection.y===y)}>{x},{y}</button>; })}</div>
          </> : maps && activation && <>
            <div className="challenge-channel-grid">{maps.map((map,index) => <button key={index} aria-label={`Choose channel ${index}`} aria-pressed={selectedChannel===index} disabled={busy || trials.some(trial=>trial.result && trial.selection.kind==='channel' && trial.selection.layer===layer && trial.selection.channel===index)} onClick={() => setSelection({kind:'channel',layer,channel:index,mean_activation:activation.baseline_mean_by_channel[index]})}><PixelCanvas values={map} upper={activation.baseline_max} label={`Channel ${index} activation`} /><b>Channel {index}</b><small>Mean {activation.baseline_mean_by_channel[index].toFixed(4)}</small></button>)}</div>
            <p className="challenge-scale">Shared display scale: 0 → {activation.baseline_max.toFixed(4)} raw activation units. Channel indices are zero-based.</p>
          </>}
          <div className="selection-summary">{selection?.kind==='patch' ? `Selected patch (${selection.x},${selection.y}); saliency sum ${selection.saliency_sum.toFixed(4)}.` : selection?.kind==='channel' ? `Selected ${selection.layer.toUpperCase()} channel ${selection.channel}; mean activity ${selection.mean_activation.toFixed(4)}.` : 'Make a selection to continue.'}</div>
          <button className="primary-button" disabled={!selection || busy} onClick={() => setStep('hypothesize')}>Make a hypothesis →</button>
        </>}
        {step === 'hypothesize' && <form className="hypothesis-form" onSubmit={recordHypothesis}>
          <h3>Predict the effect before you see it.</h3><p>How will {id==='evidence' ? 'covering this region with fixed gray' : 'zeroing this channel'} change the target margin? This is the target score minus the strongest competitor.</p>
          <label>Predicted direction<select aria-label="Predicted direction" value={direction} required onChange={event => setDirection(event.target.value as Direction)}><option value="" disabled>Choose a prediction</option><option value="decrease">Decrease · away from the target</option><option value="little_change">Little change · within ±0.1 logit units</option><option value="increase">Increase · toward the target</option></select></label>
          <label>Predicted magnitude<select aria-label="Predicted magnitude" value={magnitude} required onChange={event => setMagnitude(event.target.value as Magnitude)}><option value="" disabled>Choose a size</option><option value="small">Small · absolute change below 0.5</option><option value="moderate">Moderate · 0.5 to below 2.0</option><option value="large">Large · 2.0 or more</option></select></label>
          <label>Why do you expect this?<textarea aria-label="Hypothesis reason" value={reason} minLength={5} maxLength={1200} required onChange={event => setReason(event.target.value)} placeholder="Use something you observed in the image or activation map." /></label>
          <p className="micro-copy">These thresholds are teaching categories, not scientific significance tests. Your hypothesis will be saved separately from the measurements.</p>
          <div className="challenge-actions"><button type="button" onClick={() => setStep('observe')}>Back to observation</button><button className="primary-button" disabled={!direction || !magnitude || reason.trim().length<5}>Record hypothesis</button></div>
        </form>}
        {step === 'intervene' && latest && <div className="intervention-confirmation"><span className="eyebrow">HYPOTHESIS RECORDED</span><h3>Now test your prediction.</h3><p data-testid="recorded-hypothesis">You predicted {latest.hypothesis.direction.replace('_',' ')} with a {latest.hypothesis.magnitude} effect.</p><blockquote>{latest.hypothesis.reason}</blockquote><p>{latest.selection.kind==='patch' ? `The complete patch at (${latest.selection.x},${latest.selection.y}) will be replaced with RGB (128,128,128), including background pixels. This is controlled occlusion, not stroke-only erasure.` : `Channel ${latest.selection.channel} in ${latest.selection.layer.toUpperCase()} will be set to zero after ReLU and before pooling.`}</p><p>Each trial starts from the original baseline. Earlier interventions do not accumulate.</p><button className="ablate-button" disabled={busy} onClick={() => void intervene()}>{busy ? 'Measuring…' : id==='evidence' ? 'Run controlled occlusion' : 'Run channel ablation'}</button></div>}
        {step === 'evaluate' && latest?.result && observedEffect && <>
          <h3>Compare your hypothesis with the measurement.</h3>
          {id==='evidence' && <div className="challenge-before-after"><div><span>Baseline</span><PixelCanvas rgb={latest.result.input.baseline_rgb_chw} label="Challenge baseline drawing" /></div><div><span>After occlusion</span><PixelCanvas rgb={latest.result.input.modified_rgb_chw} label="Challenge occluded drawing" /></div></div>}
          <MeasuredComparison result={latest.result} />
          <div className="measured-feedback" data-testid="measured-feedback"><b>Observed: {observedEffect.observed.direction.replace('_',' ')} · {observedEffect.observed.magnitude} effect.</b><p>Your direction prediction {observedEffect.directionMatches ? 'matched' : 'did not match'} this measurement. Your magnitude prediction {observedEffect.magnitudeMatches ? 'matched' : 'did not match'} the measured category.</p><p>{id==='evidence' ? 'A local gradient and a finite occlusion answer different questions. The margin also includes a competing class score. Agreement or disagreement here does not establish that all heatmaps are reliable or unreliable.' : 'Activation magnitude and class influence are different. Downstream weights and other channels affect the result. A zero effect is a valid observation; a positive margin change after ablation is also possible.'}</p></div>
          <div className="challenge-actions"><button disabled={busy || trials.length>=8} onClick={another}>Compare another {id==='evidence' ? 'patch' : 'channel'}</button><button className="primary-button" disabled={count<2} onClick={() => setStep('reflect')}>Reflect on the evidence →</button></div><p className="micro-copy">Test at least two different {id==='evidence' ? 'patches' : 'channels'} to finish. Up to eight trials per attempt. This is comparison, not a score for guessing correctly.</p>
        </>}
        {step === 'reflect' && <form className="reflection-form" onSubmit={finish}><h3>Explain what you learned from this comparison.</h3><label>{id==='evidence' ? 'How did the saliency and occlusion results agree or disagree? Include a measured score change and a limit of this explanation.' : 'Did the most active of your tested channels have the largest effect? Include a measured score change and explain what you cannot infer about the channel’s meaning.'}<textarea aria-label="Challenge reflection" value={reflection} minLength={10} maxLength={2000} required onChange={event => setReflection(event.target.value)} /></label><p>Your words are a reflection, not evidence that the teaching experience improved learning.</p><div className="challenge-actions"><button type="button" onClick={() => setStep('evaluate')}>Back to measured results</button><button className="primary-button" disabled={reflection.trim().length<10}>Complete challenge</button></div></form>}
        {step === 'complete' && completion && <div className="challenge-complete"><span className="eyebrow">INVESTIGATION COMPLETE</span><h3>You made predictions and tested them.</h3><p>{id==='evidence' ? 'Attribution describes sensitivity; an input intervention measures the effect of a specific change. Neither alone identifies a unique internal explanation.' : 'High activity is not automatically high influence. Channel ablation measures an effect within this model for this input, not proof of a semantic detector.'}</p><blockquote>{completion.reflection}</blockquote><button className="primary-button" onClick={() => downloadRecord(completion)}>Download challenge record</button><p className="micro-copy">Includes your original hypotheses, exact requests, measured outputs, feedback categories, and reflection. Stored only in this browser session until exported or reloaded.</p></div>}
      </div><aside className="challenge-notes"><Baseline prediction={data.baseline.baseline} sample={sampleID} /><h3>Keep the comparison fair.</h3><p>The source drawing, target class, model checkpoint, and starting background stay fixed after your first hypothesis.</p><p>{id==='evidence' ? 'Every trial covers 16 pixels with the same neutral-gray replacement. Masking can create an unusual image.' : 'Every trial zeros just one channel, independently from baseline. Mean activation alone does not show causal importance.'}</p><details><summary>Why these examples?</summary><p>Default drawings come from the documented Phase 2 teaching scans. They are curated training examples. You may choose another catalog drawing before recording a hypothesis; no particular outcome is guaranteed.</p></details></aside></div>
      {trials.some(trial => trial.result) && <section className="trial-ledger"><h3>Your independent comparisons</h3><div className="challenge-table-scroll"><table><thead><tr><th>Selection</th><th>{id==='evidence' ? 'Saliency sum' : 'Mean activation'}</th><th>Hypothesis</th><th>Measured margin Δ</th><th>Probability Δ</th></tr></thead><tbody>{trials.filter(trial=>trial.result).map((trial,index)=><tr key={index}><th>{trial.selection.kind==='patch' ? `(${trial.selection.x},${trial.selection.y})` : `${trial.selection.layer} / ${trial.selection.channel}`}</th><td>{(trial.selection.kind==='patch' ? trial.selection.saliency_sum : trial.selection.mean_activation).toFixed(4)}</td><td>{trial.hypothesis.direction.replace('_',' ')} / {trial.hypothesis.magnitude}</td><td>{signed(trial.result!.comparison.target_logit_margin_change,4)}</td><td>{signed(trial.result!.comparison.target_probability_point_change,2)} points</td></tr>)}</tbody></table></div><p className="micro-copy">Compare only the alternatives you actually tested. Both positive and negative effects, including null effects, are retained.</p></section>}
    </>}
    <BoundaryNote />
  </section>;
}
