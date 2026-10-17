import { useCallback, useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { api } from './api';
import PixelCanvas from './components/PixelCanvas';
import Results, { PredictionAnswer } from './components/Results';
import ChallengeHub from './challenges/ChallengeHub';
import { FLIP_EXAMPLES, HINT_EXAMPLES } from './examples';
import HintPanel from './components/HintPanel';
import { operationLabel, withIntervention } from './state';
import { BACKGROUNDS, CLASSES, PLANES } from './types';
import type { ActivationResult, Background, ClassName, ExperimentRequest, ExperimentResult, Intervention, Layer, Matrix, ModelCatalog, ModelID, Plane, Region, RGBImage, SaliencyResult, Sample, HintResult, ChannelHint } from './types';

const COLOR_CSS = { green: '#46aa5a', blue: '#4178d2', yellow: '#e6c337', gray: '#808080' };
const COPY: Record<Plane, { title: string; subtitle: string; tag: string; note: string }> = {
  input: { title: 'Start with the pixels.', subtitle: 'The drawing the network actually sees.', tag: '28 × 28 × 3', note: 'Observe the drawing. Change its context or cover a region, then measure what happens.' },
  conv1: { title: 'Patterns, one layer in.', subtitle: 'Eight maps of post-ReLU activity.', tag: '8 × 28 × 28', note: 'A bright response shows activity. Zero a channel to measure its influence on this decision.' },
  conv2: { title: 'Look deeper.', subtitle: 'Sixteen maps. Different responses.', tag: '16 × 14 × 14', note: 'Strong activity does not necessarily mean strong influence. Test a channel and compare the scores.' },
  dense: { title: 'The hidden representation.', subtitle: '32 values connecting maps to class scores.', tag: '32 NEURONS', note: 'These are post-ReLU magnitudes. They show responses, not named concepts or proven causes.' },
  output: { title: 'Read the decision.', subtitle: 'Three class scores, one predicted label.', tag: '3 LOGITS', note: 'Compare the original and current scores. A change can matter even when the predicted class stays the same.' },
};
const errorText = (error: unknown) => error instanceof Error ? error.message : 'Something went wrong. Please try again.';

export default function App() {
  const [mode, setMode] = useState<'explore' | 'challenges'>('explore');
  const [challengeBusy, setChallengeBusy] = useState(false);
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [draft, setDraft] = useState({ model_id: 'balanced' as ModelID, sample_id: '', background: 'green' as Background, target_class: 'cat' as ClassName });
  const [baseline, setBaseline] = useState<ExperimentResult | null>(null);
  const [current, setCurrent] = useState<ExperimentResult | null>(null);
  const [plane, setPlane] = useState<Plane>('conv1');
  const [channels, setChannels] = useState({ conv1: 7, conv2: 0 });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [startupError, setStartupError] = useState('');
  const [boot, setBoot] = useState(0);
  const [view, setView] = useState<'current' | 'baseline'>('current');
  const [tool, setTool] = useState<'inspect' | 'drawing_erasure' | 'controlled_occlusion'>('inspect');
  const [shape, setShape] = useState<'pixel' | 'patch' | 'rectangle'>('patch');
  const [grid, setGrid] = useState(false);
  const [showSaliency, setShowSaliency] = useState(false);
  const [opacity, setOpacity] = useState(.6);
  const [activation, setActivation] = useState<{ key: string; data: ActivationResult } | null>(null);
  const [saliency, setSaliency] = useState<{ key: string; data: SaliencyResult } | null>(null);
  const [featurePower,setFeaturePower] = useState(true);
  const [erasePower,setErasePower] = useState(true);
  const [hints,setHints] = useState<{key:string;data:HintResult}|null>(null);
  const [hintError,setHintError] = useState('');
  const [hintRetry,setHintRetry] = useState(0);
  const hintCache = useRef(new Map<string,HintResult>());
  const [viewError, setViewError] = useState('');
  const [retryView, setRetryView] = useState(0);
  const [coordinates, setCoordinates] = useState({ x: 12, y: 12, width: 4, height: 4 });

  const [announcement, setAnnouncement] = useState('');
  const mutationLock = useRef(false);
  const activationCache = useRef(new Map<string, ActivationResult>());
  const saliencyCache = useRef(new Map<string, SaliencyResult>());
  const currentKey = current?.experiment_id ?? '';
  const activationKey = `${currentKey}:${plane}`;
  const active = activation?.key === activationKey ? activation.data : null;
  const saliencyData = saliency?.key === currentKey ? saliency.data : null;
  const conv = plane === 'conv1' || plane === 'conv2' ? plane : null;
  const channel = conv ? channels[conv] : 0;
  const position = PLANES.indexOf(plane);
  const canModify = !!current && !busy && view === 'current';

  const hintRequest = current ? {...current.request,interventions:[]} : null;
  const hintKey = JSON.stringify(hintRequest);
  const hintData = hints?.key===hintKey ? hints.data : null;
  const wantsHints = mode==='explore' && ((!!conv&&featurePower)||(plane==='input'&&erasePower));
  const pristine = !!current && current.request.interventions.length===0;

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    setStartupError('');
    async function load() {
      try {
        const [, models, source] = await Promise.all([api.ready(controller.signal), api.models(controller.signal), api.samples(controller.signal)]);
        if (!source.samples.length) throw new Error('No sample drawings are available.');
        const sample = source.samples.find(row=>row.id===HINT_EXAMPLES.feature.sample_id) ?? source.samples[0];
        const request: ExperimentRequest = { model_id: 'balanced', sample_id: sample.id, background: BACKGROUNDS[sample.label], target_class: sample.class, interventions: [] };
        const result = await api.infer(request);
        if (cancelled) return;
        setCatalog(models); setSamples(source.samples); setDraft(request);
        setBaseline(result); setCurrent(result);
      } catch (problem) { if (!cancelled) setStartupError(errorText(problem)); }
    }
    void load();
    return () => { cancelled = true; controller.abort(); };
  }, [boot]);

  useEffect(() => {
    if (mode !== 'explore' || !current || !['conv1', 'conv2', 'dense'].includes(plane)) return;
    const controller = new AbortController(); let valid = true;
    setViewError('');
    const cached = activationCache.current.get(activationKey);
    if (cached) { setActivation({ key: activationKey, data: cached }); return; }
    api.activations(current.request, plane as Layer, controller.signal).then(data => {
      if (!valid) return;
      activationCache.current.set(activationKey, data);
      if (activationCache.current.size > 32) activationCache.current.delete(activationCache.current.keys().next().value!);
      setActivation({ key: activationKey, data });
    }).catch(problem => { if (valid) setViewError(errorText(problem)); });
    return () => { valid = false; controller.abort(); };
  }, [currentKey, plane, activationKey, retryView, mode]); // Responses are keyed to the committed experiment, not selected channels.

  useEffect(() => {
    if (mode !== 'explore' || !current || !showSaliency || plane !== 'input') return;
    const controller = new AbortController(); let valid = true;
    setViewError('');
    const cached = saliencyCache.current.get(currentKey);
    if (cached) { setSaliency({ key: currentKey, data: cached }); return; }
    api.saliency(current.request, controller.signal).then(data => {
      if (!valid) return;
      saliencyCache.current.set(currentKey, data);
      if (saliencyCache.current.size > 32) saliencyCache.current.delete(saliencyCache.current.keys().next().value!);
      setSaliency({ key: currentKey, data });
    }).catch(problem => { if (valid) setViewError(errorText(problem)); });
    return () => { valid = false; controller.abort(); };
  }, [currentKey, showSaliency, plane, retryView, mode]);

  useEffect(()=>{
    if(!wantsHints || !hintRequest || !current) return;
    const controller=new AbortController();let valid=true;setHintError('');
    const cached=hintCache.current.get(hintKey);
    if(cached){setHints({key:hintKey,data:cached});return;}
    api.hints(hintRequest,controller.signal).then(data=>{
      if(!valid)return;
      if(data.experiment.checkpoint_sha256!==current.checkpoint_sha256 || JSON.stringify(data.experiment.baseline.logits)!==JSON.stringify(current.baseline.logits)) throw new Error('The hint baseline differs from this experiment. Start a new experiment.');
      hintCache.current.set(hintKey,data);
      if(hintCache.current.size>20)hintCache.current.delete(hintCache.current.keys().next().value!);
      setHints({key:hintKey,data});
    }).catch(problem=>{if(valid)setHintError(errorText(problem));});
    return()=>{valid=false;controller.abort();};
  },[wantsHints,hintKey,hintRetry]);

  useEffect(() => {
    function keys(event: KeyboardEvent) {
      if (mode !== 'explore') return;
      const target = event.target as HTMLElement;
      if (target.closest('input, select, textarea, [contenteditable="true"], .is-interactive') || (target.closest('button') && !target.closest('.layer-nav'))) return;
      if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
      event.preventDefault();
      const direction = event.key === 'ArrowRight' ? 1 : -1;
      const navigation = target.closest('.layer-nav');
      if (navigation) {
        const buttons = Array.from(navigation.querySelectorAll('button'));
        const next = Math.max(0, Math.min(4, buttons.indexOf(target.closest('button')!) + direction));
        setPlane(PLANES[next]); buttons[next].focus();
      } else {
        setPlane(previous => PLANES[Math.max(0, Math.min(4, PLANES.indexOf(previous) + direction))]);
      }
    }
    window.addEventListener('keydown', keys);
    return () => window.removeEventListener('keydown', keys);
  }, [mode]);

  async function start(event: FormEvent) {
    event.preventDefault();
    if (mutationLock.current) return;
    mutationLock.current = true; setBusy(true); setError('');
    try {
      const result = await api.infer({ ...draft, interventions: [] });
      setBaseline(result); setCurrent(result);
      setTool('inspect'); setView('current'); setPlane('input');
      activationCache.current.clear(); saliencyCache.current.clear();
      setAnnouncement(`New ${draft.model_id} experiment started with ${draft.sample_id}.`);
    } catch (problem) { setError(errorText(problem)); }
    finally { mutationLock.current = false; setBusy(false); }
  }
  async function loadExample(example: typeof FLIP_EXAMPLES[number]) {
    if(mutationLock.current) return;
    mutationLock.current=true;setBusy(true);setError('');
    try {
      const result=await api.infer(example.request);
      setDraft(example.request);setBaseline(result);setCurrent(result);
      setPlane(example.layer);setChannels(previous=>({...previous,[example.layer]:example.channel}));
      setView('current');setTool('inspect');setShowSaliency(false);
      activationCache.current.clear();saliencyCache.current.clear();
      setAnnouncement('Class-flip example ready. Apply the ablation to measure its effect.');
    }catch(problem){setError(errorText(problem));}
    finally{mutationLock.current=false;setBusy(false);}
  }
  async function loadHintExample(kind:'feature'|'erase') {
    if(mutationLock.current)return;
    mutationLock.current=true;setBusy(true);setError('');
    const request=HINT_EXAMPLES[kind];
    try {
      const result=await api.infer(request);
      setDraft(request);setBaseline(result);setCurrent(result);
      setView('current');setShowSaliency(false);setShape('patch');setFeaturePower(true);setErasePower(true);
      setPlane(kind==='feature'?'conv1':'input');setChannels({conv1:7,conv2:0});
      setTool(kind==='erase'?'drawing_erasure':'inspect');
      activationCache.current.clear();saliencyCache.current.clear();
      setAnnouncement('Hint example ready. No intervention has been applied.');
    }catch(problem){setError(errorText(problem));}
    finally{mutationLock.current=false;setBusy(false);}
  }
  function selectHint(row:ChannelHint) {
    setPlane(row.layer);setChannels(previous=>({...previous,[row.layer]:row.channel}));
  }
  async function apply(operation: Intervention) {
    if (!current || !canModify || mutationLock.current) return;
    mutationLock.current = true; setBusy(true); setError('');
    try {
      const request = withIntervention(current.request, operation);
      const result = await api.infer(request);
      setCurrent(result);
      setAnnouncement(`${operationLabel(operation)} applied. ${result.comparison.classification_flip ? 'The predicted class changed.' : 'The predicted class is unchanged.'}`);
    } catch (problem) { setError(errorText(problem)); }
    finally { mutationLock.current = false; setBusy(false); }
  }
  function reset() {
    if (busy || !baseline) return;
    setCurrent(baseline); setView('current'); setError('');
    setAnnouncement('Baseline restored exactly. Try another experiment.');
  }
  const selectRegion = useCallback((region: Region) => {
    if (tool === 'inspect') return;
    if (tool === 'drawing_erasure') void apply({ type: tool, region });
    else void apply({ type: 'controlled_occlusion', region, replacement_rgb: [128, 128, 128] });
  }, [tool, current, canModify]);
  function manualRegion(event: FormEvent) {
    event.preventDefault();
    if (shape === 'rectangle' && (coordinates.x + coordinates.width > 28 || coordinates.y + coordinates.height > 28)) {
      setError('The selected rectangle extends beyond the image. Keep x + width and y + height within 28 pixels.');
      return;
    }
    selectRegion(shape === 'pixel' ? { pixels: [{ x: coordinates.x, y: coordinates.y }] } : {
      rectangle: { x: Math.min(shape === 'patch' ? 24 : 27, coordinates.x), y: Math.min(shape === 'patch' ? 24 : 27, coordinates.y), width: shape === 'patch' ? 4 : coordinates.width, height: shape === 'patch' ? 4 : coordinates.height },
    });
  }
  function exportState() {
    if (!current) return;
    const payload = { schema_version: 2, experiment: current, baseline_hints: hintData };
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = `featuredive-${current.experiment_id.slice(0, 12)}.json`; anchor.click(); URL.revokeObjectURL(url);
  }

  return <>
    <a className="skip-link" href={mode === 'explore' ? '#explorer' : '#guided-challenges'}>Skip to {mode === 'explore' ? 'network explorer' : 'guided challenges'}</a>
    <header className="game-header">
      <a className="game-brand" href="/" aria-label="FeatureDive home"><span aria-hidden="true">F↘</span>FeatureDive</a>
      <nav className="game-modes" aria-label="Learning mode">
        <button aria-pressed={mode==='explore'} disabled={busy||challengeBusy} onClick={()=>setMode('explore')}>Free exploration</button>
        <button aria-pressed={mode==='challenges'} disabled={busy||challengeBusy} onClick={()=>setMode('challenges')}>Guided challenges</button>
      </nav>
      <span className="connection"><i className={current?'connected':''}/>{current?'Live model':'Connecting'}</span>
    </header>
    <main className="game-main">
      {startupError ? <section className="startup-error" role="alert"><h1>The lab isn’t ready yet.</h1><p>{startupError}</p><button onClick={() => setBoot(value => value + 1)}>Retry connection</button></section> : !current ? <section className="loading-lab" role="status">Loading the models and drawings…</section> : <>
        <div hidden={mode!=='challenges'}><ChallengeHub samples={samples} active={mode==='challenges'} onBusy={setChallengeBusy} /></div>
        <div hidden={mode!=='explore'}>
          <div className="play-heading"><div><span className="eyebrow">THE NEURAL NETWORK PLAYGROUND</span><h1>Can you change its mind?</h1></div><p>Pick a clue. Change one thing. See what happens.</p></div>
          <div className="experiment-picker">
            <div className="hunt-picker" aria-label="Hint examples"><button aria-label="Feature hunt · fish" aria-pressed={current.request.sample_id===HINT_EXAMPLES.feature.sample_id} disabled={busy} onClick={()=>void loadHintExample('feature')}>✦ Feature hunt · fish</button><button aria-label="Pixel hunt · airplane" aria-pressed={current.request.sample_id===HINT_EXAMPLES.erase.sample_id} disabled={busy} onClick={()=>void loadHintExample('erase')}>◈ Pixel hunt · airplane</button></div>
            <form onSubmit={start} aria-label="New experiment settings">
              <label className="drawing-picker"><span className="sr-only">Drawing</span><select aria-label="Drawing" value={draft.sample_id} disabled={busy} onChange={event => { const sample = samples.find(item => item.id === event.target.value)!; setDraft({ ...draft, sample_id: sample.id, target_class: sample.class }); }}>
                {CLASSES.map(name => <optgroup key={name} label={name}>{samples.filter(item => item.class === name).map((sample,index) => <option value={sample.id} key={sample.id}>{name} {String(index+1).padStart(2,'0')} · {sample.id.split(':')[1]}</option>)}</optgroup>)}
              </select></label>
              <button type="submit" disabled={busy}>Start experiment ↗</button>
              <details className="setup-details"><summary>More settings</summary><div>
                <label>Model<select aria-label="Model" value={draft.model_id} disabled={busy} onChange={event=>setDraft({...draft,model_id:event.target.value as ModelID})}><option value="balanced">Balanced</option><option value="shortcut">Shortcut</option></select></label>
                <label>Starting background<select aria-label="Starting background" value={draft.background} disabled={busy} onChange={event=>setDraft({...draft,background:event.target.value as Background})}>{BACKGROUNDS.map(name=><option key={name}>{name}</option>)}</select></label>
                <label>Target class<select aria-label="Target class" value={draft.target_class} disabled={busy} onChange={event=>setDraft({...draft,target_class:event.target.value as ClassName})}>{CLASSES.map(name=><option key={name}>{name}</option>)}</select></label>
                <small>Apply with Start experiment. Balanced uses independent colors; shortcut pairs class and color 90% of the time.</small>
              </div></details>
            </form>
            <details className="example-details"><summary>More class-flip examples</summary><div>{FLIP_EXAMPLES.map(example=><button key={example.name} disabled={busy} onClick={()=>void loadExample(example)}>Load {example.name} example</button>)}</div></details>
          </div>
          {error&&<div className="error-banner" role="alert">{error}<button onClick={()=>setError('')} aria-label="Dismiss error">×</button></div>}
          <section className="playground" id="explorer" tabIndex={-1} aria-label="Network explorer">
            <div className="playground-top"><nav className="layer-nav" aria-label="Network layers">{PLANES.map((name,index)=><button key={name} aria-label={`${['Drawing','Layer 1','Layer 2','Hidden','Answer'][index]} · ${name.toUpperCase()} · ${['Pixels','8 feature maps','16 feature maps','32 neurons','3 classes'][index]}`} onClick={()=>setPlane(name)} aria-current={plane===name?'step':undefined}><span>{['Drawing','Layer 1','Layer 2','Hidden','Answer'][index]}</span><small>{['INPUT','CONV1','CONV2','DENSE','OUTPUT'][index]}</small></button>)}</nav>
              <button className="reset-button" aria-label="Reset & try again" disabled={busy} onClick={reset}><span aria-hidden="true">↺</span> Reset & try again</button>
            </div>
            <div className="play-layout">
              <section className="experiment-stage" aria-label="Experiment stage">
                <div className="stage-heading"><div><h2>{COPY[plane].title}</h2><span>{conv?`Channel ${channel} · ${COPY[plane].tag}`:COPY[plane].tag}</span></div><div className="view-controls" role="group" aria-label="Displayed state"><button aria-pressed={view==='current'} onClick={()=>setView('current')}>Current</button><button aria-pressed={view==='baseline'} onClick={()=>setView('baseline')}>Baseline</button></div></div>
                <PredictionAnswer result={current} busy={busy}/>
                {conv&&<div className="mobile-test-action"><button className="ablate-button" aria-label={`Switch off channel ${channel}`} disabled={!canModify||!active} onClick={()=>void apply({type:'channel_ablation',layer:conv,channel})}>⌁ Switch off channel {channel}</button></div>}
                {plane==='input'&&erasePower&&hintData?.suggested_patch&&<div className="mobile-test-action"><button className="ablate-button" disabled={busy||!pristine||view!=='current'} onClick={()=>void apply({type:'drawing_erasure',region:{rectangle:hintData.suggested_patch!.rectangle}})}>Erase hinted patch</button></div>}
                {viewError&&<div role="alert" className="view-error">{viewError}<button onClick={()=>setRetryView(value=>value+1)}>Retry view</button></div>}
            <div className={`network-stage plane-${plane}`} key={plane}>
              {plane === 'input' && <div className="primary-plane"><PixelCanvas rgb={view === 'current' ? current.input.modified_rgb_chw : current.input.baseline_rgb_chw} label="28 by 28 drawing" grid={grid} interactive={tool !== 'inspect' && canModify} mode={shape} onSelect={selectRegion} support={erasePower&&hintData&&pristine ? hintData.erase_support : undefined} saliency={showSaliency && saliencyData ? view === 'current' ? saliencyData.modified_scores : saliencyData.baseline_scores : undefined} opacity={opacity} />{erasePower&&hintData?.suggested_patch&&pristine&&<span className="hint-drawing-patch" style={{left:`${hintData.suggested_patch.rectangle.x/28*100}%`,top:`${hintData.suggested_patch.rectangle.y/28*100}%`,width:`${4/28*100}%`,height:`${4/28*100}%`}}/>}</div>}
              {conv && active && <div className="primary-plane"><PixelCanvas values={(view === 'current' ? active.modified_values : active.baseline_values as RGBImage)[channel] as Matrix} upper={Math.max(active.baseline_max, active.modified_max)} label={`${conv} channel ${channel} ${view} activation`} /></div>}
              {(conv || plane === 'dense') && !active && <div className="plane-loading" role="status">{viewError ? 'This view is unavailable.' : 'Reading activations…'}</div>}
              {plane === 'dense' && active && <div className="dense-plane"><div className="dense-grid">{(view === 'current' ? active.modified_values : active.baseline_values as number[]).map((value, index) => { const number = value as number; const fraction = number / Math.max(active.baseline_max, active.modified_max, 1e-8); return <div key={index} className="neuron" data-bright={fraction > .45} style={{ '--activity': fraction } as React.CSSProperties} title={`Neuron ${index}: ${number}`}><span>{String(index).padStart(2, '0')}</span><b>{number.toFixed(2)}</b></div>; })}</div></div>}
              {plane === 'output' && <div className="output-plane"><span className="small-label">{view === 'current' ? 'Current' : 'Baseline'} prediction</span><h3>{(view === 'current' ? current.modified : current.baseline).predicted_class}<span aria-hidden="true">↗</span></h3>{CLASSES.map((name, index) => { const prediction = view === 'current' ? current.modified : current.baseline; return <div className="output-class" key={name}><span>{name}</span><strong>{(prediction.probabilities[index] * 100).toFixed(2)}%</strong><div style={{ width: `${100 * prediction.probabilities[index]}%` }} /><small>logit {prediction.logits[index].toFixed(4)}</small></div>; })}<p>Uncalibrated model probabilities.</p></div>}
            </div>

                <Results result={current} onExport={exportState}/>
                <p className="stage-prompt">{busy?'Testing the model…':view==='baseline'?'Original view. Switch to Current to experiment.':conv?'Choose a channel. Switch it off. Watch the answer.':plane==='input'?(tool==='inspect'?'Pick an eraser or a color to experiment.':'Click the drawing to test your brush.'):'Go back to Drawing or a layer to change the answer.'}</p>
                {conv&&active&&<><div className="channel-grid" aria-label="Feature channels">{((view==='current'?active.modified_values:active.baseline_values) as RGBImage).map((map,index)=><button key={index} aria-label={`Select channel ${index}`} aria-pressed={index===channel} onClick={()=>setChannels({...channels,[conv]:index})}><PixelCanvas values={map} upper={Math.max(active.baseline_max,active.modified_max)} label={`${conv} channel ${index} thumbnail`}/><span>{String(index).padStart(2,'0')}</span></button>)}</div><details className="stage-details"><summary>Raw values · selected channel</summary><p>Shared raw scale: 0 → {Math.max(active.baseline_max,active.modified_max).toFixed(4)}. Activity ≠ influence.</p><pre>{((view==='current'?active.modified_values:active.baseline_values) as RGBImage)[channel].map(row=>row.map(value=>value.toFixed(4)).join(' ')).join('\n')}</pre></details></>}
                {plane==='input'&&erasePower&&hintData&&!pristine&&<p className="micro-copy">Eraser Vision overlay is paused on the changed drawing. Reset baseline to see the original clue.</p>}
                {plane==='input'&&showSaliency&&<p className="micro-copy">{saliencyData?'Saliency: local sensitivity; overlay contrast scaled to the map maximum.':'Computing target-logit gradients…'}</p>}
                <div className="stage-meta"><span>{current.request.sample_id} · {current.request.model_id} · target {current.request.target_class}</span><span>{position+1} / 5 · Use ← → to travel</span></div>
              </section>
              <aside className="tool-dock" aria-label="Experiment tools">
                <span className="eyebrow">{conv?'SWITCH A FEATURE OFF':plane==='input'?'REMIX THE DRAWING':'FOLLOW THE SIGNAL'}</span>
                <h2>{conv?'Pull the plug.':plane==='input'?'Try a tiny edit.':'Take a look.'}</h2>
                {conv&&<>
                  <div className="channel-inspector"><span>Selected channel <b>{channel}</b></span><p>Mean activity: <b>{active?(view==='current'?active.modified_mean_by_channel:active.baseline_mean_by_channel)[channel].toFixed(4):'…'}</b></p></div>
                  <button className="ablate-button desktop-test-action" aria-label={`Switch off channel ${channel}`} disabled={!canModify||!active} onClick={()=>void apply({type:'channel_ablation',layer:conv,channel})}>⌁ Switch off channel {channel}</button>
                  <p className="micro-copy">Zeros one whole channel. Try another, or reset.</p>
                  <button className="power-button" aria-label="Feature Radar" aria-pressed={featurePower} onClick={()=>setFeaturePower(value=>!value)}><span>✦ Feature Radar</span><b>{featurePower?'ON':'OFF'}</b></button>
                </>}
                {plane==='input'&&<>
                  <div className="brush-buttons"><button aria-label="Pixel Eraser" aria-pressed={tool==='drawing_erasure'} onClick={()=>setTool(tool==='drawing_erasure'?'inspect':'drawing_erasure')}>⌫ Erase strokes</button><button aria-label="Controlled Occlusion" aria-pressed={tool==='controlled_occlusion'} onClick={()=>setTool(tool==='controlled_occlusion'?'inspect':'controlled_occlusion')}>▨ Cover with gray</button></div>
                  {tool!=='inspect'&&<div className="brush-settings"><label>Brush<select aria-label="Selection shape" value={shape} onChange={event=>setShape(event.target.value as typeof shape)}><option value="pixel">Single pixel</option><option value="patch">4 × 4 patch</option><option value="rectangle">Drag rectangle</option></select></label><details><summary>Enter exact coordinates</summary><form className="coordinate-form" onSubmit={manualRegion}>{(['x','y',...(shape==='rectangle'?['width','height']:[])] as const).map(name=><label key={name}>{name}<input aria-label={`Region ${name}`} type="number" min={name==='x'||name==='y'?0:1} max={name==='x'||name==='y'?27:28} value={coordinates[name as keyof typeof coordinates]} onChange={event=>setCoordinates({...coordinates,[name]:Number(event.target.value)})} required/></label>)}<button disabled={!canModify}>Apply region</button></form></details></div>}
                  <div className="background-swatches" aria-label="Background colors">{BACKGROUNDS.map(name=><button key={name} style={{'--swatch':COLOR_CSS[name]} as React.CSSProperties} disabled={!canModify||current.input.modified_background===name} aria-label={`Change background to ${name}`} aria-pressed={current.input.modified_background===name} onClick={()=>void apply({type:'background_swap',background:name})}><i/>{name}</button>)}</div>
                  <button className="power-button" aria-label="Eraser Vision" aria-pressed={erasePower} onClick={()=>{setErasePower(value=>!value);setShowSaliency(false);setTool('drawing_erasure');}}><span>◈ Eraser Vision</span><b>{erasePower?'ON':'OFF'}</b></button>
                </>}
                {wantsHints&&(hintData?<HintPanel data={hintData} kind={conv?'feature':'erase'} selected={conv?{layer:conv,channel}:null} pristine={pristine&&view==='current'} busy={busy} onChannel={selectHint} onErase={()=>{if(hintData.suggested_patch&&pristine&&view==='current')void apply({type:'drawing_erasure',region:{rectangle:hintData.suggested_patch.rectangle}});}}/>:<div className="hint-loading" role={hintError?'alert':'status'}>{hintError||'Finding a clue…'}{hintError&&<button onClick={()=>setHintRetry(value=>value+1)}>Retry power-up</button>}</div>)}
                {plane==='input'&&<details className="extra-tools"><summary>More drawing tools</summary><button aria-pressed={showSaliency} onClick={()=>{setShowSaliency(value=>!value);setErasePower(false);}}>◎ Saliency Scope</button>{showSaliency&&<label>Overlay opacity<input aria-label="Saliency opacity" type="range" min="0" max="1" step=".05" value={opacity} onChange={event=>setOpacity(Number(event.target.value))}/></label>}<label className="grid-toggle"><input type="checkbox" checked={grid} onChange={event=>setGrid(event.target.checked)}/>Pixel grid</label><small>Erase restores strokes to background. Cover replaces all pixels with gray. Gray is outside training colors.</small></details>}
                {!conv&&plane!=='input'&&<><p className="dock-prompt">{plane==='dense'?'These 32 responses feed the final decision. Bright means active.':'Three scores compete. The highest score wins.'}</p><button className="ablate-button" onClick={()=>setPlane('input')}>← Edit the drawing</button><button className="power-button" onClick={()=>setPlane('conv1')}>Inspect a feature →</button></>}
                <details className="extra-tools"><summary>Why try this?</summary><p>{COPY[plane].note}</p><p>Clues are estimates. The before/after result is a real model test.</p></details>
              </aside>
            </div>
          </section>
        </div>
      </>}
      <footer className="site-footer"><span>FeatureDive · Real model, real experiments{catalog?` · ${catalog.model_version}`:''}</span><span>Illustrative training drawings · Quick, Draw! · <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer">CC BY 4.0</a></span></footer>
    </main>
    <div className="sr-only" aria-live="polite">{announcement}</div>
  </>;
}
