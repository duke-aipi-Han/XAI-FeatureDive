import { useCallback, useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { api } from './api';
import PixelCanvas from './components/PixelCanvas';
import Results from './components/Results';
import ChallengeHub from './challenges/ChallengeHub';
import { FLIP_EXAMPLES, HINT_EXAMPLES } from './examples';
import HintPanel from './components/HintPanel';
import { branchHistory, operationLabel, withIntervention } from './state';
import { BACKGROUNDS, CLASSES, PLANES } from './types';
import type { ActivationResult, Background, ClassName, ExperimentRequest, HistoryEntry, Intervention, Layer, Matrix, ModelCatalog, ModelID, Plane, Region, RGBImage, SaliencyResult, Sample, HintResult, ChannelHint } from './types';

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
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [cursor, setCursor] = useState(0);
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
  const [presetHint,setPresetHint] = useState('');
  const [announcement, setAnnouncement] = useState('');
  const mutationLock = useRef(false);
  const activationCache = useRef(new Map<string, ActivationResult>());
  const saliencyCache = useRef(new Map<string, SaliencyResult>());
  const current = history[cursor]?.result;
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
        setHistory([{ label: 'Original baseline', result }]); setCursor(0);
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
    setPresetHint('');
    if (mutationLock.current) return;
    mutationLock.current = true; setBusy(true); setError('');
    try {
      const result = await api.infer({ ...draft, interventions: [] });
      setHistory([{ label: 'Original baseline', result }]); setCursor(0);
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
      setDraft(example.request);setHistory([{label:'Original baseline',result}]);setCursor(0);
      setPlane(example.layer);setChannels(previous=>({...previous,[example.layer]:example.channel}));
      setView('current');setTool('inspect');setShowSaliency(false);
      activationCache.current.clear();saliencyCache.current.clear();
      setPresetHint(`Ready: ${example.request.sample_id}, balanced model, ${example.request.background} background. Press “Ablate channel ${example.channel}” to switch this feature off and test whether the answer changes.`);
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
      setDraft(request);setHistory([{label:'Original baseline',result}]);setCursor(0);
      setView('current');setShowSaliency(false);setShape('patch');setFeaturePower(true);setErasePower(true);
      setPlane(kind==='feature'?'conv1':'input');setChannels({conv1:7,conv2:0});
      setTool(kind==='erase'?'drawing_erasure':'inspect');
      activationCache.current.clear();saliencyCache.current.clear();
      setPresetHint(kind==='feature'?'Feature hunt: follow the support clue, then ablate the selected channel.':'Pixel hunt: use Eraser Vision, predict what will happen, then erase the hinted patch.');
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
      const next = branchHistory(history, cursor, { label: operationLabel(operation), result });
      setHistory(next); setCursor(next.length - 1); setPresetHint('');
      setAnnouncement(`${operationLabel(operation)} applied. ${result.comparison.classification_flip ? 'The predicted class changed.' : 'The predicted class is unchanged.'}`);
    } catch (problem) { setError(errorText(problem)); }
    finally { mutationLock.current = false; setBusy(false); }
  }
  function restore(index: number) {
    if (busy || index < 0 || index >= history.length) return;
    setCursor(index); setView('current'); setError('');
    setAnnouncement(index === 0 ? 'Baseline restored exactly.' : `Restored ${history[index].label}.`);
  }
  const selectRegion = useCallback((region: Region) => {
    if (tool === 'inspect') return;
    if (tool === 'drawing_erasure') void apply({ type: tool, region });
    else void apply({ type: 'controlled_occlusion', region, replacement_rgb: [128, 128, 128] });
  }, [tool, current, canModify, history, cursor]);
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
    const payload = { schema_version: 1, experiment: current, baseline_hints: hintData, history: history.map(entry => ({ label: entry.label, experiment_id: entry.result.experiment_id, request: entry.result.request })) };
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = `featuredive-${current.experiment_id.slice(0, 12)}.json`; anchor.click(); URL.revokeObjectURL(url);
  }

  return <>
    <a className="skip-link" href={mode === 'explore' ? '#explorer' : '#guided-challenges'}>Skip to {mode === 'explore' ? 'network explorer' : 'guided challenges'}</a>
    <header className="site-header"><a className="brand" href="/" aria-label="FeatureDive home"><span className="brand-icon" aria-hidden="true">F<span /></span><span>Feature<span>Dive</span><small>THE NEURAL NETWORK LAB</small></span></a>
      <div className="header-status"><span className={`status-dot ${current ? 'connected' : ''}`} /><span>{current ? 'Real model · CPU inference' : 'Connecting to the lab'}</span><span className="version-tag">{catalog?.model_version ?? 'PHASE 4'}</span></div>
    </header>
    <main className={mode==='challenges'?'challenge-mode':undefined}>
      <section className="intro"><div><span className="eyebrow">EXPLORE THE NETWORK. CHALLENGE THE EXPLANATION.</span><h1>Look closer. <em>Change one thing.</em></h1><p>Follow a drawing through a real neural network. Inspect its responses, try an intervention, and compare the evidence.</p></div><div className="intro-index" aria-hidden="true"><span>OBSERVE</span><span>↘ INTERVENE</span><span>↘ MEASURE</span></div></section>
      {startupError ? <section className="startup-error" role="alert"><h2>The lab isn’t ready yet.</h2><p>{startupError}</p><p>Check that the local API is running and its model artifacts are available.</p><button onClick={() => setBoot(value => value + 1)}>Retry connection</button></section> : !current ? <section className="loading-lab" role="status"><span className="loading-block" />Loading the models and sample catalog…</section> : <>
        <nav className="mode-switch" aria-label="Learning mode"><button aria-pressed={mode==='explore'} disabled={busy||challengeBusy} onClick={()=>setMode('explore')}>Free exploration</button><button aria-pressed={mode==='challenges'} disabled={busy||challengeBusy} onClick={()=>setMode('challenges')}>Guided challenges</button><span>Explore freely, or record predictions before you test.</span></nav>
        <div hidden={mode!=='challenges'}><ChallengeHub samples={samples} active={mode==='challenges'} onBusy={setChallengeBusy} /></div>
        <div hidden={mode!=='explore'}>
        <section className="flip-examples powerup-examples" aria-label="Hint examples"><div><h2>Power-up your investigation.</h2><p>Start with a clue. Make a guess. Test the real model.</p></div><button disabled={busy} onClick={()=>void loadHintExample('feature')}>Feature hunt · fish</button><button disabled={busy} onClick={()=>void loadHintExample('erase')}>Pixel hunt · airplane</button></section>
        <details className="more-examples"><summary>More class-flip examples</summary><div>{FLIP_EXAMPLES.map(example=><button key={example.name} disabled={busy} onClick={()=>void loadExample(example)}>Load {example.name} example</button>)}</div></details>
        <form className="experiment-bar" onSubmit={start} aria-label="New experiment settings">
          <span className="experiment-label"><span className="eyebrow">YOUR EXPERIMENT</span><span>24 training examples · illustrative</span></span>
          <label>Drawing<select aria-label="Drawing" value={draft.sample_id} disabled={busy} onChange={event => { const sample = samples.find(item => item.id === event.target.value)!; setDraft({ ...draft, sample_id: sample.id, target_class: sample.class }); }}>
            {CLASSES.map(name => <optgroup key={name} label={name}>{samples.filter(item => item.class === name).map((sample, index) => <option value={sample.id} key={sample.id}>{name} {String(index + 1).padStart(2, '0')} · {sample.id.split(':')[1]}</option>)}</optgroup>)}
          </select></label>
          <details className="advanced-experiment"><summary>More settings</summary><div>
          <label>Model<select aria-label="Model" value={draft.model_id} disabled={busy} onChange={event => setDraft({ ...draft, model_id: event.target.value as ModelID })}><option value="balanced">Balanced</option><option value="shortcut">Shortcut</option></select></label>
          <label>Starting background<select aria-label="Starting background" value={draft.background} disabled={busy} onChange={event => setDraft({ ...draft, background: event.target.value as Background })}>{BACKGROUNDS.map(name => <option key={name}>{name}</option>)}</select></label>
          <label>Target class<select aria-label="Target class" value={draft.target_class} disabled={busy} onChange={event => setDraft({ ...draft, target_class: event.target.value as ClassName })}>{CLASSES.map(name => <option key={name}>{name}</option>)}</select></label>
          </div></details>
          <button className="primary-button" disabled={busy} type="submit">Start experiment <span aria-hidden="true">↗</span></button>
        </form>
        <div className="session-note"><span>Active: <b>{current.request.sample_id}</b> · {current.request.model_id} · target {current.request.target_class} · baseline {current.request.background}</span><span>Settings above take effect when you start a new experiment.</span></div>
        {error && <div className="error-banner" role="alert">{error}<button onClick={() => setError('')} aria-label="Dismiss error">×</button></div>}
        {presetHint&&<p className="preset-hint" role="status">{presetHint}</p>}
        <nav className="layer-nav" aria-label="Network layers">{PLANES.map((name, index) => <button key={name} onClick={() => setPlane(name)} aria-current={plane === name ? 'step' : undefined} className={plane === name ? 'selected' : ''}><span>{String(index + 1).padStart(2, '0')}</span>{name.toUpperCase()}<small>{['Pixels', '8 feature maps', '16 feature maps', '32 neurons', '3 classes'][index]}</small></button>)}</nav>
        <div className="lab-grid">
          <aside className="tools-panel" aria-label="Experiment tools"><div className="panel-heading"><span className="eyebrow">01 / INVESTIGATE</span><span aria-hidden="true">⌁</span></div><h2>Your toolkit</h2>
            <p className="tool-intro">{COPY[plane].note}</p>
            <p className="powerup-label">✦ POWER-UPS / CLUES</p>
            <button className={`tool-card powerup ${conv&&featurePower?'active':''}`} aria-label="Feature Radar" aria-pressed={!!conv&&featurePower} onClick={()=>{if(!conv){setPlane('conv1');setFeaturePower(true);}else setFeaturePower(value=>!value);}}><span className="tool-icon">✦</span><span><b>Feature Radar</b><small>Find likely decision support</small></span><span>{conv&&featurePower?'ON':'+'}</span></button>
            <button className={`tool-card powerup ${plane==='input'&&erasePower?'active':''}`} aria-label="Eraser Vision" aria-pressed={plane==='input'&&erasePower} onClick={()=>{if(plane!=='input'){setPlane('input');setErasePower(true);}else setErasePower(value=>!value);setShowSaliency(false);setTool('drawing_erasure');}}><span className="tool-icon">◈</span><span><b>Eraser Vision</b><small>See strokes worth testing</small></span><span>{plane==='input'&&erasePower?'ON':'+'}</span></button>
            {plane === 'input' && <>
              <button className={`tool-card ${showSaliency ? 'active' : ''}`} onClick={() => {setShowSaliency(value => !value);setErasePower(false);}} aria-pressed={showSaliency}><span className="tool-icon">◎</span><span><b>Saliency Scope</b><small>See local sensitivity</small></span><span>{showSaliency ? 'ON' : '+'}</span></button>
              {showSaliency && <div className="tool-setting"><label>Overlay opacity <b>{Math.round(opacity * 100)}%</b><input aria-label="Saliency opacity" type="range" min="0" max="1" step=".05" value={opacity} onChange={event => setOpacity(Number(event.target.value))} /></label><p>Sum of absolute RGB gradients of the target logit. Contrast scaled per image; this is local sensitivity.</p></div>}
              <div className="tool-separator" />
              <button className={`tool-card ${tool === 'drawing_erasure' ? 'active' : ''}`} onClick={() => setTool(tool === 'drawing_erasure' ? 'inspect' : 'drawing_erasure')} aria-pressed={tool === 'drawing_erasure'}><span className="tool-icon">⌫</span><span><b>Pixel Eraser</b><small>Restore strokes to background</small></span><span>{tool === 'drawing_erasure' ? 'ON' : '+'}</span></button>
              <button className={`tool-card ${tool === 'controlled_occlusion' ? 'active' : ''}`} onClick={() => setTool(tool === 'controlled_occlusion' ? 'inspect' : 'controlled_occlusion')} aria-pressed={tool === 'controlled_occlusion'}><span className="tool-icon">▨</span><span><b>Controlled Occlusion</b><small>Replace the full region with gray</small></span><span>{tool === 'controlled_occlusion' ? 'ON' : '+'}</span></button>
              {tool !== 'inspect' && <div className="tool-setting"><label>Selection<select aria-label="Selection shape" value={shape} onChange={event => setShape(event.target.value as typeof shape)}><option value="pixel">Single pixel</option><option value="patch">4 × 4 patch</option><option value="rectangle">Drag rectangle</option></select></label><p>{view === 'baseline' ? 'Switch to Current to apply an intervention.' : 'Click or drag on the image to apply. Keyboard: arrows and Enter; Shift + arrows for a rectangle.'}</p>
                <details><summary>Enter exact coordinates</summary><form className="coordinate-form" onSubmit={manualRegion}>{(['x', 'y', ...(shape === 'rectangle' ? ['width', 'height'] : [])] as const).map(name => <label key={name}>{name}<input aria-label={`Region ${name}`} type="number" min={name === 'x' || name === 'y' ? 0 : 1} max={name === 'x' || name === 'y' ? 27 : 28} value={coordinates[name as keyof typeof coordinates]} onChange={event => setCoordinates({ ...coordinates, [name]: Number(event.target.value) })} required /></label>)}<button disabled={!canModify}>Apply region</button></form></details>
              </div>}
              <div className="tool-separator" /><h3>Background Switcher</h3><p className="micro-copy">Keep the drawing fixed. Change its context.</p><div className="background-swatches">{BACKGROUNDS.map(name => <button key={name} className={current.input.modified_background === name ? 'active' : ''} style={{ '--swatch': COLOR_CSS[name] } as React.CSSProperties} disabled={!canModify || current.input.modified_background === name} onClick={() => void apply({ type: 'background_swap', background: name })} aria-label={`Change background to ${name}`} aria-pressed={current.input.modified_background === name}><span />{name}{current.input.modified_background === name && <b>✓</b>}</button>)}</div><p className="micro-copy">Gray was not used for training. Earlier fixed-gray occlusions stay fixed.</p>
            </>}
            {conv && <><div className="tool-card active static"><span className="tool-icon">▦</span><span><b>Activation Scanner</b><small>Inspect real post-ReLU values</small></span></div><div className="channel-inspector"><span className="small-label">Selected channel</span><strong>{String(channel).padStart(2, '0')}</strong><p>Mean activity: <b>{active ? (view === 'current' ? active.modified_mean_by_channel : active.baseline_mean_by_channel)[channel].toFixed(4) : '…'}</b></p></div><button className="ablate-button" aria-label={`Ablate channel ${channel}`} disabled={!canModify || !active} onClick={() => void apply({ type: 'channel_ablation', layer: conv, channel })}><span aria-hidden="true">⌁</span> Ablate channel {channel}</button><p className="micro-copy">Zeros this entire channel after ReLU, before pooling. Ablating another channel replaces the active ablation.</p>{current.input.active_channel_ablation && <div className="active-intervention">Active: {current.input.active_channel_ablation[0].toUpperCase()} / {current.input.active_channel_ablation[1]}</div>}</>}
            {plane === 'dense' && <><div className="tool-card active static"><span className="tool-icon">▦</span><span><b>Activation Scanner</b><small>Inspect the 32 hidden values</small></span></div><p className="micro-copy">Dense-channel intervention is reserved for later work. These values show activity, without semantic labels.</p></>}
            {plane === 'output' && <div className="explanation-card"><b>Read beyond the label.</b><p>A class can stay the same while its score changes. Compare probabilities, logits, and the selected target’s margin.</p></div>}
            <details className="model-explanation"><summary>Balanced vs. shortcut?</summary><p><b>Balanced:</b> background colors were assigned independently of class.</p><p><b>Shortcut:</b> cats usually appeared on green, fish on blue, and airplanes on yellow (90%).</p><p>These names describe training conditions. Balanced is not a guarantee of fairness, and color reliance must be measured.</p></details>
          </aside>
          <section className="explorer-panel" id="explorer" tabIndex={-1} aria-label="Network explorer"><div className="panel-heading"><span className="eyebrow">02 / EXPLORE</span><span className="tensor-tag">{COPY[plane].tag}</span></div><div className="explorer-title"><div><h2>{COPY[plane].title}</h2><p>{COPY[plane].subtitle}</p></div><span className="plane-badge">{position + 1}/5</span></div>
            <div className="view-controls"><div role="group" aria-label="Displayed state"><button className={view === 'current' ? 'active' : ''} onClick={() => setView('current')} aria-pressed={view === 'current'}>Current</button><button className={view === 'baseline' ? 'active' : ''} onClick={() => setView('baseline')} aria-pressed={view === 'baseline'}>Baseline</button></div>{plane === 'input' && <label className="grid-toggle"><input type="checkbox" checked={grid} onChange={event => setGrid(event.target.checked)} />Pixel grid</label>}<span className="view-state">{busy ? 'Computing…' : view === 'current' ? 'Current input & responses' : 'Original baseline view'}</span></div>
            {viewError && <div role="alert" className="view-error">{viewError}<button onClick={() => setRetryView(value => value + 1)}>Retry view</button></div>}
            {wantsHints&&(hintData?<HintPanel data={hintData} kind={conv?'feature':'erase'} selected={conv?{layer:conv,channel}:null} pristine={pristine&&view==='current'} busy={busy} onChannel={selectHint} onErase={()=>{if(hintData.suggested_patch&&pristine)void apply({type:'drawing_erasure',region:{rectangle:hintData.suggested_patch.rectangle}});}}/>:<div className="hint-loading" role={hintError?'alert':'status'}>{hintError||'Reading baseline clues…'}{hintError&&<button onClick={()=>setHintRetry(value=>value+1)}>Retry power-up</button>}</div>)}
            <div className={`network-stage plane-${plane}`} key={plane}>
              <div className="depth-stack" aria-hidden="true">{[2, 1, 0].map((offset) => <span key={offset} className="depth-plane" style={{ '--depth': offset } as React.CSSProperties} />)}</div>
              {plane === 'input' && <div className="primary-plane"><PixelCanvas rgb={view === 'current' ? current.input.modified_rgb_chw : current.input.baseline_rgb_chw} label="28 by 28 drawing" grid={grid} interactive={tool !== 'inspect' && canModify} mode={shape} onSelect={selectRegion} support={erasePower&&hintData&&pristine ? hintData.erase_support : undefined} saliency={showSaliency && saliencyData ? view === 'current' ? saliencyData.modified_scores : saliencyData.baseline_scores : undefined} opacity={opacity} />{erasePower&&hintData?.suggested_patch&&pristine&&<span className="hint-drawing-patch" style={{left:`${hintData.suggested_patch.rectangle.x/28*100}%`,top:`${hintData.suggested_patch.rectangle.y/28*100}%`,width:`${4/28*100}%`,height:`${4/28*100}%`}}/>}</div>}
              {conv && active && <div className="primary-plane"><PixelCanvas values={(view === 'current' ? active.modified_values : active.baseline_values as RGBImage)[channel] as Matrix} upper={Math.max(active.baseline_max, active.modified_max)} label={`${conv} channel ${channel} ${view} activation`} /></div>}
              {(conv || plane === 'dense') && !active && <div className="plane-loading" role="status">{viewError ? 'This view is unavailable.' : 'Reading activations…'}</div>}
              {plane === 'dense' && active && <div className="dense-plane"><div className="dense-grid">{(view === 'current' ? active.modified_values : active.baseline_values as number[]).map((value, index) => { const number = value as number; const fraction = number / Math.max(active.baseline_max, active.modified_max, 1e-8); return <div key={index} className="neuron" data-bright={fraction > .45} style={{ '--activity': fraction } as React.CSSProperties} title={`Neuron ${index}: ${number}`}><span>{String(index).padStart(2, '0')}</span><b>{number.toFixed(2)}</b></div>; })}</div></div>}
              {plane === 'output' && <div className="output-plane"><span className="small-label">{view === 'current' ? 'Current' : 'Baseline'} prediction</span><h3>{(view === 'current' ? current.modified : current.baseline).predicted_class}<span aria-hidden="true">↗</span></h3>{CLASSES.map((name, index) => { const prediction = view === 'current' ? current.modified : current.baseline; return <div className="output-class" key={name}><span>{name}</span><strong>{(prediction.probabilities[index] * 100).toFixed(2)}%</strong><div style={{ width: `${100 * prediction.probabilities[index]}%` }} /><small>logit {prediction.logits[index].toFixed(4)}</small></div>; })}<p>Uncalibrated softmax probabilities.<br />Exact raw scores appear in the comparison panel.</p></div>}
            </div>
            <div className="stage-caption"><span>{plane === 'input' ? `${current.request.sample_id} · ${view === 'current' ? current.input.modified_background : current.input.baseline_background} background` : conv ? `CHANNEL ${String(channel).padStart(2, '0')} / ${conv === 'conv1' ? '08' : '16'}` : plane === 'dense' ? '32 POST-RELU VALUES' : 'CAT / FISH / AIRPLANE'}</span><span>{plane === 'input' ? 'Nearest-neighbor pixels' : 'Real forward-pass values'}</span></div>
            {plane==='input'&&erasePower&&hintData&&!pristine&&<p className="hint-original-note">Eraser Vision overlay is paused on the changed drawing. Reset baseline to see the original clue.</p>}
            {plane === 'input' && showSaliency && <p className="scale-note">{saliencyData ? 'Saliency: raw sum |RGB logit gradients|; overlay contrast scaled to the map maximum.' : viewError ? 'Saliency unavailable.' : 'Computing target-logit gradients…'}</p>}
            {conv && active && <><div className="activation-legend"><span className="gradient-swatch" /><span>Shared raw scale: 0 → {Math.max(active.baseline_max, active.modified_max).toFixed(4)}</span><span>Baseline + current, all channels</span></div><div className="channel-heading"><h3>Choose a feature-map channel</h3><span>Activity ≠ influence</span></div><div className={`channel-grid ${conv}`}>{((view === 'current' ? active.modified_values : active.baseline_values) as RGBImage).map((map, index) => <button key={index} className={index === channel ? 'selected' : ''} aria-label={`Select channel ${index}`} aria-pressed={index === channel} onClick={() => setChannels({ ...channels, [conv]: index })}><PixelCanvas values={map} upper={Math.max(active.baseline_max, active.modified_max)} label={`${conv} channel ${index} thumbnail`} /><span><b>{String(index).padStart(2, '0')}</b><small>{(view === 'current' ? active.modified_mean_by_channel : active.baseline_mean_by_channel)[index].toFixed(3)}</small></span></button>)}</div><details className="raw-activation"><summary>Raw values · selected channel</summary><pre>{((view === 'current' ? active.modified_values : active.baseline_values) as RGBImage)[channel].map(row => row.map(value => value.toFixed(4)).join(' ')).join('\n')}</pre><p>Rounded to four decimals here. Export JSON for complete experiment metadata; this view reads raw API activations.</p></details></>}
            {plane === 'dense' && active && <p className="scale-note">Display intensity: 0 → {Math.max(active.baseline_max, active.modified_max).toFixed(4)}, shared across baseline/current. Numbers show actual values rounded to two decimals.</p>}
            <div className="plane-explanation"><span aria-hidden="true">i</span><p>{COPY[plane].note}</p></div><div className="layer-footer"><button disabled={position === 0} onClick={() => setPlane(PLANES[position - 1])}>← Previous layer</button><span>Use ← → to travel</span><button disabled={position === 4} onClick={() => setPlane(PLANES[position + 1])}>Next layer →</button></div>
          </section>
          <Results result={current} history={history} cursor={cursor} busy={busy} onRestore={restore} onReset={() => restore(0)} onExport={exportState} />
        </div>
        </div>
      </>}
      <footer className="site-footer"><span>FEATUREDIVE <span>·</span> A Responsible AI learning lab</span><span>Activation ≠ attribution ≠ intervention evidence.</span><span>Quick, Draw! by Google · <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer">CC BY 4.0</a></span></footer>
    </main>
    <div className="sr-only" aria-live="polite">{announcement}</div>
  </>;
}
