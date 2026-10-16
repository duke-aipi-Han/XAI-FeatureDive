import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import PixelCanvas from '../components/PixelCanvas';
import { signed } from '../components/Results';
import { BACKGROUNDS, CLASSES } from '../types';
import type { ActivationResult, ExperimentRequest, ExperimentResult, Intervention, SaliencyResult, Sample } from '../types';
import { MeasuredComparison } from './Common';
import { anonymousMapping, CHALLENGES, downloadRecord } from './state';
import type { ChallengeID } from './state';
import { quickFeedback, verifyQuickResult } from './quick';
import type { Guess, QuickRecord } from './quick';

type Observation = { key: string; baselines: ExperimentResult[]; saliency?: SaliencyResult; activation?: ActivationResult };
const freshMapping = () => anonymousMapping((crypto.getRandomValues(new Uint8Array(1))[0] & 1) === 1);
const PATCHES = [{x:8,y:12},{x:20,y:12}];
const QUESTIONS = { evidence: 'Which spot matters more?', hidden: 'Which feature matters more?', shortcut: 'Which model follows the background?' };
const LABELS = { evidence: ['Spot A','Spot B'], hidden: ['Feature A','Feature B'], shortcut: ['Model A','Model B'] };
export default function QuickRound({id,samples,active,onBusy,onComplete,onNext}: {
  id: ChallengeID; samples: Sample[]; active: boolean; onBusy: (busy:boolean)=>void;
  onComplete: (record:QuickRecord)=>void; onNext: ()=>void;
}) {
  const definition = CHALLENGES.find(item=>item.id===id)!;
  const sample = samples.find(row=>row.id===definition.sample) ?? samples[0];
  const [mapping,setMapping] = useState(freshMapping);
  const [attempt,setAttempt] = useState(0);
  const [observation,setObservation] = useState<Observation|null>(null);
  const [hypothesis,setHypothesis] = useState<QuickRecord['hypothesis']|null>(null);
  const [record,setRecord] = useState<QuickRecord|null>(null);
  const [busy,setBusy] = useState(false);
  const [error,setError] = useState('');
  const [retry,setRetry] = useState(0);
  const lock = useRef(false);
  const heading = useRef<HTMLHeadingElement>(null);
  const key = JSON.stringify([id,sample.id,mapping,attempt]);
  const data = observation?.key===key ? observation : null;
  const request: ExperimentRequest = { model_id:'balanced',sample_id:sample.id,target_class:sample.class,background:BACKGROUNDS[sample.label],interventions:[] };
  useEffect(()=>{
    if(active && data) { heading.current?.focus({preventScroll:true});heading.current?.scrollIntoView({block:'start'}); }
  },[active,!!data,!!record]);
  useEffect(()=>{if(active) onBusy(busy);},[active,busy,onBusy]);
  useEffect(()=>{
    if(!active || observation?.key===key) return;
    let valid=true; const controller=new AbortController(); setBusy(true); setError('');
    const load: Promise<Observation> = id==='evidence'
      ? api.saliency(request,controller.signal).then(value=>({key,baselines:[value.experiment],saliency:value}))
      : id==='hidden' ? api.activations(request,'conv2',controller.signal).then(value=>({key,baselines:[value.experiment],activation:value}))
      : Promise.all([api.infer({...request,model_id:mapping.A}),api.infer({...request,model_id:mapping.B})]).then(baselines=>{
        if(baselines[0].source_bitmap_sha256!==baselines[1].source_bitmap_sha256 || JSON.stringify(baselines[0].input.source_alpha)!==JSON.stringify(baselines[1].input.source_alpha)) throw new Error('The models need the same drawing.');
        return {key,baselines};
      });
    load.then(value=>{if(valid)setObservation(value);}).catch(problem=>{if(valid)setError(problem instanceof Error?problem.message:'Could not load this round.');}).finally(()=>{if(valid)setBusy(false);});
    return ()=>{valid=false;controller.abort();setBusy(false);};
  },[active,key,retry]);
  function predict(choice:Guess) {
    if(busy || hypothesis || !data) return;
    setHypothesis(Object.freeze({choice,recorded_at:new Date().toISOString()}));
  }
  async function test() {
    if(!data || !hypothesis || record || busy || lock.current) return;
    lock.current=true;setBusy(true);setError('');const requested_at=new Date().toISOString();
    try {
      const operations: Intervention[] = id==='evidence'
        ? PATCHES.map(patch=>({type:'controlled_occlusion',region:{rectangle:{...patch,width:4,height:4}},replacement_rgb:[128,128,128]}))
        : [{type:'channel_ablation',layer:'conv2',channel:0},{type:'channel_ablation',layer:'conv2',channel:1}];
      const results = id==='shortcut'
        ? await Promise.all((['A','B'] as const).flatMap(alias=>BACKGROUNDS.filter(color=>color!=='gray'&&color!==request.background).map(background=>api.infer({...request,model_id:mapping[alias],interventions:[{type:'background_swap',background}]}))))
        : await Promise.all(operations.map(operation=>api.infer({...request,interventions:[operation]})));
      for(const result of results) verifyQuickResult(result,data.baselines.find(row=>row.request.model_id===result.request.model_id)!);
      const completed: QuickRecord = {schema_version:2,mode:'quick_round',challenge_id:id,source_role:'illustrative_training_example',hypothesis,requested_at,measured_at:new Date().toISOString(),baselines:data.baselines,results,feedback:quickFeedback(id,data.baselines,results,hypothesis.choice,mapping),...(id==='shortcut'?{model_mapping:mapping}:{}),reflection:''};
      setRecord(completed);onComplete(completed);
    } catch(problem) {setError(problem instanceof Error?problem.message:'Test failed. Your guess is saved; try again.');}
    finally{lock.current=false;setBusy(false);}
  }
  function restart() {
    setAttempt(value=>value+1);setMapping(freshMapping());setHypothesis(null);setRecord(null);setError('');
  }
  const labels=LABELS[id];
  const winner = record?.feedback.winner;
  const winnerText = winner==='similar' ? 'Both had a similar effect' : winner ? `${labels[winner==='A'?0:1]} changed it more` : '';
  const scores = data?.saliency ? PATCHES.map(({x,y})=>data.saliency!.baseline_scores.slice(y,y+4).reduce((sum,row)=>sum+row.slice(x,x+4).reduce((a,b)=>a+b,0),0)) : data?.activation?.baseline_mean_by_channel.slice(0,2);
  const strong = scores ? scores[0]>scores[1]?'A':scores[1]>scores[0]?'B':'similar' : null;
  const targetIndex = CLASSES.indexOf(sample.class);
  return <section className="quick-round" aria-label={`Quick ${definition.title}`} aria-busy={busy}>
    <div className="quick-heading"><div><span className="eyebrow">ROUND {definition.number} / 03</span><h2 ref={heading} tabIndex={-1}>{QUESTIONS[id]}</h2></div><span className="round-status">{record?'✓ Discovered':hypothesis?'02 / Test':'01 / Guess'}</span></div>
    {error && <div role="alert" className="error-banner">{error}{!data&&<button onClick={()=>setRetry(value=>value+1)}>Retry round</button>}</div>}
    {!data ? <p role="status">{error?'Round unavailable.':'Getting your drawing ready…'}</p> : <>
      {!record && <>
        <p className="quick-prompt">{id==='evidence' ? 'A heatmap highlights sensitive pixels. Which spot will change the fish probability more when we cover it?' : id==='hidden' ? 'These are two learned features. Which will change the airplane probability more when we switch it off?' : 'Same airplane. Two models. Which model’s airplane probability will vary more when we change the background?'}</p>
        <div className="quick-options">{(['A','B'] as const).map((alias,index)=><div className="quick-option" key={alias}>
          <span className="option-letter">{alias}</span>
          {id==='hidden'&&data.activation ? <PixelCanvas values={(data.activation.baseline_values as number[][][])[index]} upper={data.activation.baseline_max} label={`${labels[index]} response map`} /> : <div className="quick-drawing"><PixelCanvas rgb={data.baselines[id==='shortcut'?index:0].input.baseline_rgb_chw} saliency={id==='evidence'?data.saliency?.baseline_scores:undefined} opacity={.65} label={`${labels[index]} drawing`} />{id==='evidence'&&<span className="chosen-patch" style={{left:`${PATCHES[index].x/28*100}%`,top:`${PATCHES[index].y/28*100}%`,width:`${4/28*100}%`,height:`${4/28*100}%`}} />}</div>}
          <h3>{labels[index]}</h3><p>{id==='shortcut' ? `${data.baselines[index].baseline.predicted_class} · ${(100*data.baselines[index].baseline.probabilities[targetIndex]).toFixed(1)}% airplane` : strong===alias ? id==='evidence'?'More highlighted':'More active' : strong==='similar'?'Equally strong':id==='evidence'?'Less highlighted':'Less active'}</p>
        </div>)}</div>
        {!hypothesis ? <div className="quick-predictions" aria-label="Make your prediction"><p>Tap your guess. Getting surprised is part of the game.</p>{(['A','B','similar'] as const).map(choice=><button key={choice} disabled={busy} onClick={()=>predict(choice)}>{choice==='similar'?'About the same':labels[choice==='A'?0:1]}</button>)}</div> : <div className="quick-test"><p data-testid="quick-guess">Your guess: <b>{hypothesis.choice==='similar'?'about the same':labels[hypothesis.choice==='A'?0:1]}</b></p><button className="primary-button" disabled={busy} onClick={()=>void test()}>{busy?'Testing the real model…':id==='shortcut'?'Swap the backgrounds':id==='hidden'?'Test both switches':'Cover both spots'}</button><small>{id==='shortcut'?'We test green, blue, and yellow with the drawing unchanged.':'We test each change separately from the same starting point.'}</small></div>}
      </>}
      {record && <div className="quick-reveal" aria-live="polite">
        <span className="eyebrow">{record.feedback.matches?'YOUR PREDICTION MATCHED':'A DISCOVERY TO TAKE WITH YOU'}</span><h3 data-testid="quick-outcome">{winnerText}.</h3>
        <div className="quick-scores">{(['A','B'] as const).map((alias,index)=><div key={alias}><span>{labels[index]}</span>
          {id==='evidence'&&<PixelCanvas rgb={record.results[index].input.modified_rgb_chw} label={`${labels[index]} after covering`} />}
          {id==='hidden'&&data.activation&&<><PixelCanvas values={(data.activation.baseline_values as number[][][])[index]} upper={data.activation.baseline_max} label={`${labels[index]} before switching off`} /><small className="quick-map-note">Original response, before switch-off</small></>}
          {id==='shortcut'&&<div className="quick-color-results">{[...record.baselines,...record.results].filter(row=>row.request.model_id===mapping[alias]).map(row=><div key={row.experiment_id}><PixelCanvas rgb={row.input.modified_rgb_chw} label={`${labels[index]} on ${row.input.modified_background}`} /><small>{row.input.modified_background}<br/><b>{row.modified.predicted_class}</b></small></div>)}</div>}
          <strong>{record.feedback[alias].toFixed(2)}<small> probability points</small></strong><p>{id==='shortcut'?'Range across three colors':'Size of the probability change'}</p>{id!=='shortcut'&&<span className="quick-class-change">{record.results[index].baseline.predicted_class} → {record.results[index].modified.predicted_class}</span>}</div>)}</div>
        <p className="quick-lesson">{id==='evidence' ? `${winner==='similar'?'The two spots had similar effects here.':strong==='similar'?'The spots looked equally highlighted, but had different effects here.':winner===strong?'The more highlighted spot had the bigger effect here.':'The less highlighted spot had the bigger effect here.'} A heatmap describes tiny changes. Covering a spot is a bigger test. Looking important and having a big effect are different things.` : id==='hidden' ? `${winner==='similar'?'These features had similar effects here.':strong==='similar'?'The features looked equally active, but had different effects here.':winner===strong?'The more active feature had the bigger effect here.':'The quieter feature had the bigger effect here.'} A feature can light up strongly without controlling the answer. Switch it off to find out.` : `Model A had ${mapping.A} training; Model B had ${mapping.B} training. ${winner==='similar'?'Both models changed by a similar amount here.':`Model ${winner} was more background-sensitive on this drawing.`} Training clues can become shortcuts.`}</p>
        {id==='shortcut'&&<div className="quick-regimes"><p><b>Balanced:</b> colors were independent of drawing type.</p><p><b>Shortcut:</b> cats usually had green, fish blue, and airplanes yellow (90%).</p></div>}
        <div className="quick-actions"><button className="primary-button" onClick={onNext}>{id==='shortcut'?'Back to the challenges':'Next challenge →'}</button><button disabled={busy} onClick={restart}>Play again</button></div>
        <details className="quick-details"><summary>See the evidence / save this round</summary>
          <p>These percentages are uncalibrated model probabilities. Comparisons within 1 probability point count as “about the same”; this is a teaching tolerance. These are curated training examples, not a learning study or held-out evaluation.</p>
          <p>Your original guess: {hypothesis?.choice==='similar'?'about the same':labels[hypothesis?.choice==='A'?0:1]}. It {record.feedback.matches?'matched':'did not match'} the measured comparison.</p>
          {id!=='shortcut'&&<p>{id==='evidence'?`Equal 4 × 4 patches at (8,12) and (20,12), each replaced with RGB (128,128,128). Raw saliency sums: ${scores?.map(value=>value.toFixed(4)).join(' / ')}.`:`Conv2 channels 0 and 1, zeroed after ReLU before pooling. Mean activity: ${scores?.map(value=>value.toFixed(4)).join(' / ')}.`} The winner compares absolute target-probability changes, regardless of direction.</p>}
          {record.results.map((result,index)=><details key={result.experiment_id}><summary>{id==='shortcut'?`Model ${result.request.model_id===mapping.A?'A':'B'} · ${result.input.modified_background}`:labels[index]} · {signed(result.comparison.target_probability_point_change,2)} probability points</summary><MeasuredComparison result={result}/></details>)}
          <label>What surprised you? (optional)<textarea aria-label="Optional round reflection" maxLength={2000} value={record.reflection} onChange={event=>{const updated={...record,reflection:event.target.value};setRecord(updated);onComplete(updated);}} /></label>
          <button onClick={()=>downloadRecord(record)}>Download round record</button>
        </details>
        <p className="quick-limit">One drawing shows one example. Try free exploration to investigate more.</p>
      </div>}
    </>}
  </section>;
}
