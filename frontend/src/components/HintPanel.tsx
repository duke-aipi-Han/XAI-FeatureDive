import type { ChannelHint, HintResult } from '../types';
import { signed } from './Results';
import './hints.css';

export default function HintPanel({ data, kind, selected, pristine, busy, onChannel, onErase }: {
  data: HintResult; kind:'feature'|'erase'; selected:{layer:'conv1'|'conv2';channel:number}|null;
  pristine:boolean; busy:boolean; onChannel:(hint:ChannelHint)=>void; onErase:()=>void;
}) {
  const max=Math.max(...data.channels.map(row=>Math.abs(row.estimated_support)),1e-8);
  const chosen=selected&&data.channels.find(row=>row.layer===selected.layer&&row.channel===selected.channel);
  function row(hint:ChannelHint) {
    return <button key={`${hint.layer}:${hint.channel}`} className="hint-channel" aria-label={`Hint ${hint.layer} channel ${hint.channel}`} aria-pressed={selected?.layer===hint.layer&&selected.channel===hint.channel} disabled={busy} onClick={()=>onChannel(hint)}>
      <span><b>#{hint.rank}</b> {hint.layer.toUpperCase()} / {hint.channel}</span><span className="support-track" aria-hidden="true"><i style={{width:`${Math.abs(hint.estimated_support)/max*50}%`,left:hint.estimated_support>=0?'50%':`${50-Math.abs(hint.estimated_support)/max*50}%`,background:hint.estimated_support>=0?'#d46d54':'#4f85b3'}} /></span><strong>{signed(hint.estimated_support,2)}</strong>
    </button>;
  }
  return <section className="hint-panel" aria-label={kind==='feature'?'Feature Radar hints':'Eraser Vision hints'}>
    <div className="hint-panel-title"><span>✦ {kind==='feature'?'FEATURE RADAR':'ERASER VISION'}</span><b>ESTIMATE · ORIGINAL INPUT</b></div>
    {kind==='feature'?<>
      <p>Which feature might support <b>{data.experiment.request.target_class}</b>? Choose a clue, then switch that channel off.</p>
      <div className="support-legend"><span><i className="weakens"/>likely weakens when removed</span><span><i className="strengthens"/>likely strengthens</span></div>
      <div className="support-head"><span>Across Conv1 + Conv2</span><span>Estimated support</span></div>
      {data.channels.slice(0,3).map(row)}
      <details className="all-hints"><summary>All 24 channels</summary>{data.channels.slice(3).map(row)}</details>
      {chosen&&<p className="chosen-hint" data-testid="selected-channel-hint">Selected {chosen.layer.toUpperCase()} / {chosen.channel}: estimated margin change <b>{signed(chosen.estimated_margin_change,3)}</b>. Original mean activity {chosen.mean_activation.toFixed(4)}.</p>}
    </>:<>
      <p>The overlay estimates which strokes support <b>{data.experiment.request.target_class}</b>. Coral suggests erasing may weaken it; blue suggests it may strengthen it.</p>
      <div className="support-legend"><span><i className="weakens"/>likely weakens</span><span><i className="strengthens"/>likely strengthens</span></div>
      {data.suggested_patch?<div className="suggested-erase"><span>4 × 4 clue at ({data.suggested_patch.rectangle.x},{data.suggested_patch.rectangle.y})</span><b data-testid="erase-hint-estimate">Estimated margin change {signed(data.suggested_patch.estimated_margin_change,3)}</b><button disabled={busy||!pristine} onClick={onErase}>Erase hinted patch</button></div>:<p>No positive weakening clue for this input. You can still investigate with Pixel Eraser.</p>}
      <p className="hint-scale">Signed overlay uses a shared −max → +max scale for this image. The outlined patch is a suggestion; no pixels change until you test.</p>
    </>}
    {!pristine&&<p className="hint-reset">These clues describe the original input. Reset baseline before testing another clue independently.</p>}
    <details className="hint-explanation"><summary>How does this power-up work?</summary><p>A first-order gradient estimate predicts a change in the selected class score minus the strongest competing score (its margin). Channels use activation × margin gradient. Eraser Vision uses the actual RGB change toward the original background. The hint uses no trial ablations. Full removal can behave differently; inspect the observed effect afterward.</p></details>
  </section>;
}
