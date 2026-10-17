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
    <span className="hint-label">CLUE · ESTIMATE FROM ORIGINAL</span>
    <p>{kind==='feature'?'These channels may':'These strokes may'} support <b>{data.experiment.request.target_class}</b>. {kind==='feature'?'Pick one, then switch it off.':'Erase the outlined patch. Does the answer change?'}</p>
    {kind==='feature'?<>
      {data.channels.slice(0,3).map(row)}
      <details className="all-hints"><summary>All 24 channels</summary>{data.channels.slice(3).map(row)}</details>
      {chosen&&<p className="chosen-hint" data-testid="selected-channel-hint">Predicted margin change <b>{signed(chosen.estimated_margin_change,3)}</b></p>}
    </>:<>
      <div className="support-legend"><span><i className="weakens"/>weaken</span><span><i className="strengthens"/>strengthen</span></div>
      {data.suggested_patch?<div className="suggested-erase"><small>4 × 4 at ({data.suggested_patch.rectangle.x},{data.suggested_patch.rectangle.y})</small><span data-testid="erase-hint-estimate">Predicted margin change <b>{signed(data.suggested_patch.estimated_margin_change,3)}</b></span><button disabled={busy||!pristine} onClick={onErase}>Erase hinted patch</button></div>:<p>No weakening clue here. Try your own patch.</p>}
    </>}
    {!pristine&&<p className="hint-reset">Reset baseline to test a fresh clue.</p>}
    <details className="hint-explanation"><summary>How does this power-up work?</summary><p>Clues estimate removal’s effect on the target’s logit margin using original-input gradients. Channels use activation × gradient; pixels use the change back to background. Coral suggests weakening, blue strengthening. The model has not tested these removals. Full removal may behave differently.</p><p>Channel rankings span both layers. Activity alone is not importance. Pixel color intensity is scaled to this image’s largest absolute estimate.</p></details>
  </section>;
}
