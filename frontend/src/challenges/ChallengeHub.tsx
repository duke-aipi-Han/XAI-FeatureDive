import { useEffect, useState } from 'react';
import type { Sample } from '../types';
import Investigation from './Investigation';
import Shortcut from './Shortcut';
import { CHALLENGES, downloadRecord } from './state';
import type { ChallengeID, CompletionRecord } from './state';
import './challenges.css';

export default function ChallengeHub({ samples, active, onBusy }: { samples: Sample[]; active: boolean; onBusy: (busy: boolean) => void }) {
  const [selected, setSelected] = useState<ChallengeID | null>(null);
  const [busy, setBusy] = useState(false);
  const [completed, setCompleted] = useState<Partial<Record<ChallengeID, CompletionRecord>>>({});
  const total = Object.keys(completed).length;
  useEffect(() => { if (active) onBusy(busy); }, [active, busy, onBusy]);
  function complete(record: CompletionRecord) { setCompleted(previous => ({ ...previous, [record.challenge_id]: record })); }
  return <section className="challenge-hub" id="guided-challenges" aria-label="Guided investigations">
    <div className="challenge-hub-heading"><div><span className="eyebrow">MAKE A PREDICTION. TEST IT. REVISE IT.</span><h2>Three investigations. Real evidence.</h2><p>Practice asking what changes a model’s decision. Each investigation records what you expected before showing the measured result.</p></div><div className="completion-badge" aria-label={`${total} of 3 challenges completed`}><strong>{total}/3</strong><span>investigations completed</span></div></div>
    <nav className="challenge-menu" aria-label="Choose an investigation">{CHALLENGES.map(item => <button key={item.id} aria-label={`Open ${item.title}`} aria-current={selected===item.id?'page':undefined} disabled={busy} onClick={()=>setSelected(item.id)}><span className="eyebrow">{item.number} {completed[item.id]?' / COMPLETE':''}</span><strong>{item.title}</strong><span>{item.description}</span><b>{selected===item.id?'In progress ↓':completed[item.id]?'Review or retry ↗':'Begin investigation ↗'}</b></button>)}</nav>
    {!selected && <div className="challenge-welcome"><h3>Start anywhere. Keep the evidence honest.</h3><p>Observe → hypothesize → intervene → evaluate → reflect. You can revisit an investigation without losing this session’s work. Completion means you made comparisons and reflected; it does not grade your guesses or prove a learning benefit.</p><p>Hypotheses and reflections remain in browser memory, not in API requests or a user database. Reloading clears progress; download a completed record to keep it.</p></div>}
    {CHALLENGES.map(item => <div hidden={selected!==item.id} key={item.id}>
      {item.id==='shortcut' ? <Shortcut samples={samples} active={active&&selected===item.id} onBusy={setBusy} onComplete={complete} /> : <Investigation id={item.id} samples={samples} active={active&&selected===item.id} onBusy={setBusy} onComplete={complete} />}
    </div>)}
    {total>0 && <details className="completed-records"><summary>Completed investigation records ({total})</summary>{CHALLENGES.filter(item=>completed[item.id]).map(item=><button key={item.id} onClick={()=>downloadRecord(completed[item.id]!)}>Download {item.title}</button>)}</details>}
  </section>;
}
