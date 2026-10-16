import { useEffect, useState } from 'react';
import type { Sample } from '../types';
import Investigation from './Investigation';
import Shortcut from './Shortcut';
import QuickRound from './QuickRound';
import type { QuickRecord } from './quick';
import { CHALLENGES, downloadRecord } from './state';
import type { ChallengeID, CompletionRecord } from './state';
import './challenges.css';

export default function ChallengeHub({ samples, active, onBusy }: { samples: Sample[]; active: boolean; onBusy: (busy: boolean) => void }) {
  const [lessonMode, setLessonMode] = useState<'quick' | 'full'>('quick');
  const [selected, setSelected] = useState<ChallengeID | null>(null);
  const [busy, setBusy] = useState(false);
  const [completed, setCompleted] = useState<Partial<Record<ChallengeID, CompletionRecord | QuickRecord>>>({});
  const total = Object.keys(completed).length;
  useEffect(() => { if (active) onBusy(busy); }, [active, busy, onBusy]);
  function complete(record: CompletionRecord | QuickRecord) { setCompleted(previous => ({ ...previous, [record.challenge_id]: record })); }
  return <section className={`challenge-hub ${selected?'is-playing':''}`} id="guided-challenges" aria-label="Guided investigations">
    <div className="challenge-hub-heading"><div><span className="eyebrow">GUESS. TEST. DISCOVER.</span><h2>Three quick discoveries.</h2><p>Tap a prediction, test the real model, and see what changes. No forms required.</p></div><div className="completion-badge" aria-label={`${total} of 3 challenges completed`}><strong>{total}/3</strong><span>discoveries made</span></div></div>
    <details className="teacher-options"><summary>Teacher options</summary><button disabled={busy} onClick={()=>setLessonMode(lessonMode==='quick'?'full':'quick')}>{lessonMode==='quick'?'Longer investigations':'Quick rounds'}</button><p>Longer investigations add open selections, written hypotheses, and required reflections.</p></details>
    <nav className="challenge-menu" aria-label="Choose an investigation">{CHALLENGES.map(item => <button key={item.id} aria-label={`Open ${item.title}`} aria-current={selected===item.id?'page':undefined} disabled={busy} onClick={()=>setSelected(item.id)}><span className="eyebrow">{item.number} {completed[item.id]?' / COMPLETE':''}</span><strong>{item.title}</strong><span>{lessonMode==='quick' ? item.id==='evidence'?'Cover a spot. See what matters.':item.id==='hidden'?'Switch a feature off. See what changes.':'Change the color. Catch the shortcut.':item.description}</span><b>{completed[item.id]?'✓ Discovered':selected===item.id?'Playing ↓':'Play →'}</b></button>)}</nav>
    {!selected && <div className="challenge-welcome"><h3>Pick a round to start.</h3><p>Make a guess, run one test, then discover the lesson. A surprise counts as progress too. You can save the evidence and add a reflection afterward.</p><p>Progress stays in this browser session; reloading starts fresh.</p></div>}
    {CHALLENGES.map((item,index) => <div hidden={selected!==item.id} key={item.id}>
      <div hidden={lessonMode!=='quick'}><QuickRound id={item.id} samples={samples} active={active&&lessonMode==='quick'&&selected===item.id} onBusy={setBusy} onComplete={complete} onNext={()=>setSelected(CHALLENGES[index+1]?.id??null)} /></div>
      <div hidden={lessonMode!=='full'}>{item.id==='shortcut' ? <Shortcut samples={samples} active={active&&lessonMode==='full'&&selected===item.id} onBusy={setBusy} onComplete={complete} /> : <Investigation id={item.id} samples={samples} active={active&&lessonMode==='full'&&selected===item.id} onBusy={setBusy} onComplete={complete} />}</div>
    </div>)}
    {total>0 && <details className="completed-records"><summary>Saved discoveries ({total})</summary>{CHALLENGES.filter(item=>completed[item.id]).map(item=><button key={item.id} onClick={()=>downloadRecord(completed[item.id]!)}>Download {item.title}</button>)}</details>}
  </section>;
}
