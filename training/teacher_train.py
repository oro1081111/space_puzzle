"""Low-load black-box training of coherent route scoring, not neural imitation."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import time
from concurrent.futures import ThreadPoolExecutor
from training.verify import Oracle, ROOT, low_priority


def tournament(oracle, size, weights, seeds, full=False, workers=1, search=None, progress=None, seats=None):
    pairs = list(itertools.product(['apex', 'bastion', 'sweep'], repeat=2)) if full else [
        ('apex', 'bastion'), ('bastion', 'sweep'), ('sweep', 'apex')]
    cases=list(itertools.product(seeds,pairs,range(3) if seats is None else seats))
    def play_cases(process,subset):
        rows=[]
        for seed,pair,seat in subset:
            styles=list(pair);styles.insert(seat,'candidate')
            r=process.call(op='fastTeacher' if search else 'fastMatch',size=size,count=3,seed=seed,
                           styles=styles,weights=weights,player=seat,options=search)
            margin=r['scores'][seat]-max(s for i,s in enumerate(r['scores']) if i!=seat)
            row=dict(seed=seed,pair=pair,seat=seat,**r,win=r['finished'] and margin>0,
                     tie=r['finished'] and margin==0,margin=margin if r['finished'] else -size*size)
            rows.append(row)
            if progress:progress(row)
        return rows
    if workers>1:
        def chunk(subset):
            child=Oracle(source_root=oracle.source_root)
            try:return play_cases(child,subset)
            finally:child.close()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            chunks=[cases[i::workers] for i in range(min(workers,len(cases)))]
            rows=[row for batch in pool.map(chunk,chunks) for row in batch]
        rows.sort(key=lambda r:(seeds.index(r['seed']),pairs.index(tuple(r['pair'])),r['seat']))
    else:
        rows=play_cases(oracle,cases)
    wins = sum(r['win'] for r in rows)
    fitness = wins/len(rows)+.15*sum(r['margin'] for r in rows)/(len(rows)*size*size)
    return dict(wins=wins, games=len(rows), fitness=fitness, rows=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--size', type=int, choices=[15,30], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--trials', type=int, default=48)
    p.add_argument('--seed', type=int, default=920001)
    p.add_argument('--extended', action='store_true')
    p.add_argument('--initial',nargs='+')
    p.add_argument('--robust', action='store_true')
    p.add_argument('--workers',type=int,choices=[1,2,3,4],default=1)
    args = p.parse_args()
    low_priority()
    out = Path(args.output);out.mkdir(parents=True, exist_ok=False)
    files = ['training/oracle.cjs','training/teacher.js','training/teacher_train.py','training/verify.py','grid-clash/index.html']
    for f in files:
        snapshot=out/'source'/f;snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes((ROOT/f).read_bytes())
    (out/'config.json').write_text(json.dumps(dict(arguments=vars(args),fingerprints={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}),indent=2))
    rng = random.Random(args.seed)
    initial=[json.loads(Path(f).read_text()) for f in (args.initial or [])]
    oracle = Oracle(source_root=out/'source');population=[];start=time.perf_counter()
    try:
        with (out/'trials.jsonl').open('w') as stream:
            for trial in range(args.trials):
                if (out/'PAUSE').exists():break
                if args.robust and trial<len(initial):
                    weights=dict(initial[trial])
                elif args.extended and not args.robust and trial<24:
                    weights=dict(raceCost=[0,1,5,20][trial%4],topology=1,
                        cuts=int(trial<12),lengthPower=[0,.5,1][(trial//4)%3],loopDepth=22)
                elif trial<18 and not args.robust:
                    weights=dict(raceCost=[0,2,5,10,20,40][trial%6],loopDepth=[14,22,30][trial//6])
                else:
                    elite=sorted(population,key=lambda x:x['fitness'],reverse=True)[:5]
                    weights=dict(rng.choice(elite)['weights'])
                    for key,base,lo,hi in [('raceCost',10,.1,150),('loopArea',76,10,300),
                        ('loopEff',44,5,200),('trap',7.32,.3,60),('trapLock',225,10,1500),
                        ('edgePlan',36,1,200),('loopLen',.96,.1,20)]:
                        if rng.random()<.6:weights[key]=round(max(lo,min(hi,weights.get(key,base)*rng.lognormvariate(0,.6))),3)
                    if rng.random()<.4:weights['loopDepth']=rng.choice([12,16,20,24,28,32])
                    if args.extended:
                        weights['beamRace']=rng.choice([0,.1,.5,2])
                        weights['turnBias']=rng.choice([-4,0,1,4])
                        weights['bbox']=rng.choice([.5,1,1.75,3])
                        weights['lengthPower']=rng.choice([0,.5,1,1.5])
                        weights['scaledPhase']=rng.choice([0,1])
                    if args.robust:
                        weights['directionBias']=rng.choice([0,1,2,3])
                        weights['avoidClash']=rng.choice([0,1])
                        weights['scaledPhase']=rng.choice([0,1])
                        for key,base in [('race',25),('contest',82),('dist',1.05),('cap',192.4),('enemy',11.41),('space',18)]:
                            if rng.random()<.35:weights[key]=round(weights.get(key,base)*rng.lognormvariate(0,.45),3)
                result=tournament(oracle,args.size,weights,list(range(args.seed,args.seed+(3 if args.robust else 1))),full=args.extended or args.robust,workers=args.workers)
                entry=dict(trial=trial,weights=weights,**result)
                population.append(entry);stream.write(json.dumps(entry)+'\n');stream.flush()
                print(json.dumps({k:v for k,v in entry.items() if k!='rows'}),flush=True)
        finalists=[]
        for entry in sorted(population,key=lambda x:x['fitness'],reverse=True)[:5]:
            result=tournament(oracle,args.size,entry['weights'],[args.seed+100,args.seed+101],full=True,workers=args.workers)
            finalist=dict(weights=entry['weights'],trial=entry['trial'],**result)
            finalists.append(finalist)
            print(json.dumps(dict(validation_trial=entry['trial'],wins=result['wins'],games=result['games'])),flush=True)
        (out/'validation.json').write_text(json.dumps(finalists,indent=2))
        best=max(finalists,key=lambda x:x['fitness'])
        (out/'selected.json').write_text(json.dumps(best['weights'],indent=2))
        (out/'summary.json').write_text(json.dumps(dict(trials=len(population),selected_trial=best['trial'],
            wins=best['wins'],games=best['games'],seconds=time.perf_counter()-start,release_approved=False),indent=2))
    finally:oracle.close()
