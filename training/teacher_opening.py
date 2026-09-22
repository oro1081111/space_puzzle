"""Train a fixed-start opening book, then use the fast route policy for the rest."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from training.teacher_train import tournament
from training.teacher_eval import summarize
from training.verify import ROOT, Oracle, low_priority


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--size',type=int,choices=[15,30],required=True)
    parser.add_argument('--initial',required=True)
    parser.add_argument('--seeds',nargs='+',type=int,required=True)
    parser.add_argument('--validation-seeds',nargs='+',type=int,required=True)
    parser.add_argument('--workers',type=int,choices=[1,2,3,4],default=2)
    parser.add_argument('--depth',type=int,default=0,help='Train short openings with a two-route beam instead of teacher proposals')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if set(args.seeds)&set(args.validation_seeds):parser.error('Seeds overlap')
    if not 0<=args.depth<=8:parser.error('Depth must be 0..8')
    low_priority();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    files=['grid-clash/index.html','training/oracle.cjs','training/teacher.js',
           'training/verify.py','training/teacher_train.py','training/teacher_eval.py','training/teacher_opening.py']
    for filename in files:
        target=out/'source'/filename;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/filename).read_bytes())
    (out/'config.json').write_text(json.dumps(dict(arguments=vars(args),fingerprints={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}),indent=2))
    weights=json.loads(Path(args.initial).read_text());weights.pop('openings',None)
    (out/'initial.json').write_text(json.dumps(weights,indent=2))
    oracle=Oracle(source_root=out/'source');start=time.perf_counter();book={};pools={};trials=0
    try:
        # Only generate proposals here. Actual complete training games rank them.
        for seat in range(3):
            oracle.call(op='reset',size=args.size,count=3,seed=101,weights=weights,
                        styles=['candidate' if i==seat else 'apex' for i in range(3)])
            if args.depth:
                pools[str(seat)]=[None]
                continue
            choice=oracle.call(op='teacher',player=seat,probe=True,
                options=dict(candidate=True,mixedPlans=True,topCuts=8,horizon=1,profiles=3,seed=101))['choice']
            pool=[None]
            for row in choice['rows']:
                opening=dict(plan=row['plan'],style='candidate' if row['style']=='cut' else row['style'])
                if opening['plan'] and opening not in pool:pool.append(opening)
                continuation=dict(opening,style='candidate')
                if continuation['plan'] and continuation not in pool:pool.append(continuation)
            pools[str(seat)]=pool
        (out/'proposals.json').write_text(json.dumps(pools,indent=2))
        with (out/'trials.jsonl').open('w') as stream:
            for seat in range(3):
                best=None
                def test(opening):
                    global trials
                    candidate=dict(weights,openings={str(seat):opening} if opening else {})
                    result=tournament(oracle,args.size,candidate,args.seeds,full=True,workers=args.workers,seats=[seat])
                    rows=result['rows']
                    wins=sum(r['win'] for r in rows)
                    objective=wins/len(rows)+.01*sum(r['margin'] for r in rows)/(len(rows)*args.size**2)
                    trial=dict(seat=seat,opening=opening,objective=objective,wins=wins,games=len(rows),rows=rows)
                    stream.write(json.dumps(trial)+'\n');stream.flush();trials+=1
                    print(json.dumps({k:v for k,v in trial.items() if k!='rows'}),flush=True)
                    return trial
                for opening in pools[str(seat)]:
                    trial=test(opening)
                    objective=trial['objective']
                    if best is None or objective>best['objective']:best=trial
                if args.depth:
                    state=oracle.call(op='reset',size=args.size,count=3,seed=101,weights=weights)
                    start_pos=tuple(state['positions'][seat]);blocked={tuple(p) for i,p in enumerate(state['positions']) if i!=seat}
                    beam=[[]];directions=[(0,-1),(1,0),(0,1),(-1,0)]
                    for depth in range(args.depth):
                        layer=[]
                        for prefix in beam:
                            x,y=start_pos;visited={start_pos}
                            for d in prefix:
                                dx,dy=directions[d];x+=dx;y+=dy;visited.add((x,y))
                            for d,(dx,dy) in enumerate(directions):
                                target=(x+dx,y+dy)
                                if not (0<=target[0]<args.size and 0<=target[1]<args.size) or target in visited or target in blocked:continue
                                trial=test(dict(plan=prefix+[d],style='candidate'));layer.append(trial)
                                if trial['objective']>best['objective']:best=trial
                        beam=[r['opening']['plan'] for r in sorted(layer,key=lambda r:r['objective'],reverse=True)[:2]]
                if best['opening']:book[str(seat)]=best['opening']
        selected=dict(weights,openings=book)
        (out/'selected.json').write_text(json.dumps(selected,indent=2))
        result=tournament(oracle,args.size,selected,args.validation_seeds,full=True,workers=args.workers)
        (out/'validation.json').write_text(json.dumps(result,indent=2))
        summary=summarize(result['rows'],27*len(args.validation_seeds))
        summary.update(trials=trials,seconds=time.perf_counter()-start,method='fixed_start_opening_book')
        (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    finally:oracle.close()
