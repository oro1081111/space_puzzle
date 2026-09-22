"""Local coordinate training with fixed multi-seed games and weakest-group reward."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from training.teacher_train import tournament
from training.teacher_eval import summarize
from training.verify import Oracle, ROOT, low_priority


def objective(result):
    report=summarize(result['rows'],result['games'])
    return (report['win_rate']+.2*min(g['win_rate'] for g in report['pairs'].values())+
            .2*min(g['win_rate'] for g in report['seats'].values())+.03*result['fitness'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--size',type=int,choices=[15,30],required=True)
    p.add_argument('--initial',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--seeds',nargs='+',type=int,required=True)
    p.add_argument('--validation-seeds',nargs='+',type=int,required=True)
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--workers',type=int,choices=[1,2,3,4],default=1)
    args=p.parse_args()
    if set(args.seeds)&set(args.validation_seeds):p.error('Validation seeds overlap training')
    low_priority()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    weights=json.loads(Path(args.initial).read_text())
    (out/'initial.json').write_text(json.dumps(weights,indent=2))
    files=['grid-clash/index.html','training/teacher.js','training/oracle.cjs',
           'training/teacher_refine.py','training/teacher_train.py','training/teacher_eval.py','training/verify.py']
    for f in files:
        target=out/'source'/f;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/f).read_bytes())
    (out/'config.json').write_text(json.dumps(dict(arguments=vars(args),fingerprints={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}),indent=2))
    oracle=Oracle(source_root=out/'source');start=time.perf_counter();trial=0
    try:
        with (out/'trials.jsonl').open('w') as stream:
            def test(w):
                global trial
                r=tournament(oracle,args.size,w,args.seeds,full=True,workers=args.workers)
                result=dict(trial=trial,weights=w,objective=objective(r),**r)
                trial+=1;stream.write(json.dumps(result)+'\n');stream.flush()
                print(json.dumps({k:v for k,v in result.items() if k not in ('rows','weights')}),flush=True)
                return result
            best=test(weights)
            for round_id in range(args.rounds):
                for key,default in [('raceCost',5),('edgePlan',36),('loopArea',76),('trap',7.32),
                                    ('race',25),('contest',82),('space',18),('enemy',11.41)]:
                    if (out/'PAUSE').exists():break
                    center=best['weights'].get(key,default)
                    for factor in (.67,1.5):
                        candidate=dict(best['weights']);candidate[key]=round(center*factor,4)
                        result=test(candidate)
                        if result['objective']>best['objective']:
                            best=result
                            (out/'best-training.json').write_text(json.dumps(best['weights'],indent=2))
                if (out/'PAUSE').exists():break
        validation=tournament(oracle,args.size,best['weights'],args.validation_seeds,full=True,workers=args.workers)
        (out/'validation.json').write_text(json.dumps(validation,indent=2))
        (out/'selected.json').write_text(json.dumps(best['weights'],indent=2))
        report=summarize(validation['rows'],validation['games'])
        report.update(trials=trial,seconds=time.perf_counter()-start)
        (out/'summary.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
    finally:oracle.close()
