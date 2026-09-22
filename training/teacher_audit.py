"""Frozen-candidate, many-seed development audit in the real JS engine."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import threading
from training.teacher_train import tournament
from training.teacher_eval import summarize
from training.verify import Oracle, ROOT, low_priority

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--size',type=int,choices=[15,30],required=True)
    p.add_argument('--weights',nargs='+',required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--count',type=int,default=20)
    p.add_argument('--workers',type=int,choices=[1,2,3,4],default=2)
    p.add_argument('--output',required=True)
    p.add_argument('--search',action='store_true')
    p.add_argument('--mixed-plans',action='store_true')
    p.add_argument('--prune',action='store_true')
    p.add_argument('--top-cuts',type=int,default=4)
    p.add_argument('--horizon',type=int,default=120)
    p.add_argument('--profiles',type=int,choices=[3,9],default=9)
    p.add_argument('--purpose',choices=['development_audit','heldout'],default='development_audit')
    args=p.parse_args();low_priority()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    files=['grid-clash/index.html','training/oracle.cjs','training/teacher.js',
           'training/teacher_train.py','training/teacher_eval.py','training/teacher_audit.py','training/verify.py']
    for f in files:
        target=out/'source'/f;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/f).read_bytes())
    (out/'config.json').write_text(json.dumps(dict(arguments=vars(args),fingerprints={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}),indent=2))
    oracle=Oracle(source_root=out/'source')
    try:
        for index,filename in enumerate(args.weights):
            weights=json.loads(Path(filename).read_text())
            name=f'{index:02d}-{Path(filename).parent.name}'
            (out/(name+'-weights.json')).write_text(json.dumps(weights,indent=2))
            start=time.perf_counter()
            search=dict(candidate=True,topCuts=args.top_cuts,horizon=args.horizon,profiles=args.profiles,mixedPlans=args.mixed_plans,prune=args.prune) if args.search else None
            lock=threading.Lock();completed=[]
            with (out/(name+'-matches.jsonl')).open('w') as stream:
                def progress(row):
                    with lock:
                        completed.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                        if len(completed)%27==0:
                            print(json.dumps(dict(candidate=name,games=len(completed),planned=27*args.count,
                                                  wins=sum(r['win'] for r in completed))),flush=True)
                result=tournament(oracle,args.size,weights,list(range(args.seed,args.seed+args.count)),True,args.workers,search,progress)
            summary=summarize(result['rows'],27*args.count)
            summary.update(seconds=time.perf_counter()-start,source_weights=filename,
                           purpose=args.purpose,engine='actual_browser_JS',python_parity='separate_regression_tests')
            (out/(name+'-summary.json')).write_text(json.dumps(summary,indent=2))
            print(json.dumps(summary),flush=True)
    finally:oracle.close()
