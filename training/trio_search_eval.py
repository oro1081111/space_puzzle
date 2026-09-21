"""Normal-opening, seat-balanced three-player controller evaluation."""
import argparse
import hashlib
import itertools
import json
import os
import subprocess
from pathlib import Path
import time
from training.three_player import logits, low_load
from training.verify import Oracle
from training.engine import Game, simple_action
from sb3_contrib import MaskablePPO
import numpy as np

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--variants',nargs='+',default=['baseline','exact','exact-noprior','exact-noprior-own'])
    p.add_argument('--seeds',nargs='+',type=int,default=[71])
    p.add_argument('--prefix',type=int,default=0)
    p.add_argument('--same-style',action='store_true')
    p.add_argument('--source-ref',default='d218222',help='Frozen controller commit, or current for the working tree')
    p.add_argument('--output',required=True)
    args=p.parse_args();low_load()
    if args.source_ref!='current':os.environ['TRIO_SOURCE_REF']=args.source_ref
    else:os.environ.pop('TRIO_SOURCE_REF',None)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    model=MaskablePPO.load('training/runs/atlas3-round1/snapshot_000016384.zip',device='cpu')
    fingerprints={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in map(Path,['grid-clash/index.html','training/oracle.cjs','training/trio_search_eval.py','training/runs/atlas3-round1/snapshot_000016384.zip'])}
    if args.source_ref!='current':fingerprints['evaluated_html']=hashlib.sha256(subprocess.check_output(['git','show',args.source_ref+':grid-clash/index.html'])).hexdigest()
    (output/'config.json').write_text(json.dumps(dict(arguments=vars(args),fingerprints=fingerprints)),encoding='utf-8')
    pairs=list(itertools.product(['apex','bastion','sweep'],repeat=2) if args.same_style else itertools.permutations(['apex','bastion','sweep'],2))
    for variant in args.variants:
        os.environ['TRIO_ABLATION']='' if variant=='baseline' else variant
        oracle=Oracle();rows=[];start=time.perf_counter()
        try:
            with (output/(variant+'.jsonl')).open('w',encoding='utf-8') as stream:
                for seed in args.seeds:
                    for pair in pairs:
                        for seat in range(3):
                            styles=list(pair);styles.insert(seat,'atlas')
                            g=Game(count=3,size=15);g.load(oracle.call(op='reset',size=15,count=3,seed=seed,styles=styles))
                            rng=np.random.default_rng(seed)
                            for _ in range(args.prefix):g.load(oracle.call(op='play',actions=[simple_action(g,i,rng) for i in range(3)]))
                            attempts=0
                            while not g.ended and attempts<3600:
                                r=oracle.call(op='atlas',logits=logits(model,g).tolist())
                                g.step(r['actions']);attempts+=1
                                for k,v in g.state().items():assert r[k]==v,k
                            diff=g.scores[seat]-max(s for i,s in enumerate(g.scores) if i!=seat)
                            row=dict(seed=seed,pair=pair,seat=seat,scores=g.scores,win=diff>0,tie=diff==0,margin=diff,finished=g.ended,attempts=attempts)
                            rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                    print(json.dumps(dict(variant=variant,games=len(rows),wins=sum(r['win'] for r in rows),ties=sum(r['tie'] for r in rows),unfinished=sum(not r['finished'] for r in rows),seconds=round(time.perf_counter()-start,1))),flush=True)
        finally:oracle.close()
