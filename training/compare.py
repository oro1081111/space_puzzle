"""Resumable, sequential model comparison; never train during evaluation."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from training.evaluate import evaluate
from training.imitate import low_load
from training.verify import ROOT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('models',nargs='+',type=Path)
    parser.add_argument('--opponents',nargs='+',default=['greedy','apex','bastion','sweep'])
    parser.add_argument('--seeds',type=int,default=5)
    parser.add_argument('--seed-start',type=int,default=9400)
    parser.add_argument('--deterministic',action='store_true')
    parser.add_argument('--spread-seeds',action='store_true')
    parser.add_argument('--search-depth',type=int,choices=[0,1,3,5],default=0)
    parser.add_argument('--no-policy-prior',action='store_true')
    parser.add_argument('--opening-steps',type=int,choices=[0,4],default=0)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); low_load()
    if args.no_policy_prior and args.search_depth!=1:
        parser.error('--no-policy-prior requires --search-depth 1')
    args.output.mkdir(parents=True,exist_ok=True)
    seed_values=(random.Random(args.seed_start).sample(range(2**31),args.seeds)
                 if args.spread_seeds else list(range(args.seed_start,args.seed_start+args.seeds)))
    source_hash=hashlib.sha256((ROOT/'grid-clash/index.html').read_bytes()).hexdigest()
    summary={}
    for model in args.models:
        name=model.parent.name+'_'+model.stem
        checkpoint_hash=hashlib.sha256(model.read_bytes()).hexdigest()
        summary[name]={}
        for opponent in args.opponents:
            opponent_tag=(Path(opponent).parent.name+'_'+Path(opponent).stem
                          if str(opponent).endswith('.zip') else opponent)
            target=args.output/f'{name}_{opponent_tag}.json'
            key=dict(model_sha256=checkpoint_hash,source_sha256=source_hash,seed_start=args.seed_start,
                     seeds=args.seeds,deterministic=args.deterministic,opponent=opponent)
            if args.spread_seeds:
                key['seed_values']=seed_values
            if args.search_depth:
                key['search_depth']=args.search_depth
                key['controller_sha256']={name:hashlib.sha256((ROOT/'training'/name).read_bytes()).hexdigest()
                                          for name in ['search.py','engine.py','evaluate.py','oracle.cjs']}
            if args.no_policy_prior:
                key['no_policy_prior']=True
            if args.opening_steps:
                key['opening_steps']=args.opening_steps
            if target.exists():
                result=json.loads(target.read_text())
                if result.get('manifest')!=key:
                    raise RuntimeError(f'Existing report has a different manifest: {target}')
            else:
                result=evaluate(model,[opponent],seed_values,
                                args.deterministic,quiet=True,search_depth=args.search_depth,no_policy_prior=args.no_policy_prior,
                                opening_steps=args.opening_steps)
                result['manifest']=key
                target.write_text(json.dumps(result,indent=2),encoding='utf-8')
            summary[name][opponent]=result['summary'][opponent]
            print(json.dumps(dict(model=name,opponent=opponent,**result['summary'][opponent])),flush=True)
            (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
