"""Evaluate the actual JS planner; replay every chosen action in Python for parity."""
import argparse
import hashlib
import itertools
import json
import os
import random
from pathlib import Path
import time
import numpy as np

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
from training.engine import Game, simple_action
from training.verify import Oracle, ROOT, low_priority


def summarize(rows, planned):
    def group(items):
        return dict(games=len(items),wins=sum(r['win'] for r in items),ties=sum(r['tie'] for r in items),
                    win_rate=sum(r['win'] for r in items)/len(items) if items else None)
    result=group(rows)
    result['planned']=planned
    result['unfinished']=sum(not r['finished'] for r in rows)
    result['seats']={str(s):group([r for r in rows if r['seat']==s]) for s in range(3)}
    result['pairs']={a+'+'+b:group([r for r in rows if tuple(r['pair'])==(a,b)])
                     for a,b in itertools.product(['apex','bastion','sweep'],repeat=2)}
    seeds=sorted({r['seed'] for r in rows})
    actual={(r['seed'],tuple(r['pair']),r['seat']) for r in rows}
    expected=set(itertools.product(seeds,itertools.product(['apex','bastion','sweep'],repeat=2),range(3)))
    result['complete_matrix']=len(actual)==len(rows)==planned and actual==expected
    # Cluster by seed, preserving correlated lineups/seats within each draw.
    if len(seeds)>=2:
        seed_rates=[group([r for r in rows if r['seed']==seed])['win_rate'] for seed in seeds]
        rng=random.Random(1947)
        samples=sorted(sum(rng.choices(seed_rates,k=len(seeds)))/len(seeds) for _ in range(2000))
        result['seed_bootstrap_95']=[samples[49],samples[1949]]
    result['strength_gate_passed']=(result['complete_matrix'] and len(seeds)>=20 and len(rows)>=540
        and result['unfinished']==0 and result['win_rate']>=.7
        and all(g['win_rate'] is not None and g['win_rate']>.5 for g in [*result['seats'].values(),*result['pairs'].values()]))
    # Browser and current-controller comparison are separate required gates.
    result['release_approved']=False
    return result


def evaluate(args):
    low_priority()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    files = ['grid-clash/index.html', 'training/oracle.cjs', 'training/teacher.js',
             'training/teacher_eval.py', 'training/engine.py', 'training/verify.py']
    if args.browser_route:files+=['grid-clash/route-worker.js','grid-clash/atlas-route-v2.json']
    weights=json.loads(Path(args.weights).read_text()) if args.weights else {}
    if args.interior_fallback:weights['interiorFallback']=args.interior_fallback
    if args.weights:
        (output/'weights.json').write_text(json.dumps(weights,indent=2))
    for f in files:
        snapshot=output/'source'/f;snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes((ROOT/f).read_bytes())
    manifest = dict(arguments=vars(args), fingerprints={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    (output/'config.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    oracle = Oracle(source_root=output/'source')
    rows = []
    start = time.perf_counter()
    try:
        with (output/'matches.jsonl').open('w', encoding='utf-8') as stream:
            matrix = itertools.product(args.seeds, itertools.product(['apex', 'bastion', 'sweep'], repeat=2), range(3))
            for seed, pair, seat in matrix:
                if (output/'PAUSE').exists() or (args.max_games and len(rows) >= args.max_games):
                    break
                styles = list(pair)
                styles.insert(seat, 'atlas' if args.browser_route else 'candidate' if args.weights else args.baseline or 'apex')
                game = Game(count=3, size=args.size)
                game.load(oracle.call(op='reset', size=args.size, count=3, styles=styles, seed=seed,
                                      weights=weights))
                rng=np.random.default_rng(seed*3+seat)
                for _ in range(args.prefix):
                    if game.ended:break
                    actual=oracle.call(op='play',actions=[simple_action(game,p,rng) for p in range(3)])
                    assert game.step(actual['actions'])==actual['advanced']
                    for key,value in game.state().items():assert actual[key]==value,key
                attempts = plans = branches = 0
                latencies = []
                while not game.ended and attempts < 12*args.size**2:
                    tick = time.perf_counter()
                    if args.browser_route:
                        result=oracle.call(op='route',players=[seat])
                    elif args.baseline or (args.weights and not args.search):
                        result = oracle.call(op='play', actions=['ai']*3)
                    else:
                        result = oracle.call(op='teacher', player=seat, options=dict(
                            horizon=args.horizon, profiles=args.profiles, seed=101+game.turn*31,
                            replan=args.replan,candidate=bool(args.weights),directions=args.directions,topCuts=args.top_cuts,mixedPlans=args.mixed_plans,prune=args.prune))
                        choices = result['choice']['rows']
                        plans += bool(choices)
                        branches += sum(c['simulated'] for c in choices)
                    latencies.append(time.perf_counter()-tick)
                    advanced = game.step(result['actions'])
                    assert advanced == result['advanced']
                    for k, v in game.state().items():
                        assert result[k] == v, (k, seed, pair, seat, attempts)
                    attempts += 1
                margin = game.scores[seat]-max(s for i, s in enumerate(game.scores) if i != seat)
                row = dict(seed=seed, pair=pair, seat=seat, scores=game.scores,
                           win=game.ended and margin>0, tie=game.ended and margin==0, finished=game.ended, margin=margin,
                           attempts=attempts, plans=plans, rollout_attempts=branches,
                           seconds=sum(latencies), max_decision_seconds=max(latencies, default=0))
                rows.append(row)
                stream.write(json.dumps(row)+'\n');stream.flush()
                print(json.dumps(dict(game=len(rows), **row)), flush=True)
    finally:
        oracle.close()
    summary = summarize(rows,27*len(args.seeds))
    summary['seconds']=time.perf_counter()-start
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, choices=[15, 30], required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[910001])
    parser.add_argument('--horizon', type=int, default=16)
    parser.add_argument('--profiles', type=int, choices=[3, 9], default=3)
    parser.add_argument('--max-games', type=int, default=0)
    parser.add_argument('--baseline', choices=['apex', 'bastion', 'sweep'])
    parser.add_argument('--weights')
    parser.add_argument('--search', action='store_true')
    parser.add_argument('--browser-route', action='store_true')
    parser.add_argument('--prefix',type=int,default=0)
    parser.add_argument('--interior-fallback',choices=['apex','bastion','sweep','reactive'])
    parser.add_argument('--mixed-plans', action='store_true')
    parser.add_argument('--prune', action='store_true')
    parser.add_argument('--directions', action='store_true')
    parser.add_argument('--top-cuts',type=int,default=0)
    parser.add_argument('--replan', action='store_true')
    parser.add_argument('--output', required=True)
    evaluate(parser.parse_args())
