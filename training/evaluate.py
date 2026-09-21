import argparse
import json
import os
from pathlib import Path
import time
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[variable]='1'
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from training.engine import Game, simple_action
from training.verify import Oracle


def evaluate(model_path, opponents, seeds, deterministic=False, quiet=False, search_depth=0, no_policy_prior=False, opening_steps=0):
    torch.set_num_threads(1)
    model=MaskablePPO.load(model_path,device='cpu')
    size=model.observation_space.shape[-1]
    oracle=Oracle() if any(o in ('apex','bastion','sweep') for o in opponents) else None
    results=[]
    try:
        for opponent in opponents:
            opponent_model=MaskablePPO.load(opponent,device='cpu') if str(opponent).endswith('.zip') else None
            for seed in seeds:
                for seat in range(2):
                    rng=np.random.default_rng(seed)
                    torch.manual_seed(seed)
                    game=Game(size=size)
                    heuristic=opponent in ('apex','bastion','sweep')
                    if heuristic:
                        game.load(oracle.call(op='reset',count=2,size=size,seed=seed,styles=[opponent]*2))
                    for _ in range(opening_steps):
                        if game.ended:break
                        prefix=[simple_action(game,p,rng,True) for p in range(2)]
                        if heuristic:game.load(oracle.call(op='play',actions=prefix))
                        else:game.step(prefix)
                    start=time.perf_counter()
                    collisions=decisions=0
                    new_moves=own_moves=stationary=0
                    decision_seconds=[]
                    # Safety cap marks an unfinished match, never converts it into a win.
                    while not game.ended and decisions<20000:
                        mask=game.mask(seat)
                        action=None
                        if mask.any():
                            decision_start=time.perf_counter()
                            if search_depth:
                                from training.search import choose_action
                                action=choose_action(None if no_policy_prior else model,game,seat,search_depth)
                            else:
                                action,_=model.predict(game.observation(seat),action_masks=mask,deterministic=deterministic)
                                action=int(action)
                            decision_seconds.append(time.perf_counter()-decision_start)
                        actions=[None,None]
                        actions[seat]=action
                        if opponent_model is not None:
                            enemy_mask=game.mask(1-seat)
                            if enemy_mask.any():
                                enemy_action,_=opponent_model.predict(game.observation(1-seat),action_masks=enemy_mask,deterministic=deterministic)
                                actions[1-seat]=int(enemy_action)
                        else:
                            actions[1-seat]='ai' if heuristic else simple_action(game,1-seat,rng,opponent!='random')
                        old_position=game.positions[seat]
                        old_board=game.board.copy()
                        if heuristic:
                            actual=oracle.call(op='play',actions=actions)
                            collisions+=not actual['advanced']
                            game.load(actual)
                        else:
                            collisions+=not game.step(actions)
                        x,y=game.positions[seat]
                        if (x,y)==old_position:
                            stationary+=1
                        elif old_board[y,x]==-1:
                            new_moves+=1
                        else:
                            own_moves+=1
                        decisions+=1
                    difference=game.scores[seat]-game.scores[1-seat]
                    row=dict(opponent=opponent,seed=int(seed),seat=seat,finished=bool(game.ended),
                             outcome=int(np.sign(difference)) if game.ended else None,
                             scores=game.scores,turns=game.turn,decisions=decisions,
                             collisions=collisions,new_cell_moves=new_moves,own_cell_moves=own_moves,
                             stationary=stationary,seconds=time.perf_counter()-start)
                    row['decision_mean_ms']=1000*float(np.mean(decision_seconds)) if decision_seconds else 0.
                    row['decision_count']=len(decision_seconds)
                    row['decision_p95_ms']=1000*float(np.percentile(decision_seconds,95)) if decision_seconds else 0.
                    results.append(row)
                    if not quiet:
                        print(json.dumps(row),flush=True)
    finally:
        if oracle:
            oracle.close()
    summary={}
    for opponent in opponents:
        rows=[r for r in results if r['opponent']==opponent]
        done=[r for r in rows if r['finished']]
        summary[opponent]=dict(games=len(rows),unfinished=len(rows)-len(done),
            wins=sum(r['outcome']==1 for r in done),draws=sum(r['outcome']==0 for r in done),
            losses=sum(r['outcome']==-1 for r in done),
            score_rate=sum((r['outcome']+1)/2 for r in done)/len(done) if done else None)
    return dict(model=str(model_path),board_size=size,deterministic=deterministic,search_depth=search_depth,no_policy_prior=no_policy_prior,opening_steps=opening_steps,summary=summary,matches=results,
                note='Small development sample, not strength certification. Both seats; fixed held-out seeds.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--opponents',nargs='+',default=['random','greedy'])
    parser.add_argument('--seeds',type=int,default=2)
    parser.add_argument('--seed-start',type=int,default=9000)
    parser.add_argument('--deterministic',action='store_true')
    parser.add_argument('--quiet',action='store_true')
    parser.add_argument('--output',default='training/runs/evaluation.json')
    args=parser.parse_args()
    report=evaluate(args.model,args.opponents,range(args.seed_start,args.seed_start+args.seeds),args.deterministic,args.quiet)
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],indent=2))
