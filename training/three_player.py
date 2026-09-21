"""15x15, exactly three players: frozen-opponent PPO and actual-browser evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from training.engine import Game,simple_action,RULE_VERSION
from training.imitate import low_load
from training.train import Progress
from training.verify import Oracle,ROOT

BASE=ROOT/'training/runs/redesign15/selected_v2.zip'
LINEUPS=[('apex','bastion'),('bastion','sweep'),('sweep','apex'),('atlas','atlas')]


def logits(model,game):
    observation=np.stack([game.observation(i) for i in range(3)])
    with torch.no_grad():
        features=model.policy.extract_features(torch.as_tensor(observation))
        return model.policy.action_net(model.policy.mlp_extractor.forward_actor(features)).numpy()


def margin(game,seat):
    return game.scores[seat]-max(s for i,s in enumerate(game.scores) if i!=seat)


def fingerprint(model):
    return {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [Path(model).resolve(),ROOT/'grid-clash/index.html',ROOT/'training/engine.py',ROOT/'training/oracle.cjs',Path(__file__).resolve()]}


class ThreePlayerEnv(gym.Env):
    def __init__(self,pool):
        self.observation_space=gym.spaces.Box(0,1,(7,15,15),np.float32)
        self.action_space=gym.spaces.Discrete(4)
        self.game=Game(count=3,size=15);self.oracle=Oracle();self.pool=Path(pool)
        self.base=MaskablePPO.load(BASE,device='cpu');self.league=None;self.league_path=None

    def action_masks(self):return self.game.mask(self.seat)

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.seat=int(self.np_random.integers(3));draw=float(self.np_random.random())
        # Two-thirds old heuristics, one-third full frozen ATLAS. Old snapshots
        # can replace the frozen actor but never change weights within an episode.
        self.pair=LINEUPS[int(self.np_random.choice(4,p=[2/9,2/9,2/9,1/3]))]
        self.opponent=self.base;self.opponent_name='+'.join(self.pair)
        snapshots=sorted(self.pool.glob('snapshot_*.zip'))
        if self.pair==('atlas','atlas') and snapshots and draw<.5:
            path=str(self.np_random.choice(snapshots))
            if path!=self.league_path:
                self.league=MaskablePPO.load(path,device='cpu');self.league_path=path
            self.opponent=self.league;self.opponent_name=Path(path).stem
        styles=list(self.pair);styles.insert(self.seat,'atlas')
        self.game.load(self.oracle.call(op='reset',size=15,count=3,
            seed=int(self.np_random.integers(2**31)),styles=styles))
        for _ in range(int(self.np_random.choice([0,4,8]))):
            self.game.load(self.oracle.call(op='play',actions=[simple_action(self.game,i,self.np_random) for i in range(3)]))
        self.decisions=0
        assert self.action_masks().any(),'Opening must give the learner a legal move'
        return self.game.observation(self.seat),{}

    def advance(self,action):
        # Learner action is fixed before opponents choose; their search cannot see it.
        prior=logits(self.opponent,self.game) if 'atlas' in self.pair else np.zeros((3,4))
        actual=self.oracle.call(op='atlas',logits=prior.tolist(),overrides={str(self.seat):action})
        self.game.step(actual['actions'])
        for key,value in self.game.state().items():assert actual[key]==value,(key,self.decisions)
        self.decisions+=1;self.last_reason=actual['endReason']

    def step(self,action):
        before=margin(self.game,self.seat);self.advance(int(action))
        while not self.game.ended and not self.action_masks().any() and self.decisions<3600:self.advance(None)
        difference=margin(self.game,self.seat)
        reward=.1*(difference-before)/225
        if self.game.ended:reward+=float(np.sign(difference))
        truncated=not self.game.ended and self.decisions>=3600
        info={}
        if self.game.ended or truncated:
            info=dict(scores=self.game.scores,player=self.seat,outcome=int(np.sign(difference)) if self.game.ended else None,
                      opponent=self.opponent_name,turns=self.game.turn,decisions=self.decisions,reason=self.last_reason)
        return self.game.observation(self.seat),reward,self.game.ended,truncated,info

    def close(self):self.oracle.close()


def evaluate(model_path,output,seeds,seed):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    selected=random.Random(seed).sample(range(2**31),seeds)
    manifest=dict(model=str(model_path),seeds=selected,prefix=4,players=3,size=15,
                  lineups=LINEUPS,fingerprints=fingerprint(model_path),baseline_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest())
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    model=MaskablePPO.load(model_path,device='cpu');base=MaskablePPO.load(BASE,device='cpu')
    oracle=Oracle();rows=[];start=time.perf_counter()
    try:
        with (output/'matches.jsonl').open('w',encoding='utf-8') as stream:
            for pair in LINEUPS:
                for value in selected:
                    for seat in range(3):
                        styles=list(pair);styles.insert(seat,'atlas')
                        game=Game(count=3,size=15);game.load(oracle.call(op='reset',size=15,count=3,seed=value,styles=styles))
                        rng=np.random.default_rng(value)
                        for _ in range(4):game.load(oracle.call(op='play',actions=[simple_action(game,i,rng) for i in range(3)]))
                        trace=[];max_idle=0
                        while not game.ended and len(trace)<3600:
                            prior=logits(model,game)
                            # Candidate plans with its own model for all possible enemy moves,
                            # but actual frozen opponents plan using their frozen model.
                            actual=oracle.call(op='match3',logits=prior.tolist(),opponent_logits=logits(base,game).tolist(),seat=seat)
                            game.step(actual['actions'])
                            for key,item in game.state().items():assert actual[key]==item,(pair,value,seat,key)
                            trace.append(actual['actions']);max_idle=max(max_idle,game.idle_attempts)
                        diff=margin(game,seat)
                        row=dict(seed=value,seat=seat,lineup='+'.join(pair),scores=game.scores,margin=diff,
                                 outcome=int(np.sign(diff)) if game.ended else None,finished=game.ended,
                                 turns=game.turn,decisions=len(trace),max_idle=max_idle,reason=actual['endReason'],actions=trace)
                        rows.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                print(json.dumps(dict(lineup=pair,games=len(rows),seconds=round(time.perf_counter()-start,1))),flush=True)
    finally:oracle.close()
    report=dict(games=len(rows),wins=sum(r['outcome']==1 for r in rows),ties_first=sum(r['outcome']==0 for r in rows),
                losses=sum(r['outcome']==-1 for r in rows),unfinished=sum(not r['finished'] for r in rows),
                seconds=time.perf_counter()-start)
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(report,flush=True)


def train(run,steps,seed):
    run=Path(run);run.mkdir(parents=True,exist_ok=False)
    verification=json.loads((ROOT/'training/runs/verification15.json').read_text())
    assert verification['source_sha256']==hashlib.sha256((ROOT/'grid-clash/index.html').read_bytes()).hexdigest() and verification['passed']
    config=dict(players=3,size=15,steps=steps,seed=seed,rule_version=RULE_VERSION,base=str(BASE),
                fingerprints=fingerprint(BASE),algorithm='MaskablePPO warm start, frozen-opponent league',
                threads=1,learning_rate=3e-5,gamma=.999,gae_lambda=.98,shaping=.1,terminal='first +1, tied first 0, below first -1')
    (run/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    env=Monitor(ThreePlayerEnv(run))
    model=MaskablePPO.load(BASE,env=env,device='cpu',learning_rate=3e-5,seed=seed,
                           n_steps=256,batch_size=128,n_epochs=4,ent_coef=.01,target_kl=.02,gamma=.999,gae_lambda=.98)
    from stable_baselines3.common.logger import configure
    model.set_logger(configure(str(run/'metrics'),['csv']))
    callback=Progress(run,interval=16384);start=time.perf_counter()
    try:model.learn(total_timesteps=steps,callback=callback,reset_num_timesteps=True)
    finally:model.save(run/'latest.zip');env.close()
    (run/'training_report.json').write_text(json.dumps(dict(steps=model.num_timesteps,episodes=callback.episodes,
         seconds=time.perf_counter()-start,paused=callback.pause_requested),indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','evaluate','test'])
    parser.add_argument('--run',default='training/runs/atlas3-round1');parser.add_argument('--steps',type=int,default=32768)
    parser.add_argument('--model',default=str(BASE));parser.add_argument('--output');parser.add_argument('--seeds',type=int,default=3)
    parser.add_argument('--seed',type=int,default=310001)
    args=parser.parse_args();low_load()
    if args.stage=='train':train(args.run,args.steps,args.seed)
    elif args.stage=='evaluate':evaluate(args.model,args.output,args.seeds,args.seed)
    else:
        env=ThreePlayerEnv(Path(args.run));obs,_=env.reset(seed=33)
        assert obs[3].sum()==1 and obs[4].sum()==2
        for _ in range(3600):
            legal=np.flatnonzero(env.action_masks());assert len(legal)
            obs,reward,done,truncated,info=env.step(int(legal[0]))
            assert np.isfinite(reward)
            if done or truncated:break
        env.close();assert done and not truncated,info
        print('Three-player observations, masks, forced waits, full episode and JS/Python parity passed',info)
