"""Original JavaScript teacher trajectories and masked behavior-cloning warm start."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
import torch
from sb3_contrib import MaskablePPO
from training.engine import Game, simple_action
from training.env import TrainingEnv
from training.model import HeadAwareCNN
from training.verify import Oracle, ROOT


def low_load():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if os.name=='nt':
        import ctypes
        kernel=ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        kernel.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
        kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x4000)


def collect(path, episodes, seed):
    source_hash=hashlib.sha256((ROOT/'grid-clash/index.html').read_bytes()).hexdigest()
    verification=json.loads((ROOT/'training/runs/verification15.json').read_text())
    assert verification['passed'] and verification['source_sha256']==source_hash
    obs=[]; masks=[]; labels=[]; groups=[]; values=[]
    oracle=Oracle()
    started=time.perf_counter()
    styles=['apex','bastion','sweep']
    try:
        for episode in range(episodes):
            rng=np.random.default_rng(seed+episode)
            game=Game(size=15)
            game.load(oracle.call(op='reset',size=15,count=2,seed=seed+episode,
                styles=[styles[episode%3],styles[(episode//3)%3]]))
            seats=[]; turns=[]
            decisions=0
            while not game.ended and decisions<4000:
                # Half teacher-v-teacher; otherwise teacher-v-greedy/random.
                teacher_seats=[0,1] if episode%2==0 else [episode%4//2]
                actions=[simple_action(game,p,rng,episode%3!=0) for p in range(2)]
                prior={p:(game.observation(p),game.mask(p)) for p in teacher_seats}
                for p in teacher_seats:
                    actions[p]='ai'
                actual=oracle.call(op='play',actions=actions)
                for p,(observation,mask) in prior.items():
                    action=actual['actions'][p]
                    if action is not None and mask[int(action)]:
                        obs.append(observation.astype(np.float16))
                        masks.append(mask); labels.append(int(action)); groups.append(episode)
                        seats.append(p); turns.append(game.turn)
                game.load(actual)
                decisions+=1
            assert game.ended, 'Unfinished teacher episode: do not label as a terminal result'
            for p,turn in zip(seats,turns):
                values.append(float(np.sign(game.scores[p]-game.scores[1-p]))*.999**(game.turn-turn))
            assert len(values)==len(labels)
            if (episode+1)%20==0:
                print(json.dumps(dict(episodes=episode+1,samples=len(labels),seconds=round(time.perf_counter()-started,1))),flush=True)
    finally:
        oracle.close()
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,observations=np.array(obs),masks=np.array(masks),actions=np.array(labels),
                        episodes=np.array(groups),values=np.array(values,dtype=np.float32))
    path.with_suffix('.json').write_text(json.dumps(dict(episodes=episodes,samples=len(labels),seed=seed,
        source_sha256=source_hash,seconds=time.perf_counter()-started,board_size=15),indent=2))


def augment(obs,masks,actions,rng):
    k=int(rng.integers(4))
    obs=torch.rot90(obs,k,(-2,-1))
    masks=masks[:,[(i+k)%4 for i in range(4)]]
    actions=(actions-k)%4
    if rng.random()<.5:
        obs=obs.flip(-1)
        masks=masks[:,[0,3,2,1]]
        actions=(-actions)%4
    return obs,masks,actions


def train(data,run,epochs,seed):
    run.mkdir(parents=True,exist_ok=True)
    provenance=json.loads(data.with_suffix('.json').read_text())
    source_hash=hashlib.sha256((ROOT/'grid-clash/index.html').read_bytes()).hexdigest()
    if provenance['source_sha256']!=source_hash:
        raise RuntimeError('Teacher dataset is from different browser rules; recollect it')
    dataset=np.load(data)
    observations=dataset['observations']; masks=dataset['masks']; actions=dataset['actions']
    groups=dataset['episodes']; values=dataset['values']
    # Whole held-out trajectories, never adjacent frames split across train/validation.
    rng=np.random.default_rng(seed)
    # A single 1801-turn stalled game must not dominate dozens of decisive games.
    selected=[]
    for group in np.unique(groups):
        ids=np.flatnonzero(groups==group)
        selected.extend(rng.choice(ids,min(len(ids),256),replace=False))
    selected=np.array(selected)
    unique=np.unique(groups)
    held_out=np.random.default_rng(2026).choice(unique,max(1,len(unique)//10),replace=False)
    is_valid=np.isin(groups[selected],held_out)
    train_ids=selected[~is_valid]; valid_ids=selected[is_valid]
    model=MaskablePPO('CnnPolicy',TrainingEnv(size=15),device='cpu',seed=seed,
        n_steps=256,batch_size=128,n_epochs=4,learning_rate=1e-4,
        gamma=.999,gae_lambda=.98,ent_coef=.005,target_kl=.02,
        policy_kwargs=dict(features_extractor_class=HeadAwareCNN,normalize_images=False,
                           net_arch=dict(pi=[64],vf=[64])))
    model.save(run/'untrained.zip')
    best=float('inf'); history=[]; started=time.perf_counter()
    optimizer=torch.optim.Adam(model.policy.parameters(),lr=3e-4)
    for epoch in range(epochs):
        rng.shuffle(train_ids)
        model.policy.set_training_mode(True)
        for start in range(0,len(train_ids),128):
            ids=train_ids[start:start+128]
            x=torch.as_tensor(observations[ids],dtype=torch.float32)
            m=torch.as_tensor(masks[ids]); y=torch.as_tensor(actions[ids])
            x,m,y=augment(x,m,y,rng)
            predicted,logp,_=model.policy.evaluate_actions(x,y,action_masks=m)
            value_loss=(predicted.flatten()-torch.as_tensor(values[ids])).square().mean()
            loss=-logp.mean()+.1*value_loss
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(),.5)
            optimizer.step()
        total=correct=0; nll=0.
        model.policy.set_training_mode(False)
        with torch.no_grad():
            for start in range(0,len(valid_ids),256):
                ids=valid_ids[start:start+256]
                x=torch.as_tensor(observations[ids],dtype=torch.float32)
                dist=model.policy.get_distribution(x,action_masks=masks[ids])
                y=torch.as_tensor(actions[ids])
                nll-=dist.log_prob(y).sum().item()
                correct+=(dist.get_actions(deterministic=True)==y).sum().item(); total+=len(ids)
        row=dict(epoch=epoch+1,val_nll=nll/total,val_accuracy=correct/total,
                 seconds=round(time.perf_counter()-started,1))
        history.append(row); print(json.dumps(row),flush=True)
        if row['val_nll']<best:
            best=row['val_nll']; model.save(run/'bc_best.zip')
        model.save(run/'bc_latest.zip')
        (run/'imitation_report.json').write_text(json.dumps(dict(seed=seed,dataset=str(data),
            train_samples=len(train_ids),validation_samples=len(valid_ids),history=history,
            parameters=sum(p.numel() for p in model.policy.parameters())),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['collect','train'])
    parser.add_argument('--data',type=Path,default=Path('training/runs/teacher15.npz'))
    parser.add_argument('--run',type=Path,default=Path('training/runs/imitate15'))
    parser.add_argument('--episodes',type=int,default=300)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args(); low_load()
    if args.stage=='collect':
        collect(args.data,args.episodes,args.seed)
    else:
        train(args.data,args.run,args.epochs,args.seed)
