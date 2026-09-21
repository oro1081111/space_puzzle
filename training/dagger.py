"""On-policy teacher queries; matched-compute replay control and DAgger update."""
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
from training.engine import Game
from training.imitate import low_load, augment
from training.verify import Oracle, ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(model_path,path,episodes,seed):
    verification=json.loads((ROOT/'training/runs/verification15.json').read_text())
    assert verification['passed'] and verification['source_sha256']==digest(ROOT/'grid-clash/index.html')
    model=MaskablePPO.load(model_path,device='cpu')
    assert model.observation_space.shape==(7,15,15)
    oracle=Oracle(); observations=[]; masks=[]; labels=[]; groups=[]; matches=[]
    start=time.perf_counter()
    try:
        for episode in range(episodes):
            opponent=['apex','bastion','sweep'][(episode//2)%3]; seat=episode%2
            torch.manual_seed(seed+episode//2)
            game=Game(size=15)
            game.load(oracle.call(op='reset',size=15,count=2,seed=seed+episode//2,
                                  styles=['apex' if i==seat else opponent for i in range(2)]))
            episode_obs=[]; episode_masks=[]; episode_labels=[]
            decisions=choices=agreed=sampled_agreed=0; nll=0.; collisions=0
            while not game.ended and decisions<4000:
                mask=game.mask(seat); action=None
                if mask.any():
                    obs=game.observation(seat)
                    with torch.no_grad():
                        dist=model.policy.get_distribution(torch.as_tensor(obs[None]),action_masks=mask[None])
                        action=int(dist.get_actions(deterministic=False)[0])
                        top=int(dist.get_actions(deterministic=True)[0])
                    # No opponent proposal is supplied or inspected by the teacher.
                    teacher=oracle.call(op='advise',player=seat,style='apex')['action']
                    if teacher is not None:
                        assert mask[teacher]
                        episode_obs.append(obs.astype(np.float16)); episode_masks.append(mask)
                        episode_labels.append(teacher)
                        if mask.sum()>1:
                            choices+=1; agreed+=top==teacher; sampled_agreed+=action==teacher
                            nll-=float(dist.log_prob(torch.tensor([teacher]))[0])
                actions=['ai','ai']; actions[seat]=action
                actual=oracle.call(op='play',actions=actions)
                collisions+=not actual['advanced']; game.load(actual); decisions+=1
            if not game.ended:raise RuntimeError('Unfinished trajectory: refuse to use it')
            selected=np.random.default_rng(seed+episode).choice(len(episode_labels),
                min(256,len(episode_labels)),replace=False)
            observations.extend(episode_obs[i] for i in selected)
            masks.extend(episode_masks[i] for i in selected); labels.extend(episode_labels[i] for i in selected)
            groups.extend([episode]*len(selected))
            matches.append(dict(episode=episode,opponent=opponent,seat=seat,seed=seed+episode//2,
                outcome=int(np.sign(game.scores[seat]-game.scores[1-seat])),scores=game.scores,
                turns=game.turn,decisions=decisions,collisions=collisions,choices=choices,
                greedy_agreement=agreed,sampled_agreement=sampled_agreed,teacher_nll=nll))
            if (episode+1)%20==0:
                print(json.dumps(dict(episodes=episode+1,samples=len(labels),seconds=round(time.perf_counter()-start,1))),flush=True)
    finally:
        oracle.close()
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,observations=np.array(observations),masks=np.array(masks),
                        actions=np.array(labels),episodes=np.array(groups))
    choices=sum(m['choices'] for m in matches)
    report=dict(model=str(model_path),model_sha256=digest(model_path),source_sha256=digest(ROOT/'grid-clash/index.html'),
        teacher='apex',episodes=episodes,seed=seed,samples=len(labels),seconds=time.perf_counter()-start,
        nonforced_choices=choices,greedy_agreement=sum(m['greedy_agreement'] for m in matches)/choices,
        teacher_nll=sum(m['teacher_nll'] for m in matches)/choices,matches=matches)
    path.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='matches'}),flush=True)


def load_split(path,paired=False):
    with np.load(path) as data:
        groups=data['episodes']//2 if paired else data['episodes']
        unique=np.unique(groups)
        held_out=np.random.default_rng(2026).choice(unique,max(1,len(unique)//10),replace=False)
        rng=np.random.default_rng(42); selected=[]
        for group in unique:
            ids=np.flatnonzero(groups==group)
            selected.extend(rng.choice(ids,min(256,len(ids)),replace=False))
        selected=np.array(selected); valid=np.isin(groups[selected],held_out)
        arrays=tuple(data[key][selected] for key in ['observations','masks','actions'])
        return tuple(x[~valid] for x in arrays),tuple(x[valid] for x in arrays)


def validate(model,data):
    obs,masks,labels=data; correct=choice_correct=choices=0; nll=0.
    with torch.no_grad():
        for start in range(0,len(labels),256):
            x=torch.as_tensor(obs[start:start+256],dtype=torch.float32)
            y=torch.as_tensor(labels[start:start+256])
            dist=model.policy.get_distribution(x,action_masks=masks[start:start+256])
            hits=dist.get_actions(deterministic=True)==y
            nonforced=torch.as_tensor(masks[start:start+256].sum(axis=1)>1)
            correct+=hits.sum().item();choice_correct+=hits[nonforced].sum().item()
            choices+=nonforced.sum().item()
            nll-=dist.log_prob(y).sum().item()
    return dict(samples=len(labels),nll=nll/len(labels),accuracy=correct/len(labels),
                choice_accuracy=choice_correct/choices if choices else None)


def train(model_path,paths,run,updates,seed,control):
    run.mkdir(parents=True,exist_ok=True)
    source_hash=digest(ROOT/'grid-clash/index.html')
    original=Path('training/runs/teacher15.npz')
    for path in [original,*paths]:
        if json.loads(path.with_suffix('.json').read_text())['source_sha256']!=source_hash:
            raise RuntimeError('Dataset uses different rules')
    old_train,old_valid=load_split(original)
    new_splits=[load_split(path,paired=True) for path in paths]
    new_train=tuple(np.concatenate([pair[0][i] for pair in new_splits]) for i in range(3))
    new_valid=tuple(np.concatenate([pair[1][i] for pair in new_splits]) for i in range(3))
    model=MaskablePPO.load(model_path,device='cpu')
    optimizer=torch.optim.Adam(model.policy.parameters(),lr=2e-5)
    rng=np.random.default_rng(seed); torch.manual_seed(seed)
    history=[dict(update=0,old=validate(model,old_valid),on_policy=validate(model,new_valid))]
    start=time.perf_counter()
    for update in range(1,updates+1):
        batch=[]
        # Same architecture, starting weights, batch size, optimizer and updates for control.
        for data in [old_train,old_train if control else new_train]:
            ids=rng.integers(len(data[2]),size=64)
            batch.append(tuple(x[ids] for x in data))
        obs,masks,labels=(torch.as_tensor(np.concatenate([b[i] for b in batch])) for i in range(3))
        obs,masks,labels=augment(obs.float(),masks,labels,rng)
        dist=model.policy.get_distribution(obs,action_masks=masks)
        loss=-dist.log_prob(labels).mean()
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(),.5);optimizer.step()
        if update%256==0 or update==updates:
            row=dict(update=update,seconds=round(time.perf_counter()-start,1),
                     old=validate(model,old_valid),on_policy=validate(model,new_valid))
            history.append(row); print(json.dumps(row),flush=True)
            model.save(run/f'update_{update:04d}.zip')
            (run/'report.json').write_text(json.dumps(dict(model=str(model_path),model_sha256=digest(model_path),
                data=[str(p) for p in paths],seed=seed,control=control,learning_rate=2e-5,
                updates=update,history=history,source_sha256=source_hash,
                note='Actor-only imitation updates; shared CNN changes, critic not recalibrated.'),indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['collect','train'])
    parser.add_argument('--model',type=Path,default=Path('training/runs/redesign15/selected_v2.zip'))
    parser.add_argument('--data',type=Path,nargs='+',default=[Path('training/runs/dagger15/round1.npz')])
    parser.add_argument('--episodes',type=int,default=180)
    parser.add_argument('--seed',type=int,default=110000)
    parser.add_argument('--run',type=Path,default=Path('training/runs/dagger15/round1'))
    parser.add_argument('--updates',type=int,default=768)
    parser.add_argument('--control',action='store_true')
    args=parser.parse_args();low_load()
    if args.stage=='collect':collect(args.model,args.data[0],args.episodes,args.seed)
    else:train(args.model,args.data,args.run,args.updates,args.seed,args.control)
