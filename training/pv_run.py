"""Bounded CPU pilot: teacher warm-up, search self-play, training and gated evaluation."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import torch
from training.pv import PolicyValue,encode,augment,load,choose,winners,fork
from training.engine import Game,simple_action
from training.verify import Oracle,ROOT
from training.imitate import low_load


class GateRejected(Exception):
    pass


def cannot_reach_gate(wins,played,planned):
    assert 0<=wins<=played<=planned
    return wins+planned-played<(planned*7+9)//10


def manifest(args):
    files=['training/pv.py','training/pv_run.py','training/engine.py','training/oracle.cjs','grid-clash/index.html']
    return dict(arguments=vars(args),sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files})


def collect(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    (output/'manifest.json').write_text(json.dumps(manifest(args),indent=2),encoding='utf-8')
    model=load(args.model) if args.model else None
    rng=np.random.default_rng(args.seed);oracle=Oracle();start=time.perf_counter()
    data={k:[] for k in ['obs','policy','mask','value','territory','game']};rows=[]
    try:
        for episode in range(args.games):
            if (output/'PAUSE').exists():break
            styles=rng.choice(['apex','bastion','sweep'],3).tolist()
            g=Game(3,args.size);g.load(oracle.call(op='reset',size=args.size,count=3,seed=args.seed+episode,styles=styles))
            if episode%4==3:
                for _ in range(4):g.load(oracle.call(op='play',actions=[simple_action(g,p,rng) for p in range(3)]))
            history=[];attempts=0;last_report=time.perf_counter()
            while not g.ended and attempts<args.size**2*10:
                if model:
                    # All three searches see the same pre-move state, never each other's chosen action.
                    learners=[episode%3] if args.legacy_only else [(episode//3)%3] if args.league and episode%3!=2 else list(range(3))
                    actions=['ai']*3;targets=np.zeros((3,4),np.float32)
                    for p in learners:actions[p],targets[p]=choose(model,g,p,rng,args.depth,args.samples)
                    actual=oracle.call(op='play',actions=actions);actions=actual['actions']
                else:
                    actual=oracle.call(op='play',actions=['ai']*3);actions=actual['actions']
                    targets=np.zeros((3,4),np.float32)
                    for p,action in enumerate(actions):
                        if action is not None:targets[p,action]=1
                history.append((fork(g),targets))
                g.step(actions);attempts+=1
                for key,value in g.state().items():assert actual[key]==value,key
                if time.perf_counter()-last_report>20:
                    print(json.dumps(dict(game=episode,attempts=attempts,seconds=round(time.perf_counter()-start,1))),flush=True);last_report=time.perf_counter()
            assert g.ended,'Never label truncated games as completed training outcomes'
            z=winners(g);territory=np.array(g.scores)/sum(g.scores)
            for index in np.unique(np.linspace(0,len(history)-1,min(16,len(history)),dtype=int)):
                state,targets=history[index]
                for p in range(3):
                    order=[p,(p+1)%3,(p+2)%3]
                    for key,value in dict(obs=encode(state,p).astype(np.float16),policy=targets[p],mask=state.mask(p),value=z[order],territory=territory[order],game=episode).items():data[key].append(value)
            rows.append(dict(game=episode,seed=args.seed+episode,styles=styles,scores=g.scores,attempts=attempts))
            if episode%5==0:print(json.dumps(dict(games=episode+1,samples=len(data['obs']),seconds=round(time.perf_counter()-start,1))),flush=True)
    finally:oracle.close()
    np.savez_compressed(output/'data.npz',**{k:np.array(v) for k,v in data.items()})
    (output/'games.json').write_text(json.dumps(rows),encoding='utf-8')
    print('Collected',len(rows),'games',len(data['obs']),'positions',flush=True)


def batch(data,indices,rng,training):
    fields=['obs','policy','mask','value','territory']
    if training:
        rows=[augment(*[data[k][i] for k in fields],rng) for i in indices]
        arrays=[np.stack([row[j] for row in rows]) for j in range(5)]
    else:arrays=[data[k][indices] for k in fields]
    return [torch.from_numpy(a.astype(np.float32)) for a in arrays]


def losses(model,values):
    x,pi,mask,z,territory=values;policy,value,area=model(x)
    policy=policy.masked_fill(mask==0,-1e9)
    eligible=pi.sum(1)>0
    ploss=-(pi*torch.log_softmax(policy,1)).sum(1)
    ploss=ploss[eligible].mean() if eligible.any() else policy.sum()*0
    vloss=-(z*torch.log_softmax(value,1)).sum(1).mean()
    aloss=-(territory*torch.log_softmax(area,1)).sum(1).mean()
    brier=((torch.softmax(value,1)-z)**2).sum(1).mean()
    return ploss+vloss+.2*aloss,ploss,vloss,brier


def train(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    info=manifest(args);info['datasets']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.data}
    (output/'manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    model=load(args.model) if args.model else PolicyValue()
    datasets=[dict(np.load(p)) for p in args.data]
    for data in datasets:
        assert data['obs'].shape[1]==14 and data['obs'].shape[-1] in (15,30)
        assert (data['game']%5==0).any() and (data['game']%5!=0).any(),'Need separate complete training and validation games'
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    best=float('inf');best_sizes={};start=time.perf_counter();last_report=start
    with (output/'metrics.jsonl').open('w') as stream:
        for epoch in range(args.epochs):
            if (output/'PAUSE').exists():break
            model.train();batches=[]
            for j,data in enumerate(datasets):
                indices=np.flatnonzero(data['game']%5!=0);rng.shuffle(indices)
                batches.extend((j,indices[start:start+64]) for start in range(0,len(indices),64))
            if args.batches:
                # Equal dataset sampling keeps small new search buffers relevant,
                # while retaining old teacher trajectories against forgetting.
                batches=[]
                for _ in range(args.batches):
                    j=int(rng.integers(len(datasets)));indices=np.flatnonzero(datasets[j]['game']%5!=0)
                    batches.append((j,rng.choice(indices,64,replace=len(indices)<64)))
            rng.shuffle(batches)
            for j,indices in batches:
                optimizer.zero_grad();loss=losses(model,batch(datasets[j],indices,rng,True))[0]
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                if time.perf_counter()-last_report>20:
                    print(json.dumps(dict(training_epoch=epoch+1,seconds=round(time.perf_counter()-start,1))),flush=True);last_report=time.perf_counter()
            model.eval();metrics=[]
            with torch.no_grad():
                for data in datasets:
                    indices=np.flatnonzero(data['game']%5==0)
                    sums=np.zeros(4);count=0
                    for offset in range(0,len(indices),64):
                        selected=indices[offset:offset+64];results=losses(model,batch(data,selected,rng,False))
                        sums+=np.array([v.item() for v in results])*len(selected);count+=len(selected)
                    metrics.append((sums/count).tolist())
            validation=float(np.mean([v[0] for v in metrics]))
            row=dict(epoch=epoch+1,validation=metrics,seconds=round(time.perf_counter()-start,1))
            stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
            if validation<best:best=validation;torch.save(model.state_dict(),output/'best.pt')
            for size in {data['obs'].shape[-1] for data in datasets}:
                score=float(np.mean([v[0] for data,v in zip(datasets,metrics) if data['obs'].shape[-1]==size]))
                if score<best_sizes.get(size,float('inf')):
                    best_sizes[size]=score;torch.save(model.state_dict(),output/f'best-{size}.pt')
            torch.save(model.state_dict(),output/'latest.pt')


def evaluate(args):
    from sb3_contrib import MaskablePPO
    from training.three_player import logits,BASE
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    info=manifest(args)
    if args.model:info['model_sha256']=hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    (output/'manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    model=load(args.model) if args.model else None
    old=MaskablePPO.load('training/runs/atlas3-round1/snapshot_000016384.zip' if args.size==15 else BASE,device='cpu')
    oracle=Oracle();start=time.perf_counter();rows=[]
    planned=args.games*27
    pairs=list(itertools.product(['apex','bastion','sweep'],repeat=2))
    try:
        with (output/'matches.jsonl').open('w') as stream:
            for seed in range(args.seed,args.seed+args.games):
                for pair in pairs:
                    for seat in range(3):
                        styles=list(pair);styles.insert(seat,'atlas')
                        g=Game(3,args.size);g.load(oracle.call(op='reset',size=args.size,count=3,seed=seed,styles=styles))
                        rng=np.random.default_rng(seed*3+seat);attempts=0;search_seconds=0
                        for _ in range(args.prefix):g.load(oracle.call(op='play',actions=[simple_action(g,p,rng) for p in range(3)]))
                        while not g.ended and attempts<args.size**2*10:
                            if model:
                                before=time.perf_counter();action,_=choose(model,g,seat,rng,args.depth,args.samples)
                                search_seconds+=time.perf_counter()-before
                                actions=['ai']*3;actions[seat]=action
                                actual=oracle.call(op='play',actions=actions)
                            else:actual=oracle.call(op='atlas',logits=logits(old,g).tolist())
                            g.step(actual['actions']);attempts+=1
                            for key,value in g.state().items():assert actual[key]==value,key
                        diff=g.scores[seat]-max(s for p,s in enumerate(g.scores) if p!=seat)
                        row=dict(seed=seed,pair=pair,seat=seat,scores=g.scores,win=g.ended and diff>0,tie=g.ended and diff==0,finished=g.ended,attempts=attempts,search_seconds=search_seconds)
                        rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                        if args.gate_stop and (not g.ended or cannot_reach_gate(sum(r['win'] for r in rows),len(rows),planned)):
                            raise GateRejected('Cannot reach the predeclared 70% screen even if all remaining games win')
                    print(json.dumps(dict(games=len(rows),wins=sum(r['win'] for r in rows),seconds=round(time.perf_counter()-start,1))),flush=True)
    except GateRejected as error:print(str(error),flush=True)
    finally:oracle.close()
    report=dict(games=len(rows),planned=planned,not_run=planned-len(rows),wins=sum(r['win'] for r in rows),ties=sum(r['tie'] for r in rows),unfinished=sum(not r['finished'] for r in rows),seconds=time.perf_counter()-start,release_approved=False)
    (output/'summary.json').write_text(json.dumps(report,indent=2));print(report,flush=True)


def export(args):
    import onnxruntime as ort
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    model=load(args.model);reference=load(args.model);target=output/f'policy-value-{args.size}.onnx'
    class StaticPool(torch.nn.Module):
        def forward(self,x):
            n=args.size
            return torch.cat([torch.cat([x[:,:,y*n//4:((y+1)*n+3)//4,z*n//4:((z+1)*n+3)//4].mean((2,3),keepdim=True) for z in range(4)],3) for y in range(4)],2)
    # ONNX cannot export non-divisible adaptive pooling directly. Exact static
    # slice means preserve its overlapping bin boundaries for 15 and 30.
    model.pool=StaticPool()
    torch.onnx.export(model,torch.zeros(1,14,args.size,args.size),str(target),
                      input_names=['board'],output_names=['policy','value','territory'],
                      dynamic_axes={'board':{0:'batch'},'policy':{0:'batch'},'value':{0:'batch'},'territory':{0:'batch'}},
                      opset_version=17,dynamo=False)
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(target),options,providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(args.seed);game=Game(3,args.size)
    with torch.no_grad():
        for _ in range(12):
            x=np.stack([encode(game,p) for p in range(3)])
            expected=reference(torch.from_numpy(x));actual=session.run(None,{'board':x})
            for a,b in zip(actual,expected):np.testing.assert_allclose(a,b.numpy(),atol=2e-5,rtol=2e-5)
            game.step([simple_action(game,p,rng) for p in range(3)])
    info=manifest(args);info.update(model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),onnx_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),parity_passed=True,release_approved=False)
    (output/'manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print('Local export parity passed; NOT approved or copied to the website.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','train','evaluate','export'])
    p.add_argument('--size',type=int,choices=[15,30],default=15);p.add_argument('--games',type=int,default=1)
    p.add_argument('--seed',type=int,default=800001);p.add_argument('--output',required=True);p.add_argument('--model')
    p.add_argument('--depth',type=int,default=3);p.add_argument('--samples',type=int,default=4)
    p.add_argument('--data',nargs='+');p.add_argument('--epochs',type=int,default=6);p.add_argument('--lr',type=float,default=.0003)
    p.add_argument('--league',action='store_true')
    p.add_argument('--legacy-only',action='store_true',help='Keep two stable legacy opponents while bootstrapping a weak search agent')
    p.add_argument('--prefix',type=int,default=0)
    p.add_argument('--batches',type=int,default=0)
    p.add_argument('--gate-stop',action='store_true',help='Stop only when the fixed 70-percent screen becomes mathematically impossible')
    args=p.parse_args();low_load()
    assert args.games>0 and args.depth>=0 and args.samples>0
    if args.stage=='train' and (not args.data or args.epochs<1):p.error('Training needs datasets and at least one epoch')
    if args.stage=='export' and not args.model:p.error('Export needs a model')
    globals()[args.stage](args)
