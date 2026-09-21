"""Three-player policy/value model and simultaneous, private-copy rollout search."""
import copy
import numpy as np
import torch
from torch import nn


def encode(game, player):
    assert game.count == 3
    n=game.n;order=[player,(player+1)%3,(player+2)%3]
    x=np.zeros((14,n,n),np.float32)
    for slot,owner in enumerate(order):
        x[slot]=game.board==owner
        px,py=game.positions[owner];x[3+slot,py,px]=1
    x[6]=game.board==-1
    for px,py in game.blocks:x[7,py,px]=1
    x[8].fill(max(0,1-game.turn/(8*n*n+1)))
    x[9].fill(min(1,game.idle_attempts/(2*n*n)))
    x[10].fill(n/30)
    x[11]=np.linspace(-1,1,n)[None,:];x[12]=np.linspace(-1,1,n)[:,None]
    # Search retains the exact history; this scalar is only a value-network feature.
    x[13].fill(len(game.tail_history)/96)
    return x


class Residual(nn.Module):
    def __init__(self):
        super().__init__();self.layers=nn.Sequential(nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,32,3,padding=1))
    def forward(self,x):return torch.relu(x+self.layers(x))


class PolicyValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(14,32,3,padding=1),nn.ReLU(),Residual(),Residual())
        self.pool=nn.AdaptiveAvgPool2d((4,4))
        self.hidden=nn.Sequential(nn.Linear(32*19,128),nn.ReLU())
        self.policy=nn.Linear(128,4);self.value=nn.Linear(128,3);self.territory=nn.Linear(128,3)
    def forward(self,x):
        f=self.body(x)
        heads=[(f*x[:,3+i:4+i]).sum((2,3)) for i in range(3)]
        h=self.hidden(torch.cat([self.pool(f).flatten(1),*heads],1))
        return self.policy(h),self.value(h),self.territory(h)


def load(path):
    model=PolicyValue();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));return model.eval()


def augment(obs,policy,mask,value,territory,rng):
    obs=obs.copy();policy=policy.copy();mask=mask.copy();value=value.copy();territory=territory.copy()
    if rng.random()<.5:
        obs[[1,2]]=obs[[2,1]];obs[[4,5]]=obs[[5,4]]
        value[[1,2]]=value[[2,1]];territory[[1,2]]=territory[[2,1]]
    k=int(rng.integers(4));obs=np.rot90(obs,k,axes=(1,2)).copy()
    policy=np.roll(policy,-k);mask=np.roll(mask,-k)
    if rng.random()<.5:
        obs=obs[:,:,::-1].copy();policy=policy[[0,3,2,1]];mask=mask[[0,3,2,1]]
    # Coordinates describe the transformed board, not the original orientation.
    n=obs.shape[-1];obs[11]=np.linspace(-1,1,n)[None,:];obs[12]=np.linspace(-1,1,n)[:,None]
    return obs,policy,mask,value,territory


def fork(game):
    result=copy.copy(game);result.board=game.board.copy();result.positions=game.positions.copy()
    result.blocks=game.blocks.copy();result.tail_history=game.tail_history.copy();return result


def winners(game):
    values=np.array(game.scores)==max(game.scores);return values/values.sum()


@torch.no_grad()
def infer(model,games,players):
    return [x.numpy() for x in model(torch.from_numpy(np.stack([encode(g,p) for g,p in zip(games,players)])))]


def probabilities(logits,mask):
    if not mask.any():return np.zeros(4)
    values=np.where(mask,logits,-1e9);values=np.exp(values-values.max())*mask
    return values/values.sum()


def sample_action(probs,u):
    return None if probs.sum()==0 else min(3,int(np.searchsorted(np.cumsum(probs),u,side='right')))


@torch.no_grad()
def choose(model,game,player,rng,depth=3,samples=4):
    legal=np.flatnonzero(game.mask(player));target=np.zeros(4,np.float32)
    if not len(legal):return None,target
    if len(legal)==1:target[legal[0]]=1;return int(legal[0]),target
    root_logits,_,_=infer(model,[game]*3,list(range(3)))
    policies=[probabilities(root_logits[p],game.mask(p)) for p in range(3)]
    if depth==0:
        target[:]=policies[player];return int(np.argmax(target)),target
    # Common random numbers compare own actions against identical enemy draws.
    draws=rng.random((samples,depth,3));branches=[];owners=[]
    for d in legal:
        for sample in range(samples):
            g=fork(game);actions=[]
            for p in range(3):
                mask=game.mask(p);uniform=mask/max(1,mask.sum())
                actions.append(int(d) if p==player else sample_action(.8*policies[p]+.2*uniform,draws[sample,0,p]))
            g.step(actions);branches.append(g);owners.append((int(d),sample))
    for step in range(1,depth):
        active=[i for i,g in enumerate(branches) if not g.ended]
        if not active:break
        inputs=[branches[i] for i in active for _ in range(3)]
        output,_,_=infer(model,inputs,list(range(3))*len(active))
        for j,i in enumerate(active):
            g=branches[i];sample=owners[i][1]
            actions=[sample_action(probabilities(output[j*3+p],g.mask(p)),draws[sample,step,p]) for p in range(3)]
            g.step(actions)
    _,values,_=infer(model,branches,[player]*len(branches))
    values=torch.softmax(torch.from_numpy(values),1).numpy()[:,0]
    for i,g in enumerate(branches):
        if g.ended:values[i]=winners(g)[player]
    q=np.array([values[[i for i,(move,_) in enumerate(owners) if move==d]].mean() for d in legal])
    scores=q+.01*np.log(np.maximum(policies[player][legal],1e-8))
    weights=np.exp((scores-scores.max())/.05);target[legal]=weights/weights.sum()
    return int(legal[np.argmax(scores)]),target
